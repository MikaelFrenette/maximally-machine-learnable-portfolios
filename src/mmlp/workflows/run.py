"""
End-to-end run workflow for MMLP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mmlp.analysis import summarize_mace_run
from mmlp.config.run import RunConfig
from mmlp.evaluation import apply_mean_variance_overlay
from mmlp.logger import LogLevel, get_logger
from mmlp.model import (
    build_mace_panel_matrix,
    build_marx_features,
    fit_mace,
)
from mmlp.model import (
    build_test_marx_features as build_test_marx_features_from_train_tail,
)
from mmlp.workflows.extract import extract_features_from_run_config
from mmlp.workflows.plotting import generate_plots_from_config

__all__ = ["RunArtifacts", "run_pipeline_from_config"]


@dataclass(slots=True)
class RunArtifacts:
    """
    Materialized output paths from an end-to-end run.

    Parameters
    ----------
    raw_panel_path : pathlib.Path
        Extracted raw panel path.
    feature_panel_path : pathlib.Path
        Preprocessed feature panel path.
    diagnostics_path : pathlib.Path
        Alternating-fit diagnostics path.
    predictions_path : pathlib.Path
        In-sample prediction artifact path.
    weights_raw_path : pathlib.Path
        Raw fitted portfolio weights path.
    weights_normalized_path : pathlib.Path
        Sum-normalized portfolio weights path.
    summary_path : pathlib.Path
        One-row run summary metrics path.
    trading_path : pathlib.Path
        Mean-variance trading return path.
    trading_summary_path : pathlib.Path
        Mean-variance trading summary path.
    trading_yearly_summary_path : pathlib.Path
        Per-year mean-variance trading summary path.
    cumulative_plot_path : pathlib.Path
        Cumulative return plot path.
    yearly_heatmap_path : pathlib.Path
        Yearly Sharpe heatmap plot path.
    log_path : pathlib.Path
        Persistent text log for the run.
    """

    raw_panel_path: Path
    feature_panel_path: Path
    diagnostics_path: Path
    predictions_path: Path
    weights_raw_path: Path
    weights_normalized_path: Path
    summary_path: Path
    trading_path: Path
    trading_summary_path: Path
    trading_yearly_summary_path: Path
    cumulative_plot_path: Path
    yearly_heatmap_path: Path
    log_path: Path


def run_pipeline_from_config(config: RunConfig) -> RunArtifacts:
    """
    Execute extraction, preprocessing, model-input assembly, and training.

    Parameters
    ----------
    config : RunConfig
        Validated top-level run configuration.

    Returns
    -------
    RunArtifacts
        Materialized artifact paths from the run.
    """

    log_path = _resolve_output_path(config=config, filename="run.txt")
    logger = get_logger(
        f"mmlp.{config.run_name}",
        level=LogLevel[config.outputs.log_level],
        verbosity=config.outputs.verbosity,
        log_path=log_path,
    )

    logger.info("Running pipeline: run_name=%s", config.run_name)
    logger.info(
        "Configured date range: %s to %s",
        config.dataset.start_date.isoformat(),
        config.dataset.end_date.isoformat(),
    )
    logger.info(
        "Configured split: train=%s to %s, test=%s to %s",
        config.split.train_start.isoformat(),
        config.split.train_end.isoformat(),
        config.split.test_start.isoformat(),
        config.split.test_end.isoformat(),
    )

    feature_panel_path = extract_features_from_run_config(config=config)
    raw_panel_path = _resolve_output_path(config=config, filename="panel.csv")
    logger.info("Prepared dataset artifacts under %s", raw_panel_path.parent)
    raw_panel = pd.read_csv(raw_panel_path, parse_dates=[config.preprocessing.date_column])
    feature_panel = pd.read_csv(feature_panel_path, parse_dates=[config.preprocessing.date_column])
    _ = feature_panel
    train_panel, test_panel = _split_panel_by_date(panel=raw_panel, config=config)
    train_matrix = build_mace_panel_matrix(panel=train_panel, config=config.preprocessing)
    test_matrix = build_mace_panel_matrix(panel=test_panel, config=config.preprocessing)
    fitted_model = fit_mace(
        panel_matrix=train_matrix,
        preprocessing_config=config.preprocessing,
        model_config=config.model,
        logger=logger,
    )

    diagnostics_path = _resolve_output_path(config=config, filename="diagnostics.csv")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    fitted_model.diagnostics_.to_csv(diagnostics_path, index=False)

    weights_raw = fitted_model.weights_.astype(float)
    weights_normalized = _normalize_weights_sum_to_one(weights_raw)

    prediction_set = _reconstruct_prediction_set(
        fitted_model=fitted_model,
        train_returns=train_matrix.returns,
        test_returns=test_matrix.returns,
        lags=config.preprocessing.lags,
    )
    train_returns = _portfolio_returns_from_weights(train_matrix.returns, weights_normalized)
    test_returns = _portfolio_returns_from_weights(test_matrix.returns, weights_normalized)
    full_portfolio_returns = pd.concat([train_returns, test_returns]).sort_index()
    full_signal_predictions = pd.concat(
        [prediction_set.mace_prediction_train, prediction_set.mace_prediction_test]
    ).sort_index()
    predictions_frame = pd.DataFrame(
        {
            config.preprocessing.date_column: full_portfolio_returns.index,
            "split": ["train"] * len(train_returns) + ["test"] * len(test_returns),
            "portfolio_return": full_portfolio_returns.to_numpy(),
            "signal_prediction": full_signal_predictions.to_numpy(),
            "hy": pd.concat([prediction_set.hy_train, prediction_set.hy_test])
            .sort_index()
            .to_numpy(),
            "hx": pd.concat([prediction_set.hx_train, prediction_set.hx_test])
            .sort_index()
            .to_numpy(),
            "fit": pd.concat(
                [prediction_set.fit_train, prediction_set.fit_test]
            ).sort_index().to_numpy(),
            "z1": pd.concat(
                [fitted_model.z1_, pd.Series(index=test_returns.index, dtype=float)]
            ).to_numpy(),
            "z2": pd.concat(
                [fitted_model.z2_, pd.Series(index=test_returns.index, dtype=float)]
            ).to_numpy(),
        }
    )

    predictions_path = _resolve_output_path(config=config, filename="predictions.csv")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(predictions_path, index=False)

    weights_raw_frame = weights_raw.rename("weight").reset_index()
    weights_raw_frame.columns = [config.preprocessing.id_column, "weight"]
    weights_raw_path = _resolve_output_path(config=config, filename="weights_raw.csv")
    weights_raw_path.parent.mkdir(parents=True, exist_ok=True)
    weights_raw_frame.to_csv(weights_raw_path, index=False)

    weights_normalized_frame = weights_normalized.rename("weight").reset_index()
    weights_normalized_frame.columns = [config.preprocessing.id_column, "weight"]
    weights_normalized_path = _resolve_output_path(
        config=config,
        filename="weights_normalized.csv",
    )
    weights_normalized_path.parent.mkdir(parents=True, exist_ok=True)
    weights_normalized_frame.to_csv(weights_normalized_path, index=False)

    summary_frame = summarize_mace_run(
        fitted_model=fitted_model,
        panel_matrix=train_matrix,
        normalized_weights=weights_normalized,
    )
    summary_path = _resolve_output_path(config=config, filename="summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(summary_path, index=False)

    trading_path = _resolve_output_path(config=config, filename="trading.csv")
    trading_summary_path = _resolve_output_path(config=config, filename="trading_summary.csv")
    trading_yearly_summary_path = _resolve_output_path(
        config=config,
        filename="trading_yearly_summary.csv",
    )
    trading_path.parent.mkdir(parents=True, exist_ok=True)
    trading_summary_path.parent.mkdir(parents=True, exist_ok=True)
    trading_yearly_summary_path.parent.mkdir(parents=True, exist_ok=True)
    if config.trading.enabled:
        trading_result = apply_mean_variance_overlay(
            realized_returns=test_returns.rename("portfolio_return"),
            predicted_returns=prediction_set.mace_prediction_test.rename("mace_signal"),
            in_sample_returns=train_returns.rename("portfolio_return"),
            config=config.trading,
        )
        trading_frame = trading_result.frame.reset_index()
        trading_frame.columns = [config.preprocessing.date_column, *trading_result.frame.columns]
        trading_frame.to_csv(trading_path, index=False)
        trading_result.summary.to_csv(trading_summary_path, index=False)
        trading_result.yearly_summary.to_csv(trading_yearly_summary_path, index=False)
    else:
        pd.DataFrame().to_csv(trading_path, index=False)
        pd.DataFrame().to_csv(trading_summary_path, index=False)
        pd.DataFrame().to_csv(trading_yearly_summary_path, index=False)

    plot_artifacts = generate_plots_from_config(config=config)

    logger.info(
        "Finished pipeline: outputs=%s",
        raw_panel_path.parent,
    )

    return RunArtifacts(
        raw_panel_path=raw_panel_path,
        feature_panel_path=feature_panel_path,
        diagnostics_path=diagnostics_path,
        predictions_path=predictions_path,
        weights_raw_path=weights_raw_path,
        weights_normalized_path=weights_normalized_path,
        summary_path=summary_path,
        trading_path=trading_path,
        trading_summary_path=trading_summary_path,
        trading_yearly_summary_path=trading_yearly_summary_path,
        cumulative_plot_path=plot_artifacts.cumulative_plot_path,
        yearly_heatmap_path=plot_artifacts.yearly_heatmap_path,
        log_path=log_path,
    )

def _resolve_output_path(config: RunConfig, filename: str) -> Path:
    return Path("outputs") / config.run_name / filename


def _split_panel_by_date(
    config: RunConfig,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_column = config.preprocessing.date_column
    train_mask = (
        (panel[date_column] >= pd.Timestamp(config.split.train_start))
        & (panel[date_column] <= pd.Timestamp(config.split.train_end))
    )
    test_mask = (
        (panel[date_column] >= pd.Timestamp(config.split.test_start))
        & (panel[date_column] <= pd.Timestamp(config.split.test_end))
    )
    train_panel = panel.loc[train_mask].copy()
    test_panel = panel.loc[test_mask].copy()
    if train_panel.empty:
        raise ValueError("Train split produced an empty panel.")
    if test_panel.empty:
        raise ValueError("Test split produced an empty panel.")
    return train_panel, test_panel


@dataclass(slots=True)
class _PredictionSet:
    hy_train: pd.Series
    hy_test: pd.Series
    hx_train: pd.Series
    hx_test: pd.Series
    fit_train: pd.Series
    fit_test: pd.Series
    mace_prediction_train: pd.Series
    mace_prediction_test: pd.Series


def _reconstruct_prediction_set(
    fitted_model,
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    lags: int,
) -> _PredictionSet:
    hy_train = fitted_model.portfolio_returns(train_returns).rename("hy_train")
    hy_test = fitted_model.portfolio_returns(test_returns).rename("hy_test")

    hx_train = fitted_model.random_forest_model_.predict(
        build_marx_features(hy_train, lags)
    ).rename("hx_train")
    hx_test = fitted_model.random_forest_model_.predict(
        _build_test_marx_features(hy_train=hy_train, hy_test=hy_test, lags=lags)
    ).rename("hx_test")

    ols_coef, ols_intercept = _fit_signal_alignment(hy_train=hy_train, hx_train=hx_train)
    fit_train = pd.Series(
        ols_intercept + ols_coef * hx_train.to_numpy(dtype=float),
        index=hy_train.index,
        name="fit_train",
    )
    fit_test = pd.Series(
        ols_intercept + ols_coef * hx_test.to_numpy(dtype=float),
        index=hy_test.index,
        name="fit_test",
    )

    raw_weight_sum = float(fitted_model.weights_.sum())
    if abs(raw_weight_sum) <= 1e-12:
        raw_weight_sum = 1.0
    mace_prediction_train = (fit_train / raw_weight_sum).rename("mace_prediction_train")
    mace_prediction_test = (fit_test / raw_weight_sum).rename("mace_prediction_test")
    return _PredictionSet(
        hy_train=hy_train,
        hy_test=hy_test,
        hx_train=hx_train,
        hx_test=hx_test,
        fit_train=fit_train,
        fit_test=fit_test,
        mace_prediction_train=mace_prediction_train,
        mace_prediction_test=mace_prediction_test,
    )


def _build_test_marx_features(
    hy_train: pd.Series,
    hy_test: pd.Series,
    lags: int,
) -> pd.DataFrame:
    return build_test_marx_features_from_train_tail(
        train_series=hy_train,
        test_series=hy_test,
        max_lag=lags,
    )


def _fit_signal_alignment(hy_train: pd.Series, hx_train: pd.Series) -> tuple[float, float]:
    x = hx_train.to_numpy(dtype=float)
    y = hy_train.to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x), dtype=float), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept = float(beta[0])
    slope = float(beta[1])
    return slope, intercept


def _portfolio_returns_from_weights(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    aligned_weights = weights.loc[returns.columns].astype(float)
    return pd.Series(
        returns.to_numpy() @ aligned_weights.to_numpy(),
        index=returns.index,
        name="portfolio_return",
    )


def _normalize_weights_sum_to_one(weights: pd.Series) -> pd.Series:
    weight_sum = float(weights.sum())
    if abs(weight_sum) <= 1e-12:
        return pd.Series(0.0, index=weights.index, name=weights.name, dtype=float)
    return (weights / weight_sum).rename(weights.name)
