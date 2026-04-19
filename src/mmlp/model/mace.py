"""
MACE-specific data structures and training loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from mmlp.config.model import MaceModelConfig
from mmlp.config.preprocessing import PreprocessingConfig
from mmlp.model.backends import resolve_model_backend

__all__ = [
    "FittedMaceModel",
    "MacePanelMatrix",
    "build_mace_panel_matrix",
    "build_marx_features",
    "build_test_marx_features",
    "fit_mace",
]


@dataclass(slots=True)
class MacePanelMatrix:
    """
    Matrix representation of a return panel for MACE training.

    Parameters
    ----------
    dates : tuple of pandas.Timestamp
        Ordered training dates.
    asset_ids : tuple of str
        Ordered asset identifiers.
    returns : pandas.DataFrame
        Wide return matrix indexed by date with one column per asset.
    """

    dates: tuple[pd.Timestamp, ...]
    asset_ids: tuple[str, ...]
    returns: pd.DataFrame


@dataclass(slots=True)
class FittedMaceModel:
    """
    Fitted non-bagged MACE model.

    Parameters
    ----------
    intercept_ : float
        Final portfolio intercept.
    weights_ : pandas.Series
        Final portfolio weights indexed by asset identifier.
    z1_ : pandas.Series
        Final RF-side latent series.
    z2_ : pandas.Series
        Final portfolio-return latent series.
    diagnostics_ : pandas.DataFrame
        Per-iteration diagnostics.
    selected_iteration_ : int
        One-based iteration selected for post-fit outputs.
    selection_rule_ : str
        Name of the post-fit iteration-selection rule.
    selection_metric_: str
        Diagnostics column used to select the final iterate.
    selection_score_: float
        Winning score for the selected iterate under ``selection_metric_``.
    ridge_model_ : object
        Final fitted ridge adapter.
    random_forest_model_ : object
        Final fitted random-forest adapter.
    """

    intercept_: float
    weights_: pd.Series
    z1_: pd.Series
    z2_: pd.Series
    diagnostics_: pd.DataFrame
    selected_iteration_: int
    selection_rule_: str
    selection_metric_: str
    selection_score_: float
    ridge_model_: object
    random_forest_model_: object

    def portfolio_returns(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute portfolio returns from a wide asset-return matrix.
        """

        return pd.Series(
            self.intercept_ + returns.to_numpy() @ self.weights_.loc[returns.columns].to_numpy(),
            index=returns.index,
            name="portfolio_return",
        )


@dataclass(slots=True)
class _IterationCandidate:
    iteration: int
    intercept: float
    weights: pd.Series
    z1: pd.Series
    z2: pd.Series
    ridge_model: object
    random_forest_model: object


def build_mace_panel_matrix(
    panel: pd.DataFrame,
    config: PreprocessingConfig,
) -> MacePanelMatrix:
    """
    Convert a canonical long-format panel into the wide matrix used by MACE.
    """

    required_columns = {
        config.date_column,
        config.id_column,
        config.return_column,
    }
    missing_columns = sorted(required_columns.difference(panel.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Panel is missing required MACE columns: {missing_text}")

    wide_returns = (
        panel.loc[:, [config.date_column, config.id_column, config.return_column]]
        .pivot(index=config.date_column, columns=config.id_column, values=config.return_column)
        .sort_index()
        .sort_index(axis=1)
    )
    wide_returns.index = pd.to_datetime(wide_returns.index)
    wide_returns = wide_returns.apply(pd.to_numeric, errors="coerce").astype(float)
    wide_returns = wide_returns.replace([np.inf, -np.inf], np.nan)
    wide_returns = wide_returns.dropna(axis=0, how="all")
    if wide_returns.empty:
        raise ValueError("Wide MACE return matrix is empty.")

    missing_assets = wide_returns.columns[wide_returns.isna().any(axis=0)].tolist()
    if missing_assets:
        missing_counts = (
            wide_returns[missing_assets].isna().sum(axis=0).sort_values(ascending=False)
        )
        missing_summary = ", ".join(
            f"{asset}({int(count)} missing)" for asset, count in missing_counts.items()
        )
        raise ValueError(
            "Requested asset universe does not have complete return history over the configured "
            f"date range. Incomplete assets: {missing_summary}"
        )

    return MacePanelMatrix(
        dates=tuple(pd.Timestamp(date) for date in wide_returns.index),
        asset_ids=tuple(str(asset_id) for asset_id in wide_returns.columns),
        returns=wide_returns,
    )


def build_marx_features(series: pd.Series, max_lag: int) -> pd.DataFrame:
    """
    Build MARX features from the current portfolio-return series.
    """

    lagged = pd.DataFrame(index=series.index)
    for lag in range(1, max_lag + 1):
        lagged[f"L_{lag}"] = series.shift(lag).fillna(0.0)
    marx = lagged.cumsum(axis=1)
    marx.columns = [f"MARX_{lag}" for lag in range(1, max_lag + 1)]
    return marx


def build_test_marx_features(
    train_series: pd.Series,
    test_series: pd.Series,
    max_lag: int,
) -> pd.DataFrame:
    """
    Build test-set MARX features seeded from the tail of the train series.

    This mirrors the R implementation's intent: leading test lags are filled
    from the most recent in-sample portfolio values before the cumulative MARX
    transform is applied.
    """

    lagged = pd.DataFrame(index=test_series.index)
    for lag in range(1, max_lag + 1):
        shifted = test_series.shift(lag)
        leading_count = min(lag, len(shifted))
        if leading_count > 0:
            seed_values = train_series.tail(lag).to_numpy(dtype=float)[:leading_count]
            shifted.iloc[:leading_count] = seed_values
        lagged[f"L_{lag}"] = shifted.fillna(0.0)
    marx = lagged.cumsum(axis=1)
    marx.columns = [f"MARX_{lag}" for lag in range(1, max_lag + 1)]
    return marx


def fit_mace(
    panel_matrix: MacePanelMatrix,
    preprocessing_config: PreprocessingConfig,
    model_config: MaceModelConfig,
    logger: logging.Logger | None = None,
) -> FittedMaceModel:
    """
    Fit the non-bagged MACE core aligned to the R algorithm structure.
    """

    backend = resolve_model_backend(config=model_config)
    returns = _ensure_finite_frame(panel_matrix.returns)
    n_obs = len(returns)
    rng = np.random.default_rng(model_config.random_state)

    if logger is not None:
        if model_config.stopping_rule == "tolerance":
            logger.info(
                "Starting MACE fit: n_obs=%s n_assets=%s backend=%s stop=tolerance "
                "max_iterations=%s min_iterations=%s tolerance=%.6g",
                n_obs,
                len(panel_matrix.asset_ids),
                backend.name,
                model_config.max_iterations,
                model_config.min_iterations,
                model_config.tolerance,
            )
        else:
            logger.info(
                "Starting MACE fit: n_obs=%s n_assets=%s backend=%s stop=max_iterations "
                "max_iterations=%s",
                n_obs,
                len(panel_matrix.asset_ids),
                backend.name,
                model_config.max_iterations,
            )

    z1 = pd.Series(rng.normal(size=n_obs), index=returns.index, name="z1")
    z2, intercept, weights = _initialize_portfolio_state(
        returns=returns,
        config=model_config,
        rng=rng,
    )
    beta_state = _pack_portfolio_parameters(intercept=intercept, weights=weights)

    diagnostics: list[dict[str, float | int]] = []
    candidates: list[_IterationCandidate] = []
    ridge_model = backend.ridge_factory()
    random_forest_model = backend.random_forest_factory()

    for iteration in range(1, model_config.max_iterations + 1):
        marx_features = build_marx_features(z2, preprocessing_config.lags)

        random_forest_model = backend.random_forest_factory().fit(marx_features, z2)
        oob_prediction_mse = _compute_oob_prediction_mse(
            random_forest_model=random_forest_model,
            target=z2,
            backend_name=backend.name,
        )
        rf_predictions = _ensure_finite_series(
            random_forest_model.predict(marx_features),
            name="rf_prediction",
        )

        if iteration > 1:
            z1 = _ensure_finite_series(
                model_config.learning_rate * rf_predictions
                + (1.0 - model_config.learning_rate) * z1
            )
        else:
            z1 = _ensure_finite_series(
                rf_predictions,
                name="z1",
            )

        ridge_model = backend.ridge_factory().fit(returns, z1)
        selected_ridge_alpha = _extract_ridge_alpha(ridge_model)
        selected_ridge_r2 = _extract_ridge_selected_r2(ridge_model)
        target_ridge_r2 = _extract_ridge_target_r2(ridge_model)
        ridge_r2_gap = abs(selected_ridge_r2 - target_ridge_r2)
        return_ins = pd.Series(
            ridge_model.predict(returns),
            index=returns.index,
            name="return_ins",
        )
        raw_intercept, raw_weights = _extract_ridge_parameters(
            ridge_model=ridge_model,
            asset_ids=panel_matrix.asset_ids,
        )
        zero_weight_solution = bool(np.all(np.abs(raw_weights.to_numpy()) <= 1e-20))

        if zero_weight_solution:
            return_ins_noisy = pd.Series(
                0.001 * rng.normal(size=n_obs) / iteration,
                index=returns.index,
                name="return_ins_noisy",
            )
            updated_z2 = _ensure_finite_series(
                model_config.learning_rate * return_ins_noisy
                + (1.0 - model_config.learning_rate) * z2,
                name="z2",
            )
            beta_target = np.concatenate(([raw_intercept], np.zeros(len(raw_weights), dtype=float)))
        else:
            return_ins_std = float(return_ins.std(ddof=0))
            if return_ins_std <= 1e-12:
                return_ins_noisy = pd.Series(
                    0.001 * rng.normal(size=n_obs) / iteration,
                    index=returns.index,
                    name="return_ins_noisy",
                )
            else:
                return_ins_noisy = _standardize_series(return_ins) + pd.Series(
                    0.001 * rng.normal(size=n_obs) / iteration,
                    index=returns.index,
                )

            mean_returns = returns.mean(axis=1)
            orientation = np.corrcoef(mean_returns.to_numpy(), return_ins_noisy.to_numpy())[0, 1]
            if not np.isnan(orientation) and orientation < 0:
                return_ins_noisy = -return_ins_noisy

            updated_z2 = _standardize_series(
                model_config.learning_rate * return_ins_noisy
                + (1.0 - model_config.learning_rate) * z2
            )
            beta_target = np.concatenate(
                (
                    [raw_intercept - float(return_ins.mean())],
                    raw_weights.to_numpy() / return_ins_std,
                )
            )

        delta = float(np.sqrt(np.mean((updated_z2 - z2) ** 2)))
        z2 = updated_z2

        beta_state = (1.0 - model_config.learning_rate) * beta_state + (
            model_config.learning_rate * beta_target
        )
        intercept, weights = _unpack_portfolio_parameters(
            beta_state=beta_state,
            asset_ids=panel_matrix.asset_ids,
        )
        diagnostics.append(
            {
                "iteration": iteration,
                "latent_delta_rmse": delta,
                "oob_prediction_mse": oob_prediction_mse,
                "ridge_alpha": selected_ridge_alpha,
                "ridge_selected_r2": selected_ridge_r2,
                "ridge_target_r2": target_ridge_r2,
                "ridge_r2_gap": ridge_r2_gap,
                "z1_std": float(z1.std(ddof=0)),
                "z2_std": float(z2.std(ddof=0)),
                "weight_l1": float(np.abs(weights.to_numpy()).sum()),
            }
        )
        candidates.append(
            _IterationCandidate(
                iteration=iteration,
                intercept=intercept,
                weights=weights.copy(),
                z1=z1.copy(),
                z2=z2.copy(),
                ridge_model=ridge_model,
                random_forest_model=random_forest_model,
            )
        )

        if logger is not None and (
            iteration == 1
            or iteration % model_config.log_every_n_iterations == 0
            or iteration == model_config.max_iterations
        ):
            logger.info(
                "MACE iteration=%s delta_rmse=%.6g ridge_alpha=%.6g "
                "ridge_selected_r2=%.6g ridge_target_r2=%.6g ridge_r2_gap=%.6g "
                "weight_l1=%.6g z2_std=%.6g",
                iteration,
                delta,
                selected_ridge_alpha,
                selected_ridge_r2,
                target_ridge_r2,
                ridge_r2_gap,
                float(np.abs(weights.to_numpy()).sum()),
                float(z2.std(ddof=0)),
            )

        if (
            model_config.stopping_rule == "tolerance"
            and iteration >= model_config.min_iterations
            and delta <= model_config.tolerance
        ):
            if logger is not None:
                logger.info(
                    "MACE converged at iteration=%s with delta_rmse=%.6g",
                    iteration,
                    delta,
                )
            break

    if logger is not None and model_config.stopping_rule == "tolerance" and (
        not diagnostics or diagnostics[-1]["latent_delta_rmse"] > model_config.tolerance
    ):
        logger.info(
            "MACE stopped without convergence after iteration=%s final_delta_rmse=%.6g",
            len(diagnostics),
            float(diagnostics[-1]["latent_delta_rmse"]) if diagnostics else float("nan"),
        )

    diagnostics_frame = pd.DataFrame(diagnostics)
    selected_index, selection_metric, selection_score = _select_iteration(
        diagnostics=diagnostics_frame,
        selection_rule=model_config.selection_rule,
        selection_smoothing_window=model_config.selection_smoothing_window,
        backend_name=backend.name,
    )
    selected_candidate = candidates[selected_index]
    diagnostics_frame["smoothed_oob_prediction_mse"] = _compute_smoothed_oob_prediction_mse(
        diagnostics=diagnostics_frame,
        window=model_config.selection_smoothing_window,
    )
    diagnostics_frame["selection_metric"] = selection_metric
    diagnostics_frame["selection_score"] = np.nan
    diagnostics_frame["is_selected"] = (
        diagnostics_frame["iteration"] == selected_candidate.iteration
    )
    diagnostics_frame.loc[
        diagnostics_frame["iteration"] == selected_candidate.iteration,
        "selection_score",
    ] = selection_score

    if logger is not None:
        logger.info(
            "Selected iteration=%s using selection_rule=%s metric=%s score=%.6g",
            selected_candidate.iteration,
            model_config.selection_rule,
            selection_metric,
            selection_score,
        )

    return FittedMaceModel(
        intercept_=selected_candidate.intercept,
        weights_=selected_candidate.weights,
        z1_=selected_candidate.z1,
        z2_=selected_candidate.z2,
        diagnostics_=diagnostics_frame,
        selected_iteration_=selected_candidate.iteration,
        selection_rule_=model_config.selection_rule,
        selection_metric_=selection_metric,
        selection_score_=selection_score,
        ridge_model_=selected_candidate.ridge_model,
        random_forest_model_=selected_candidate.random_forest_model,
    )


def _initialize_portfolio_state(
    returns: pd.DataFrame,
    config: MaceModelConfig,
    rng: np.random.Generator,
) -> tuple[pd.Series, float, pd.Series]:
    covariance_sample = _select_initial_covariance_sample(
        returns=returns,
        sample_fraction=config.rhs_init_cov_sample,
        rng=rng,
    )
    covariance = covariance_sample.cov()
    ones = np.ones(len(covariance.columns), dtype=float)
    covariance_array = covariance.to_numpy(dtype=float)
    if _covariance_needs_shrinkage(covariance_array):
        covariance_array = _apply_initial_covariance_shrinkage(
            returns=covariance_sample,
            shrinkage=config.rhs_init_cov_sample_shrinkage,
        )
    try:
        inverse_covariance_weights = np.linalg.solve(covariance_array, ones)
    except np.linalg.LinAlgError:
        inverse_covariance_weights = np.linalg.pinv(covariance_array) @ ones
    weight_sum = float(inverse_covariance_weights.sum())
    if not np.isfinite(weight_sum) or abs(weight_sum) <= 1e-12:
        raise ValueError(
            "Initial inverse-covariance weights sum to zero; unable to initialize MACE state."
        )
    normalized_weights = inverse_covariance_weights / weight_sum
    portfolio_raw = pd.Series(
        returns.to_numpy() @ normalized_weights,
        index=returns.index,
        name="portfolio_raw",
    )
    portfolio_std = float(portfolio_raw.std(ddof=0))
    if portfolio_std <= 1e-12:
        raise ValueError("Initial portfolio variance is zero; unable to initialize MACE state.")

    z2 = _standardize_series(portfolio_raw)
    intercept = float(-portfolio_raw.mean())
    weights = pd.Series(
        normalized_weights / portfolio_std,
        index=returns.columns,
        name="weight",
    )
    return z2, intercept, weights


def _select_initial_covariance_sample(
    returns: pd.DataFrame,
    sample_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if sample_fraction <= 0.0 or sample_fraction >= 1.0:
        return returns

    n_obs = len(returns)
    sample_size = max(2, int(np.ceil(n_obs * sample_fraction)))
    if sample_size >= n_obs:
        return returns

    # The R code uses block-style sampling here. For initialization we keep the
    # same time-series intent by drawing one contiguous block rather than an
    # iid row sample.
    start = int(rng.integers(0, n_obs - sample_size + 1))
    return returns.iloc[start : start + sample_size]


def _covariance_needs_shrinkage(covariance: np.ndarray) -> bool:
    if not np.isfinite(covariance).all():
        return True
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError:
        return True
    return bool(np.min(eigenvalues) <= 1e-10)


def _apply_initial_covariance_shrinkage(
    returns: pd.DataFrame,
    shrinkage: str,
) -> np.ndarray:
    if shrinkage == "none":
        return returns.cov().to_numpy(dtype=float)
    if shrinkage == "lw03":
        return LedoitWolf().fit(returns.to_numpy(dtype=float)).covariance_
    raise ValueError(f"Unsupported initialization covariance shrinkage: {shrinkage}")


def _extract_ridge_parameters(
    ridge_model: object,
    asset_ids: tuple[str, ...],
) -> tuple[float, pd.Series]:
    coefficients = _to_host_numpy(ridge_model._model.coef_)  # type: ignore[attr-defined]
    intercept = _to_host_scalar(ridge_model._model.intercept_)  # type: ignore[attr-defined]
    return (
        intercept,
        pd.Series(coefficients, index=list(asset_ids), name="weight"),
    )


def _extract_ridge_alpha(ridge_model: object) -> float:
    alpha = getattr(ridge_model._model, "alpha_", None)  # type: ignore[attr-defined]
    if alpha is None:
        alpha = getattr(ridge_model._model, "alpha", np.nan)  # type: ignore[attr-defined]
    try:
        return float(alpha)
    except (TypeError, ValueError):
        return float("nan")


def _extract_ridge_selected_r2(ridge_model: object) -> float:
    selected_r2 = getattr(ridge_model._model, "selected_r2_", np.nan)  # type: ignore[attr-defined]
    try:
        return float(selected_r2)
    except (TypeError, ValueError):
        return float("nan")


def _extract_ridge_target_r2(ridge_model: object) -> float:
    target_r2 = getattr(ridge_model._model, "target_r2_", np.nan)  # type: ignore[attr-defined]
    try:
        return float(target_r2)
    except (TypeError, ValueError):
        return float("nan")


def _pack_portfolio_parameters(intercept: float, weights: pd.Series) -> np.ndarray:
    return np.concatenate(([float(intercept)], weights.to_numpy(dtype=float)))


def _unpack_portfolio_parameters(
    beta_state: np.ndarray,
    asset_ids: tuple[str, ...],
) -> tuple[float, pd.Series]:
    return (
        float(beta_state[0]),
        pd.Series(beta_state[1:], index=list(asset_ids), name="weight", dtype=float),
    )


def _standardize_series(values: pd.Series) -> pd.Series:
    clean = _ensure_finite_series(values)
    std = float(clean.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(clean)), index=clean.index, name=clean.name, dtype=float)
    mean = float(clean.mean())
    standardized = (clean - mean) / std
    return _ensure_finite_series(standardized, name=clean.name)


def _ensure_finite_frame(values: pd.DataFrame) -> pd.DataFrame:
    clean = values.apply(pd.to_numeric, errors="coerce").astype(float)
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(axis=0, how="any")
    if clean.empty:
        raise ValueError("MACE input frame is empty after removing non-finite rows.")
    return clean


def _ensure_finite_series(values: pd.Series, name: str | None = None) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").astype(float)
    clean = clean.replace([np.inf, -np.inf], np.nan)
    finite = clean.dropna()
    if finite.empty:
        return pd.Series(
            np.zeros(len(clean)),
            index=clean.index,
            name=name or values.name,
            dtype=float,
        )
    fill_value = float(finite.mean())
    return pd.Series(
        clean.fillna(fill_value).to_numpy(dtype=float),
        index=clean.index,
        name=name or values.name,
        dtype=float,
    )


def _to_host_numpy(value: object) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy(), dtype=float)
    if hasattr(value, "to_pandas"):
        pandas_value = value.to_pandas()
        return np.asarray(pandas_value, dtype=float)
    if hasattr(value, "to_cupy"):
        return np.asarray(value.to_cupy().get(), dtype=float)
    return np.asarray(value, dtype=float)


def _to_host_scalar(value: object) -> float:
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    host_array = _to_host_numpy(value)
    return float(host_array.reshape(-1)[0])


def _compute_oob_prediction_mse(
    random_forest_model: object,
    target: pd.Series,
    backend_name: str,
) -> float:
    """
    Compute a Python-defined OOB MSE from sklearn RF OOB predictions.

    This is intentionally not treated as ranger parity. The R implementation
    uses ``ranger::prediction.error``; here we standardize the Python path by
    explicitly computing mean squared error from ``oob_prediction_`` so the
    metric definition is transparent and stable.
    """
    if backend_name != "sklearn":
        return float("nan")

    model = random_forest_model._model  # type: ignore[attr-defined]
    oob_prediction = getattr(model, "oob_prediction_", None)
    if oob_prediction is None:
        return float("nan")
    oob_prediction_array = np.asarray(oob_prediction, dtype=float).reshape(-1)
    target_array = np.asarray(target, dtype=float).reshape(-1)
    valid_mask = np.isfinite(oob_prediction_array) & np.isfinite(target_array)
    if not np.any(valid_mask):
        return float("nan")
    return float(np.mean((target_array[valid_mask] - oob_prediction_array[valid_mask]) ** 2))


def _compute_smoothed_oob_prediction_mse(diagnostics: pd.DataFrame, window: int) -> pd.Series:
    if "oob_prediction_mse" not in diagnostics.columns or diagnostics.empty:
        return pd.Series(dtype=float)
    return diagnostics["oob_prediction_mse"].rolling(window=window, center=True).mean()


def _select_iteration(
    diagnostics: pd.DataFrame,
    selection_rule: str,
    selection_smoothing_window: int,
    backend_name: str,
) -> tuple[int, str, float]:
    if diagnostics.empty:
        raise ValueError("Cannot select an iteration from empty diagnostics.")
    if selection_rule == "last_iteration":
        selection_value = float(diagnostics["iteration"].iloc[-1])
        return len(diagnostics) - 1, "iteration", selection_value
    if selection_rule != "best_oob":
        raise ValueError(f"Unsupported selection rule: {selection_rule}")
    if backend_name != "sklearn":
        raise ValueError(
            "selection_rule='best_oob' is only supported with the sklearn backend because "
            "cuML OOB error is not currently available."
        )

    smoothed_oob = _compute_smoothed_oob_prediction_mse(
        diagnostics=diagnostics,
        window=selection_smoothing_window,
    )
    valid_smoothed = smoothed_oob.dropna()
    if not valid_smoothed.empty:
        selected_index = int(valid_smoothed.idxmin())
        return (
            selected_index,
            "smoothed_oob_prediction_mse",
            float(valid_smoothed.loc[selected_index]),
        )
    valid_raw = diagnostics["oob_prediction_mse"].dropna()
    if not valid_raw.empty:
        selected_index = int(valid_raw.idxmin())
        return selected_index, "oob_prediction_mse", float(valid_raw.loc[selected_index])
    raise ValueError("selection_rule='best_oob' requires valid OOB errors, but none were found.")
