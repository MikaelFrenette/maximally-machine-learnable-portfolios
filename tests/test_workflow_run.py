from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmlp.config.run import RunConfig
from mmlp.workflows.run import run_pipeline_from_config


def test_run_pipeline_from_config_writes_training_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    universe_path = tmp_path / "mace_universe.json"
    universe_path.write_text(
        json.dumps({"equities": ["A", "B"], "bonds": [], "contracts": [], "forex": []}),
        encoding="utf-8",
    )
    raw_panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=8, freq="D").repeat(2),
            "asset_id": ["A", "B"] * 8,
            "return": [
                0.1,
                0.2,
                0.15,
                0.25,
                0.12,
                0.22,
                0.18,
                0.28,
                0.2,
                0.3,
                0.21,
                0.31,
                0.19,
                0.29,
                0.17,
                0.27,
            ],
            "ticker": ["A", "B"] * 8,
            "sector": ["unknown", "unknown"] * 8,
            "industry": ["unknown", "unknown"] * 8,
        }
    )

    def fake_extract_features_from_run_config(config: RunConfig) -> Path:
        raw_path = Path("outputs") / config.run_name / "panel.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_panel.to_csv(raw_path, index=False)

        feature_path = Path("outputs") / config.run_name / "features.csv"
        feature_panel = raw_panel.copy()
        feature_panel["lag_1"] = feature_panel.groupby("asset_id")["return"].shift(1)
        feature_panel = feature_panel.dropna().reset_index(drop=True)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_panel.to_csv(feature_path, index=False)
        return feature_path

    monkeypatch.setattr(
        "mmlp.workflows.run.extract_features_from_run_config",
        fake_extract_features_from_run_config,
    )

    config = RunConfig.model_validate(
        {
            "run_name": "test_run",
            "dataset": {
                "provider": "yahoo",
                "calendar": "XNYS",
                "price_field": "Adj Close",
                "start_date": "2020-01-01",
                "end_date": "2020-02-01",
                "universe": {"path": str(universe_path)},
            },
            "preprocessing": {
                "lags": 1,
            },
            "split": {
                "train_start": "2020-01-01",
                "train_end": "2020-01-04",
                "test_start": "2020-01-05",
                "test_end": "2020-01-08",
            },
            "model": {
                "device": "cpu",
                "max_iterations": 4,
                "min_iterations": 2,
                "random_forest_n_estimators": 10,
            },
            "outputs": {
                "log_level": "INFO",
                "verbosity": 1,
            },
        }
    )

    artifacts = run_pipeline_from_config(config)

    assert artifacts.raw_panel_path == Path("outputs") / "test_run" / "panel.csv"
    assert artifacts.feature_panel_path == Path("outputs") / "test_run" / "features.csv"
    assert artifacts.diagnostics_path.exists()
    assert artifacts.predictions_path.exists()
    assert artifacts.weights_raw_path.exists()
    assert artifacts.weights_normalized_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.trading_path.exists()
    assert artifacts.trading_summary_path.exists()
    assert artifacts.trading_yearly_summary_path.exists()
    assert artifacts.cumulative_plot_path.exists()
    assert artifacts.yearly_heatmap_path.exists()
    assert artifacts.log_path.exists()
    diagnostics = pd.read_csv(artifacts.diagnostics_path)
    predictions = pd.read_csv(artifacts.predictions_path)
    weights_raw = pd.read_csv(artifacts.weights_raw_path)
    weights_normalized = pd.read_csv(artifacts.weights_normalized_path)
    summary = pd.read_csv(artifacts.summary_path)
    trading = pd.read_csv(artifacts.trading_path)
    trading_summary = pd.read_csv(artifacts.trading_summary_path)
    trading_yearly_summary = pd.read_csv(artifacts.trading_yearly_summary_path)
    log_text = artifacts.log_path.read_text(encoding="utf-8")
    assert not diagnostics.empty
    assert {"date", "split", "portfolio_return", "signal_prediction", "z1", "z2"} <= set(
        predictions.columns
    )
    assert {"asset_id", "weight"} <= set(weights_raw.columns)
    assert {"asset_id", "weight"} <= set(weights_normalized.columns)
    assert {
        "date",
        "buy_and_hold_return",
        "mace_signal",
        "mace_mv_position",
        "mace_mv_return",
        "mace_pm_signal",
        "mace_pm_position",
        "mace_pm_return",
    } <= set(trading.columns)
    assert {"strategy", "sharpe_ratio", "max_drawdown"} <= set(trading_summary.columns)
    assert {"year", "strategy", "sharpe_ratio", "max_drawdown"} <= set(
        trading_yearly_summary.columns
    )
    assert {
        "annual_return",
        "sharpe_ratio",
        "gross_exposure",
        "selected_iteration",
        "selection_rule",
    } <= set(summary.columns)
    assert "Running pipeline: run_name=test_run" in log_text
