from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmlp.config.run import RunConfig
from mmlp.workflows import generate_plots_from_config


def test_generate_plots_from_config_uses_materialized_artifacts(tmp_path: Path) -> None:
    universe_path = tmp_path / "mace_universe.json"
    universe_path.write_text(
        json.dumps({"equities": ["A", "B"], "bonds": [], "contracts": [], "forex": []}),
        encoding="utf-8",
    )

    output_dir = Path("outputs") / "plot_test_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    trading = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "buy_and_hold_return": [0.01, -0.02, 0.015, 0.005],
            "mace_signal": [0.01, 0.01, 0.01, 0.01],
            "mace_mv_position": [1.0, 1.0, 1.0, 1.0],
            "mace_mv_return": [0.02, -0.01, 0.02, 0.01],
            "mace_pm_signal": [0.005, 0.005, 0.005, 0.005],
            "mace_pm_position": [0.5, 0.5, 0.5, 0.5],
            "mace_pm_return": [0.005, -0.005, 0.01, 0.002],
        }
    )
    yearly = pd.DataFrame(
        {
            "year": [2020, 2020, 2020],
            "strategy": ["buy_and_hold", "mace_mv", "mace_pm"],
            "sharpe_ratio": [0.5, 1.0, 0.2],
        }
    )
    trading.to_csv(output_dir / "trading.csv", index=False)
    yearly.to_csv(output_dir / "trading_yearly_summary.csv", index=False)

    config = RunConfig.model_validate(
        {
            "run_name": "plot_test_run",
            "dataset": {
                "provider": "yahoo",
                "calendar": "XNYS",
                "price_field": "Adj Close",
                "start_date": "2020-01-01",
                "end_date": "2020-02-01",
                "universe": {"path": str(universe_path)},
            },
            "preprocessing": {"lags": 1},
            "split": {
                "train_start": "2020-01-01",
                "train_end": "2020-01-04",
                "test_start": "2020-01-05",
                "test_end": "2020-01-08",
            },
            "outputs": {"log_level": "INFO", "verbosity": 1},
        }
    )

    artifacts = generate_plots_from_config(config)

    assert artifacts.cumulative_plot_path.exists()
    assert artifacts.cumulative_plot_path.stat().st_size > 0
    assert artifacts.yearly_heatmap_path.exists()
    assert artifacts.yearly_heatmap_path.stat().st_size > 0
