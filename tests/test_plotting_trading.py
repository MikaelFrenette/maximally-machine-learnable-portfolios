from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmlp.plotting import plot_cumulative_returns, plot_yearly_metric_heatmap


def test_trading_plotters_write_output_files(tmp_path: Path) -> None:
    trading_frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "buy_and_hold_return": [0.01, -0.02, 0.015, 0.005],
            "mace_mv_return": [0.02, -0.01, 0.02, 0.01],
            "mace_pm_return": [0.005, -0.005, 0.01, 0.002],
        }
    )
    yearly_summary = pd.DataFrame(
        {
            "year": [2020, 2020, 2020],
            "strategy": ["buy_and_hold", "mace_mv", "mace_pm"],
            "sharpe_ratio": [0.5, 1.0, 0.2],
        }
    )

    cumulative_path = tmp_path / "cumulative_returns.png"
    heatmap_path = tmp_path / "yearly_sharpe_heatmap.png"
    benchmark = pd.Series(
        [0.01, 0.0, 0.005, -0.002],
        index=trading_frame["date"],
        name="^GSPC",
    )

    plot_cumulative_returns(
        trading_frame=trading_frame,
        output_path=cumulative_path,
        benchmark_returns=benchmark,
        benchmark_label="S&P 500",
    )
    plot_yearly_metric_heatmap(yearly_summary=yearly_summary, output_path=heatmap_path)

    assert cumulative_path.exists()
    assert cumulative_path.stat().st_size > 0
    assert heatmap_path.exists()
    assert heatmap_path.stat().st_size > 0
