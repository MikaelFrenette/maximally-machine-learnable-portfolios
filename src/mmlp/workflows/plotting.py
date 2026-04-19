"""
Plot-generation workflow for existing run artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mmlp.config.run import RunConfig
from mmlp.dataset.yahoo import YahooDailyReturnsLoader, YahooDailyReturnsRequest
from mmlp.plotting import plot_cumulative_returns, plot_yearly_metric_heatmap

__all__ = ["PlotArtifacts", "generate_plots_from_config"]


@dataclass(slots=True)
class PlotArtifacts:
    """
    Materialized plot paths for a run.
    """

    cumulative_plot_path: Path
    yearly_heatmap_path: Path


def generate_plots_from_config(config: RunConfig) -> PlotArtifacts:
    """
    Generate plots from previously materialized trading artifacts.
    """

    cumulative_plot_path = _resolve_output_path(
        config=config,
        filename=f"cumulative_returns.{config.plotting.image_format}",
    )
    yearly_heatmap_path = _resolve_output_path(
        config=config,
        filename=f"yearly_sharpe_heatmap.{config.plotting.image_format}",
    )

    if not config.plotting.enabled:
        cumulative_plot_path.touch()
        yearly_heatmap_path.touch()
        return PlotArtifacts(
            cumulative_plot_path=cumulative_plot_path,
            yearly_heatmap_path=yearly_heatmap_path,
        )

    trading_path = _resolve_output_path(config=config, filename="trading.csv")
    yearly_summary_path = _resolve_output_path(config=config, filename="trading_yearly_summary.csv")
    trading_frame = pd.read_csv(trading_path, parse_dates=[config.preprocessing.date_column])
    yearly_summary = pd.read_csv(yearly_summary_path)
    benchmark_returns = _load_benchmark_returns(config=config)

    plot_cumulative_returns(
        trading_frame=trading_frame,
        output_path=cumulative_plot_path,
        benchmark_returns=benchmark_returns,
        benchmark_label=config.plotting.benchmark_label,
        date_column=config.preprocessing.date_column,
        dpi=config.plotting.dpi,
    )
    plot_yearly_metric_heatmap(
        yearly_summary=yearly_summary,
        output_path=yearly_heatmap_path,
        metric="sharpe_ratio",
        dpi=config.plotting.dpi,
    )
    return PlotArtifacts(
        cumulative_plot_path=cumulative_plot_path,
        yearly_heatmap_path=yearly_heatmap_path,
    )


def _resolve_output_path(config: RunConfig, filename: str) -> Path:
    return Path("outputs") / config.run_name / filename


def _load_benchmark_returns(config: RunConfig) -> pd.Series | None:
    ticker = config.plotting.benchmark_ticker
    if ticker is None:
        return None

    loader = YahooDailyReturnsLoader(
        auto_adjust=config.dataset.auto_adjust,
        progress=False,
    )
    request = YahooDailyReturnsRequest(
        tickers=(ticker.upper(),),
        start_date=config.split.test_start,
        end_date=config.split.test_end,
        price_field=config.dataset.price_field,
        calendar=config.dataset.calendar,
        drop_missing=False,
    )
    try:
        returns = loader.load_returns(request=request)
    except Exception:
        return None
    if returns.empty:
        return None
    series = returns.iloc[:, 0].astype(float)
    series.name = ticker.upper()
    return series
