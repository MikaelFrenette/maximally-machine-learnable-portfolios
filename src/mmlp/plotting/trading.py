"""
Trading result plots for MMLP runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "plot_cumulative_returns",
    "plot_yearly_metric_heatmap",
]


def plot_cumulative_returns(
    trading_frame: pd.DataFrame,
    output_path: Path,
    benchmark_returns: pd.Series | None = None,
    benchmark_label: str = "S&P 500",
    date_column: str = "date",
    dpi: int = 150,
) -> Path:
    """
    Plot test-set wealth indices for buy-and-hold, trading overlays, and an
    optional benchmark series.
    """

    frame = trading_frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    buy_and_hold_wealth = _wealth_index(frame["buy_and_hold_return"])
    mace_mv_wealth = _wealth_index(frame["mace_mv_return"])
    mace_pm_wealth = _wealth_index(frame["mace_pm_return"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        frame[date_column],
        buy_and_hold_wealth,
        label="Buy & Hold",
        linewidth=2.0,
        color="#4c566a",
    )
    ax.plot(
        frame[date_column],
        mace_mv_wealth,
        label="MACE",
        linewidth=2.0,
        color="#c0392b",
    )
    ax.plot(
        frame[date_column],
        mace_pm_wealth,
        label="MACE (PM)",
        linewidth=2.0,
        color="#e67e22",
    )
    if benchmark_returns is not None:
        benchmark = benchmark_returns.reindex(frame[date_column]).astype(float)
        if benchmark.notna().any():
            ax.plot(
                frame[date_column],
                _wealth_index(benchmark.fillna(0.0)),
                label=benchmark_label,
                linewidth=2.0,
                color="#1f77b4",
            )
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("Cumulative Wealth")
    ax.set_xlabel("")
    ax.set_ylabel("Wealth Index (Start = 1.0)")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _wealth_index(returns: pd.Series) -> pd.Series:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    return pd.Series(np.exp(clean.cumsum()), index=clean.index)


def plot_yearly_metric_heatmap(
    yearly_summary: pd.DataFrame,
    output_path: Path,
    metric: str = "sharpe_ratio",
    dpi: int = 150,
) -> Path:
    """
    Plot a yearly strategy heatmap for a selected metric.
    """

    pivot = yearly_summary.pivot(index="strategy", columns="year", values=metric).sort_index()

    fig_width = max(8, len(pivot.columns) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 3.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
    ax.set_title(f"Yearly {metric.replace('_', ' ').title()}")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(column) for column in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))

    for row_index, strategy in enumerate(pivot.index):
        for col_index, year in enumerate(pivot.columns):
            value = pivot.loc[strategy, year]
            if pd.notna(value):
                ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
