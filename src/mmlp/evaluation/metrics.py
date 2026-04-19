"""
Performance metrics for return series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "annual_return",
    "annual_volatility",
    "calmar_ratio",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
]


def annual_return(series: pd.Series, annualization_factor: int = 252) -> float:
    return float(series.mean()) * annualization_factor


def annual_volatility(series: pd.Series, annualization_factor: int = 252) -> float:
    return float(series.std(ddof=1)) * np.sqrt(annualization_factor)


def sharpe_ratio(
    series: pd.Series,
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
    trading_cost: float = 0.0,
) -> float:
    mean_return = annual_return(series, annualization_factor) - trading_cost
    volatility = annual_volatility(series, annualization_factor)
    if volatility <= 1e-12:
        return float("nan")
    return float((mean_return - risk_free_rate) / volatility)


def max_drawdown(series: pd.Series) -> float:
    pnl = np.exp(series.cumsum())
    return float(np.min(pnl / np.maximum.accumulate(pnl) - 1.0))


def calmar_ratio(series: pd.Series, annualization_factor: int = 252) -> float:
    drawdown = abs(max_drawdown(series))
    if drawdown <= 1e-12:
        return float("nan")
    return float(annual_return(series, annualization_factor) / drawdown)


def sortino_ratio(series: pd.Series, annualization_factor: int = 252) -> float:
    downside = np.minimum(np.asarray(series, dtype=float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2))) * np.sqrt(annualization_factor)
    if downside_deviation <= 1e-12:
        return float("nan")
    return float(annual_return(series, annualization_factor) / downside_deviation)
