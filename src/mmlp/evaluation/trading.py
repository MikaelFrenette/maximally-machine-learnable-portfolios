"""
Mean-variance trading overlay for MACE return signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mmlp.config.trading import TradingConfig
from mmlp.evaluation.metrics import (
    annual_return,
    annual_volatility,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

__all__ = [
    "MeanVarianceResult",
    "apply_mean_variance_overlay",
    "build_trading_summary",
    "build_yearly_trading_summary",
    "rolling_prevailing_mean",
]


@dataclass(slots=True)
class MeanVarianceResult:
    """
    Materialized mean-variance overlay outputs.
    """

    frame: pd.DataFrame
    summary: pd.DataFrame
    yearly_summary: pd.DataFrame


def rolling_prevailing_mean(
    realized_returns: pd.Series,
    in_sample_returns: pd.Series,
    lookback: int,
    horizon: int,
) -> pd.Series:
    return _rolling_prevailing_mean_with_history(
        out_of_sample_returns=realized_returns.astype(float),
        in_sample_returns=in_sample_returns.astype(float),
        lookback=lookback,
        horizon=horizon,
    )


def apply_mean_variance_overlay(
    realized_returns: pd.Series,
    predicted_returns: pd.Series,
    in_sample_returns: pd.Series,
    config: TradingConfig,
) -> MeanVarianceResult:
    """
    Apply the R-style mean-variance trading overlay to a return series.
    """

    realized = realized_returns.astype(float)
    predicted = predicted_returns.astype(float).reindex(realized.index)
    prevailing_mean = rolling_prevailing_mean(
        realized_returns=realized,
        in_sample_returns=in_sample_returns.astype(float),
        lookback=config.lookback,
        horizon=config.horizon,
    )

    mace_positions = _mean_variance_positions(
        realized_returns=realized,
        predicted_returns=predicted,
        in_sample_returns=in_sample_returns.astype(float),
        config=config,
    )
    pm_positions = _mean_variance_positions(
        realized_returns=realized,
        predicted_returns=prevailing_mean,
        in_sample_returns=in_sample_returns.astype(float),
        config=config,
    )

    frame = pd.DataFrame(
        {
            "buy_and_hold_return": realized.to_numpy(),
            "mace_signal": predicted.to_numpy(),
            "mace_mv_position": mace_positions.to_numpy(),
            "mace_mv_return": (mace_positions * realized).to_numpy(),
            "mace_pm_signal": prevailing_mean.to_numpy(),
            "mace_pm_position": pm_positions.to_numpy(),
            "mace_pm_return": (pm_positions * realized).to_numpy(),
        },
        index=realized.index,
    )
    frame.index.name = realized.index.name

    summary = build_trading_summary(frame=frame)
    yearly_summary = build_yearly_trading_summary(frame=frame)
    return MeanVarianceResult(frame=frame, summary=summary, yearly_summary=yearly_summary)


def build_trading_summary(
    frame: pd.DataFrame,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Summarize buy-and-hold and mean-variance overlay returns.
    """

    strategies = {
        "buy_and_hold": frame["buy_and_hold_return"],
        "mace_mv": frame["mace_mv_return"],
        "mace_pm": frame["mace_pm_return"],
    }
    rows = []
    for name, series in strategies.items():
        rows.append(
            {
                "strategy": name,
                "annual_return": annual_return(series, annualization_factor),
                "annual_volatility": annual_volatility(series, annualization_factor),
                "sharpe_ratio": sharpe_ratio(series, annualization_factor),
                "max_drawdown": max_drawdown(series),
                "calmar_ratio": calmar_ratio(series, annualization_factor),
                "sortino_ratio": sortino_ratio(series, annualization_factor),
            }
        )
    return pd.DataFrame(rows)


def build_yearly_trading_summary(
    frame: pd.DataFrame,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Summarize trading returns by calendar year and strategy.
    """

    strategies = {
        "buy_and_hold": frame["buy_and_hold_return"],
        "mace_mv": frame["mace_mv_return"],
        "mace_pm": frame["mace_pm_return"],
    }
    rows = []
    years = pd.Index(frame.index.year, name="year")
    for strategy_name, series in strategies.items():
        for year in sorted(years.unique()):
            yearly_series = series.loc[years == year]
            rows.append(
                {
                    "year": int(year),
                    "strategy": strategy_name,
                    "annual_return": annual_return(yearly_series, annualization_factor),
                    "annual_volatility": annual_volatility(yearly_series, annualization_factor),
                    "sharpe_ratio": sharpe_ratio(yearly_series, annualization_factor),
                    "max_drawdown": max_drawdown(yearly_series),
                    "calmar_ratio": calmar_ratio(yearly_series, annualization_factor),
                    "sortino_ratio": sortino_ratio(yearly_series, annualization_factor),
                }
            )
    return pd.DataFrame(rows)


def _mean_variance_positions(
    realized_returns: pd.Series,
    predicted_returns: pd.Series,
    in_sample_returns: pd.Series,
    config: TradingConfig,
) -> pd.Series:
    positions = []
    for index in range(len(predicted_returns)):
        history = _history_available_at_prediction_time(
            out_of_sample_returns=realized_returns,
            in_sample_returns=in_sample_returns,
            prediction_index=index,
            horizon=config.horizon,
        )
        sigma2 = _ewma_variance(history.tail(config.lookback), alpha=config.alpha)
        raw_weight = predicted_returns.iloc[index] / (config.gamma * sigma2)
        bounded_weight = min(max(raw_weight, config.position_min), config.position_max)
        positions.append(float(bounded_weight))
    return pd.Series(positions, index=predicted_returns.index, name="position")


def _rolling_prevailing_mean_with_history(
    out_of_sample_returns: pd.Series,
    in_sample_returns: pd.Series,
    lookback: int,
    horizon: int,
) -> pd.Series:
    signal = []
    for index in range(len(out_of_sample_returns)):
        history = _history_available_at_prediction_time(
            out_of_sample_returns=out_of_sample_returns,
            in_sample_returns=in_sample_returns,
            prediction_index=index,
            horizon=horizon,
        )
        window = history.tail(lookback)
        signal.append(float(window.mean()) if not window.empty else 0.0)
    return pd.Series(signal, index=out_of_sample_returns.index, name="prevailing_mean_signal")


def _history_available_at_prediction_time(
    out_of_sample_returns: pd.Series,
    in_sample_returns: pd.Series,
    prediction_index: int,
    horizon: int,
) -> pd.Series:
    available_oos_count = max(prediction_index - horizon + 1, 0)
    available_oos = out_of_sample_returns.iloc[:available_oos_count]
    return pd.concat([in_sample_returns, available_oos])


def _ewma_variance(series: pd.Series, alpha: float) -> float:
    values = np.asarray(series, dtype=float)
    if len(values) == 0:
        return 1.0

    if len(values) == 1:
        variance = float(values[0] ** 2)
    else:
        variance = float(np.var(values, ddof=1))

    if not np.isfinite(variance) or variance <= 1e-12:
        variance = 1.0

    # Mirror the R helper semantics:
    # vol[1] <- var(x)
    # for (i in 2:n) vol[i] <- alpha * vol[i - 1] + (1 - alpha) * x[i - 1]^2
    # and the trading code uses tail(ewma(...), 1), so the final forecast is
    # based on a one-step-lagged recursion.
    for lagged_value in values[:-1]:
        variance = alpha * variance + (1.0 - alpha) * float(lagged_value**2)
    return max(variance, 1e-12)
