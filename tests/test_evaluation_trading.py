from __future__ import annotations

import pandas as pd

from mmlp.config.trading import TradingConfig
from mmlp.evaluation import apply_mean_variance_overlay
from mmlp.evaluation.trading import _ewma_variance, rolling_prevailing_mean


def test_apply_mean_variance_overlay_returns_expected_artifacts() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    in_sample = pd.Series(
        [0.02, 0.01, -0.01, 0.015],
        index=pd.date_range("2019-12-28", periods=4, freq="D"),
    )
    realized = pd.Series([0.01, -0.02, 0.015, 0.01, -0.005, 0.02], index=dates)
    predicted = pd.Series([0.005, 0.002, 0.006, 0.004, 0.003, 0.007], index=dates)

    result = apply_mean_variance_overlay(
        realized_returns=realized,
        predicted_returns=predicted,
        in_sample_returns=in_sample,
        config=TradingConfig(lookback=3, gamma=5.0, alpha=0.94, horizon=1),
    )

    assert len(result.frame) == len(realized)
    assert {
        "buy_and_hold_return",
        "mace_signal",
        "mace_mv_position",
        "mace_mv_return",
        "mace_pm_signal",
        "mace_pm_position",
        "mace_pm_return",
    } <= set(result.frame.columns)
    assert set(result.summary["strategy"]) == {"buy_and_hold", "mace_mv", "mace_pm"}
    assert {"year", "strategy", "sharpe_ratio", "max_drawdown"} <= set(
        result.yearly_summary.columns
    )
    assert set(result.yearly_summary["strategy"]) == {"buy_and_hold", "mace_mv", "mace_pm"}


def test_rolling_prevailing_mean_uses_in_sample_history_for_early_oos_predictions() -> None:
    in_sample = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2019-12-29", periods=3, freq="D"),
    )
    realized = pd.Series(
        [10.0, 20.0],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    signal = rolling_prevailing_mean(
        realized_returns=realized,
        in_sample_returns=in_sample,
        lookback=2,
        horizon=1,
    )

    assert signal.iloc[0] == 2.5
    assert signal.iloc[1] == 6.5


def test_ewma_variance_matches_r_style_one_step_lagged_recursion() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    variance = _ewma_variance(series, alpha=0.5)

    expected = float(series.var(ddof=1))
    for value in series.iloc[:-1]:
        expected = 0.5 * expected + 0.5 * float(value**2)

    assert variance == expected
