from __future__ import annotations

import math

import pandas as pd

from mmlp.evaluation.metrics import annual_volatility, sortino_ratio


def test_annual_volatility_uses_sample_standard_deviation() -> None:
    series = pd.Series([1.0, 2.0, 3.0])

    observed = annual_volatility(series, annualization_factor=1)

    assert math.isclose(observed, float(series.std(ddof=1)))


def test_sortino_ratio_uses_root_mean_square_downside() -> None:
    series = pd.Series([0.02, -0.01, -0.03, 0.01])

    observed = sortino_ratio(series, annualization_factor=1)

    downside = [0.0, -0.01, -0.03, 0.0]
    expected = float(series.mean()) / math.sqrt(sum(value**2 for value in downside) / len(downside))
    assert math.isclose(observed, expected)
