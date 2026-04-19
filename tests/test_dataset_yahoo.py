from __future__ import annotations

from datetime import date

import pytest

from mmlp.dataset.yahoo import YahooDailyReturnsRequest


def test_yahoo_request_validates_ticker_case() -> None:
    with pytest.raises(ValueError):
        YahooDailyReturnsRequest(
            tickers=("spy",),
            start_date=date(2020, 1, 1),
            end_date=date(2020, 2, 1),
        )


def test_yahoo_request_validates_date_order() -> None:
    with pytest.raises(ValueError):
        YahooDailyReturnsRequest(
            tickers=("SPY",),
            start_date=date(2020, 2, 1),
            end_date=date(2020, 1, 1),
        )
