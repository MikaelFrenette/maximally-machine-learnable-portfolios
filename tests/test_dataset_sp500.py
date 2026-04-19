from __future__ import annotations

from mmlp.dataset.sp500 import _parse_sp500_constituents_csv


def test_parse_sp500_constituents_normalizes_yahoo_symbols() -> None:
    csv_text = "\n".join(
        [
            "Symbol,Security,Sector",
            "BRK.B,Berkshire Hathaway,Financials",
            "BF.B,Brown-Forman,Consumer Staples",
            "AAPL,Apple,Information Technology",
        ]
    )

    tickers = _parse_sp500_constituents_csv(csv_text)

    assert tickers == ("BRK-B", "BF-B", "AAPL")


def test_parse_sp500_constituents_rejects_missing_symbol_column() -> None:
    csv_text = "Ticker,Security\nAAPL,Apple\n"

    try:
        _parse_sp500_constituents_csv(csv_text)
    except ValueError as error:
        assert "Symbol" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for missing Symbol column.")
