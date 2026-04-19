"""
S&P 500 constituent helpers for demo and extraction workflows.

This module provides a lightweight way to fetch the current S&P 500
constituent list from a public CSV source, plus local ordered MACE reference
universe utilities.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

__all__ = [
    "DEFAULT_SP500_CONSTITUENTS_CSV_URL",
    "DEFAULT_MACE_UNIVERSE_PATH",
    "fetch_current_sp500_tickers",
    "load_mace_reference_universe",
]

DEFAULT_SP500_CONSTITUENTS_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
DEFAULT_MACE_UNIVERSE_PATH = Path(__file__).resolve().parents[3] / "mace_universe.json"


def fetch_current_sp500_tickers(
    csv_url: str = DEFAULT_SP500_CONSTITUENTS_CSV_URL,
) -> tuple[str, ...]:
    """
    Fetch the current S&P 500 constituent list from a public CSV source.

    Parameters
    ----------
    csv_url : str, default=DEFAULT_SP500_CONSTITUENTS_CSV_URL
        Source URL for the current constituent CSV.

    Returns
    -------
    tuple of str
        Yahoo-compatible ticker symbols.
    """

    with urlopen(csv_url) as response:  # noqa: S310
        csv_text = response.read().decode("utf-8")
    return _parse_sp500_constituents_csv(csv_text)


def load_mace_reference_universe(
    path: Path | str = DEFAULT_MACE_UNIVERSE_PATH,
    *,
    size: int | None = None,
) -> tuple[str, ...]:
    """
    Load the ordered MACE reference universe from the local JSON file.

    Parameters
    ----------
    path : Path or str, default=DEFAULT_MACE_UNIVERSE_PATH
        Path to the local MACE reference universe file.
    size : int or None, default=None
        Optional number of leading tickers to retain.

    Returns
    -------
    tuple of str
        Ordered Yahoo-compatible ticker symbols.
    """

    universe_path = Path(path)
    payload = json.loads(universe_path.read_text(encoding="utf-8"))
    if "equities" not in payload or not isinstance(payload["equities"], list):
        raise ValueError("mace_universe.json must contain an 'equities' list.")

    tickers = tuple(_normalize_yahoo_symbol(symbol) for symbol in payload["equities"])
    if size is not None:
        return tickers[:size]
    return tickers


def _parse_sp500_constituents_csv(csv_text: str) -> tuple[str, ...]:
    """
    Parse a constituent CSV payload into Yahoo-compatible ticker symbols.

    Parameters
    ----------
    csv_text : str
        CSV payload containing at least a ``Symbol`` column.

    Returns
    -------
    tuple of str
        Unique Yahoo-compatible ticker symbols in source order.
    """

    reader = csv.DictReader(StringIO(csv_text))
    if reader.fieldnames is None or "Symbol" not in reader.fieldnames:
        raise ValueError("S&P 500 constituent CSV must contain a 'Symbol' column.")

    tickers: list[str] = []
    seen: set[str] = set()
    for row in reader:
        raw_symbol = str(row["Symbol"]).strip().upper()
        if not raw_symbol:
            continue
        yahoo_symbol = _normalize_yahoo_symbol(raw_symbol)
        if yahoo_symbol in seen:
            continue
        seen.add(yahoo_symbol)
        tickers.append(yahoo_symbol)

    if not tickers:
        raise ValueError("S&P 500 constituent CSV did not contain any ticker symbols.")

    return tuple(tickers)


def _normalize_yahoo_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")
