"""
Utilities for the local MACE paper dataset.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

__all__ = [
    "extract_primary_asset_columns",
    "load_mace_paper_returns",
]


def extract_primary_asset_columns(columns: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """
    Extract the ordered primary asset-return columns from the paper dataset.

    The paper CSV stores the raw return universe first, followed by engineered
    lag and volatility columns such as ``AAPL_L1`` or ``AAPL_vol_L1``. For
    MACE20 we want the leading raw asset-return block only, preserving its
    original order.
    """

    asset_columns: list[str] = []
    for column in columns:
        if column == "date":
            continue
        if "_L" in column or "_vol_" in column:
            break
        asset_columns.append(column)
    return tuple(asset_columns)


def load_mace_paper_returns(
    path: Path,
    start_date: date,
    end_date: date,
    size: int | None = None,
) -> pd.DataFrame:
    """
    Load the ordered raw return matrix from the local MACE paper CSV.
    """

    panel = pd.read_csv(path, parse_dates=["date"])
    primary_columns = extract_primary_asset_columns(list(panel.columns))
    if not primary_columns:
        raise ValueError("Could not identify primary asset-return columns in MACE paper CSV.")

    selected_columns = primary_columns[:size] if size is not None else primary_columns
    filtered = panel.loc[
        (panel["date"] >= pd.Timestamp(start_date)) & (panel["date"] <= pd.Timestamp(end_date)),
        ["date", *selected_columns],
    ].copy()
    filtered = filtered.set_index("date").sort_index()
    filtered = filtered.apply(pd.to_numeric, errors="coerce").astype(float)
    return filtered
