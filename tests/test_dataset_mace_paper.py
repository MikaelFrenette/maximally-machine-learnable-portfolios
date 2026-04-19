from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmlp.dataset.mace_paper import extract_primary_asset_columns, load_mace_paper_returns


def test_extract_primary_asset_columns_stops_before_engineered_features() -> None:
    columns = [
        "date",
        "AAPL",
        "AMZN",
        "MSFT",
        "AAPL_L1",
        "AAPL_L2",
        "AAPL_vol_L1",
    ]

    observed = extract_primary_asset_columns(columns)

    assert observed == ("AAPL", "AMZN", "MSFT")


def test_load_mace_paper_returns_preserves_primary_order_and_size(tmp_path: Path) -> None:
    path = tmp_path / "daily__MACE_paper.csv"
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "AAPL": [0.01, 0.02],
            "AMZN": [0.03, 0.04],
            "MSFT": [0.05, 0.06],
            "AAPL_L1": [0.0, 0.01],
        }
    )
    frame.to_csv(path, index=False)

    returns = load_mace_paper_returns(
        path=path,
        start_date=pd.Timestamp("2020-01-01").date(),
        end_date=pd.Timestamp("2020-01-02").date(),
        size=2,
    )

    assert list(returns.columns) == ["AAPL", "AMZN"]
    assert returns.shape == (2, 2)
