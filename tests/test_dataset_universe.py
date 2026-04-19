from __future__ import annotations

import json
from pathlib import Path

from mmlp.dataset.sp500 import load_mace_reference_universe


def test_load_mace_reference_universe_preserves_order_and_size(tmp_path: Path) -> None:
    universe_path = tmp_path / "mace_universe.json"
    universe_path.write_text(
        json.dumps(
            {
                "equities": ["AAPL", "MSFT", "BRK.B"],
                "bonds": [],
                "contracts": [],
                "forex": [],
            }
        ),
        encoding="utf-8",
    )

    tickers = load_mace_reference_universe(universe_path, size=2)

    assert tickers == ("AAPL", "MSFT")


def test_load_mace_reference_universe_normalizes_yahoo_symbols(tmp_path: Path) -> None:
    universe_path = tmp_path / "mace_universe.json"
    universe_path.write_text(
        json.dumps({"equities": ["BRK.B"], "bonds": [], "contracts": [], "forex": []}),
        encoding="utf-8",
    )

    tickers = load_mace_reference_universe(universe_path)

    assert tickers == ("BRK-B",)
