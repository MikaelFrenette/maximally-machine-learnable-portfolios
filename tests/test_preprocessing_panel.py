from __future__ import annotations

import pandas as pd

from mmlp.config.preprocessing import PreprocessingConfig
from mmlp.preprocessing.panel import build_lagged_feature_panel


def test_build_lagged_feature_panel_uses_scalar_max_lag() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                ]
            ),
            "asset_id": ["A", "A", "A", "A"],
            "return": [0.1, 0.2, 0.3, 0.4],
        }
    )
    config = PreprocessingConfig(lags=2)

    feature_panel = build_lagged_feature_panel(panel=panel, config=config)

    assert list(feature_panel.columns) == ["date", "asset_id", "return", "lag_1", "lag_2"]
    assert len(feature_panel) == 2
    assert feature_panel["lag_1"].tolist() == [0.2, 0.3]
    assert feature_panel["lag_2"].tolist() == [0.1, 0.2]


def test_build_lagged_feature_panel_keeps_rows_when_requested() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "asset_id": ["A", "A"],
            "return": [0.1, 0.2],
        }
    )
    config = PreprocessingConfig(lags=1, drop_rows_with_missing_lags=False)

    feature_panel = build_lagged_feature_panel(panel=panel, config=config)

    assert len(feature_panel) == 2
    assert pd.isna(feature_panel.loc[0, "lag_1"])
