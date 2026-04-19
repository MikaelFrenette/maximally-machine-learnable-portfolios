from __future__ import annotations

import pandas as pd
import pytest

from mmlp.config.preprocessing import PreprocessingConfig
from mmlp.model import build_model_input, validate_model_input_panel


def test_build_model_input_selects_expected_keys_target_and_features() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03", "2020-01-04"]),
            "asset_id": ["A", "A"],
            "return": [0.3, 0.4],
            "lag_1": [0.2, 0.3],
            "lag_2": [0.1, 0.2],
            "ticker": ["A", "A"],
        }
    )
    config = PreprocessingConfig(lags=2)

    model_input = build_model_input(panel=panel, config=config)

    assert list(model_input.keys.columns) == ["date", "asset_id"]
    assert model_input.target.tolist() == [0.3, 0.4]
    assert list(model_input.features.columns) == ["lag_1", "lag_2"]
    assert model_input.feature_columns == ("lag_1", "lag_2")


def test_validate_model_input_panel_rejects_missing_lag_column() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03"]),
            "asset_id": ["A"],
            "return": [0.3],
            "lag_1": [0.2],
        }
    )
    config = PreprocessingConfig(lags=2)

    with pytest.raises(ValueError, match="missing required columns"):
        validate_model_input_panel(panel=panel, config=config)


def test_validate_model_input_panel_rejects_duplicate_keys() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03", "2020-01-03"]),
            "asset_id": ["A", "A"],
            "return": [0.3, 0.4],
            "lag_1": [0.2, 0.3],
        }
    )
    config = PreprocessingConfig(lags=1)

    with pytest.raises(ValueError, match="duplicate"):
        validate_model_input_panel(panel=panel, config=config)
