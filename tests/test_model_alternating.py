from __future__ import annotations

import numpy as np
import pandas as pd

from mmlp.config.model import MaceModelConfig
from mmlp.config.preprocessing import PreprocessingConfig
from mmlp.model import build_model_input, fit_alternating_mace


def test_fit_alternating_mace_returns_fitted_model_with_history() -> None:
    rng = np.random.default_rng(1234)
    base = rng.normal(size=64)
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=64, freq="D"),
            "asset_id": ["A"] * 64,
            "return": base,
            "lag_1": np.roll(base, 1),
            "lag_2": np.roll(base, 2),
        }
    ).iloc[2:].reset_index(drop=True)

    model_input = build_model_input(panel=panel, config=PreprocessingConfig(lags=2))
    fitted = fit_alternating_mace(
        model_input=model_input,
        config=MaceModelConfig(
            max_iterations=8,
            min_iterations=2,
            random_forest_n_estimators=20,
            random_forest_min_node_size=1,
            random_state=1234,
        ),
    )

    assert fitted.fit_result.n_iterations >= 2
    assert not fitted.fit_result.iteration_history.empty
    predictions = fitted.predict(model_input.features)
    assert len(predictions) == len(model_input.features)
