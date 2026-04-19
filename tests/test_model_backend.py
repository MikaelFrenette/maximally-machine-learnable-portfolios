from __future__ import annotations

import pandas as pd
import pytest

from mmlp.config.model import MaceModelConfig
from mmlp.model.backends import _resolve_random_forest_mtry, resolve_model_backend


def test_resolve_model_backend_uses_sklearn_for_cpu() -> None:
    backend = resolve_model_backend(MaceModelConfig(device="cpu"))
    assert backend.name == "sklearn"


def test_resolve_model_backend_raises_when_cuda_backend_is_unavailable() -> None:
    with pytest.raises(RuntimeError, match="cuML"):
        resolve_model_backend(MaceModelConfig(device="cuda"))


def test_resolve_random_forest_mtry_matches_integer_rule() -> None:
    assert _resolve_random_forest_mtry(n_features=37, mtry_denom=10) == 3
    assert _resolve_random_forest_mtry(n_features=9, mtry_denom=10) == 1


def test_model_config_can_generate_log_spaced_ridge_alpha_grid() -> None:
    config = MaceModelConfig(
        ridge_alpha_grid=None,
        ridge_alpha_min=1e-4,
        ridge_alpha_max=10.0,
        ridge_alpha_grid_size=100,
    )

    grid = config.resolved_ridge_alpha_grid()

    assert len(grid) == 100
    assert grid[0] == 1e-4
    assert grid[-1] == 10.0


def test_cpu_ridge_respects_configured_lower_bound() -> None:
    backend = resolve_model_backend(
        MaceModelConfig(
            device="cpu",
            ridge_lower_bound=0.0,
            ridge_upper_bound=None,
        )
    )
    ridge = backend.ridge_factory().fit(
        pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [-1.0, -2.0, -3.0]}),
        pd.Series([1.0, 2.0, 3.0]),
    )

    assert (ridge._model.coef_ >= -1e-10).all()


def test_cpu_ridge_cv_selects_alpha_when_bounds_are_disabled() -> None:
    backend = resolve_model_backend(
        MaceModelConfig(
            device="cpu",
            ridge_alpha_selection="cv",
            ridge_alpha_grid=(1e-4, 1e-2, 1.0),
            ridge_lower_bound=None,
            ridge_upper_bound=None,
        )
    )
    ridge = backend.ridge_factory().fit(
        pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [0.0, 1.0, 0.0, 1.0]}),
        pd.Series([1.0, 2.1, 2.9, 4.2]),
    )

    assert hasattr(ridge._model, "alpha_")
    assert float(ridge._model.alpha_) in {1e-4, 1e-2, 1.0}


def test_cpu_bounded_ridge_supports_stock_specific_penalty() -> None:
    backend = resolve_model_backend(
        MaceModelConfig(
            device="cpu",
            ridge_alpha_selection="fixed",
            ridge_lower_bound=-3.0,
            ridge_upper_bound=None,
            ridge_stock_specific_penalty=True,
        )
    )
    ridge = backend.ridge_factory().fit(
        pd.DataFrame({"x1": [1.0, 2.0, 4.0, 8.0], "x2": [1.0, 1.1, 0.9, 1.0]}),
        pd.Series([0.5, 1.0, 2.0, 4.0]),
    )

    predictions = ridge.predict(pd.DataFrame({"x1": [16.0], "x2": [1.0]}))
    assert len(predictions) == 1


def test_cpu_ridge_target_r2_selects_alpha_from_grid() -> None:
    backend = resolve_model_backend(
        MaceModelConfig(
            device="cpu",
            ridge_alpha_selection="target_r2",
            ridge_alpha_grid=(1e-4, 1e-2, 1.0),
            ridge_target_r2=0.01,
            ridge_lower_bound=-3.0,
            ridge_upper_bound=None,
            ridge_stock_specific_penalty=True,
        )
    )
    ridge = backend.ridge_factory().fit(
        pd.DataFrame({"x1": [1.0, 2.0, 4.0, 8.0], "x2": [1.0, 1.1, 0.9, 1.0]}),
        pd.Series([0.5, 1.0, 2.0, 4.0]),
    )

    assert hasattr(ridge._model, "alpha_")
    assert float(ridge._model.alpha_) in {1e-4, 1e-2, 1.0}


def test_lambda_tranquilizer_scales_selected_ridge_alpha() -> None:
    backend = resolve_model_backend(
        MaceModelConfig(
            device="cpu",
            ridge_alpha_selection="target_r2",
            ridge_alpha_grid=(1e-4, 1e-2, 1.0),
            ridge_target_r2=0.01,
            lambda_tranquilizer=5.0,
            ridge_lower_bound=-3.0,
            ridge_upper_bound=None,
            ridge_stock_specific_penalty=True,
        )
    )
    ridge = backend.ridge_factory().fit(
        pd.DataFrame({"x1": [1.0, 2.0, 4.0, 8.0], "x2": [1.0, 1.1, 0.9, 1.0]}),
        pd.Series([0.5, 1.0, 2.0, 4.0]),
    )

    assert hasattr(ridge._model, "alpha_")
    assert float(ridge._model.alpha_) in {5e-4, 5e-2, 5.0}
