from __future__ import annotations

import numpy as np
import pandas as pd

from mmlp.config.model import MaceModelConfig
from mmlp.config.preprocessing import PreprocessingConfig
from mmlp.model import (
    build_mace_panel_matrix,
    build_marx_features,
    build_test_marx_features,
    fit_mace,
)


def test_build_mace_panel_matrix_creates_wide_returns() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "asset_id": ["A", "B", "A", "B"],
            "return": [0.1, 0.2, 0.15, 0.25],
        }
    )

    matrix = build_mace_panel_matrix(panel=panel, config=PreprocessingConfig())

    assert matrix.returns.shape == (2, 2)
    assert matrix.asset_ids == ("A", "B")


def test_build_mace_panel_matrix_fails_on_incomplete_assets() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "asset_id": ["A", "B", "A", "B", "A", "B"],
            "return": [0.1, None, 0.2, None, 0.3, 0.4],
        }
    )

    try:
        build_mace_panel_matrix(panel=panel, config=PreprocessingConfig())
    except ValueError as error:
        message = str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected build_mace_panel_matrix to fail on incomplete assets.")

    assert "Incomplete assets" in message
    assert "B(2 missing)" in message


def test_build_mace_panel_matrix_drops_globally_missing_dates() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "asset_id": ["A", "B", "A", "B", "A", "B"],
            "return": [None, None, 0.2, 0.1, 0.3, 0.4],
        }
    )

    matrix = build_mace_panel_matrix(panel=panel, config=PreprocessingConfig())

    assert matrix.asset_ids == ("A", "B")
    assert matrix.returns.shape == (2, 2)
    assert matrix.dates == (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))


def test_build_marx_features_uses_cumulative_lag_structure() -> None:
    series = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
    marx = build_marx_features(series=series, max_lag=2)

    assert list(marx.columns) == ["MARX_1", "MARX_2"]
    assert marx.iloc[0].tolist() == [0.0, 0.0]
    assert marx.iloc[2].tolist() == [2.0, 3.0]


def test_build_test_marx_features_seeds_leading_test_lags_from_train_tail() -> None:
    train_series = pd.Series(
        [10.0, 11.0, 12.0],
        index=pd.date_range("2020-01-01", periods=3),
    )
    test_series = pd.Series(
        [20.0, 21.0],
        index=pd.date_range("2020-01-04", periods=2),
    )

    marx = build_test_marx_features(
        train_series=train_series,
        test_series=test_series,
        max_lag=2,
    )

    assert list(marx.columns) == ["MARX_1", "MARX_2"]
    assert marx.iloc[0].tolist() == [12.0, 23.0]
    assert marx.iloc[1].tolist() == [20.0, 32.0]


def test_fit_mace_returns_weights_and_diagnostics() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-05",
                    "2020-01-06",
                    "2020-01-06",
                ]
            ),
            "asset_id": ["A", "B"] * 6,
            "return": [0.1, 0.2, 0.12, 0.21, 0.11, 0.19, 0.09, 0.18, 0.13, 0.22, 0.08, 0.17],
        }
    )
    matrix = build_mace_panel_matrix(panel=panel, config=PreprocessingConfig(lags=2))

    fitted = fit_mace(
        panel_matrix=matrix,
        preprocessing_config=PreprocessingConfig(lags=2),
        model_config=MaceModelConfig(
            max_iterations=4,
            min_iterations=2,
            random_forest_n_estimators=20,
            random_forest_min_node_size=1,
        ),
    )

    assert not fitted.diagnostics_.empty
    assert len(fitted.weights_) == 2
    assert len(fitted.z1_) == len(matrix.returns)
    assert len(fitted.z2_) == len(matrix.returns)


def test_fit_mace_honors_fixed_iteration_stopping_rule() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-05",
                    "2020-01-06",
                    "2020-01-06",
                ]
            ),
            "asset_id": ["A", "B"] * 6,
            "return": [0.1, 0.2, 0.12, 0.21, 0.11, 0.19, 0.09, 0.18, 0.13, 0.22, 0.08, 0.17],
        }
    )
    matrix = build_mace_panel_matrix(panel=panel, config=PreprocessingConfig(lags=2))

    fitted = fit_mace(
        panel_matrix=matrix,
        preprocessing_config=PreprocessingConfig(lags=2),
        model_config=MaceModelConfig(
            stopping_rule="max_iterations",
            max_iterations=4,
            min_iterations=2,
            random_forest_n_estimators=20,
            random_forest_min_node_size=1,
        ),
    )

    assert len(fitted.diagnostics_) == 4


def test_fit_mace_defaults_to_last_iteration_selection() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-05",
                    "2020-01-06",
                    "2020-01-06",
                ]
            ),
            "asset_id": ["A", "B"] * 6,
            "return": [0.1, 0.2, 0.12, 0.21, 0.11, 0.19, 0.09, 0.18, 0.13, 0.22, 0.08, 0.17],
        }
    )
    matrix = build_mace_panel_matrix(panel=panel, config=PreprocessingConfig(lags=2))

    fitted = fit_mace(
        panel_matrix=matrix,
        preprocessing_config=PreprocessingConfig(lags=2),
        model_config=MaceModelConfig(
            selection_rule="last_iteration",
            max_iterations=4,
            min_iterations=2,
            random_forest_n_estimators=20,
            random_forest_min_node_size=1,
        ),
    )

    assert fitted.selected_iteration_ == 4
    assert fitted.selection_rule_ == "last_iteration"
    assert fitted.selection_metric_ == "iteration"
    assert fitted.diagnostics_["is_selected"].sum() == 1
    assert bool(fitted.diagnostics_["is_selected"].iloc[-1])


def test_fit_mace_best_oob_selection_marks_one_iteration() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-05",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-07",
                    "2020-01-08",
                    "2020-01-08",
                ]
            ),
            "asset_id": ["A", "B"] * 8,
            "return": [
                0.1,
                0.2,
                0.12,
                0.21,
                0.11,
                0.19,
                0.09,
                0.18,
                0.13,
                0.22,
                0.08,
                0.17,
                0.14,
                0.24,
                0.1,
                0.2,
            ],
        }
    )
    matrix = build_mace_panel_matrix(panel=panel, config=PreprocessingConfig(lags=2))

    fitted = fit_mace(
        panel_matrix=matrix,
        preprocessing_config=PreprocessingConfig(lags=2),
        model_config=MaceModelConfig(
            device="cpu",
            selection_rule="best_oob",
            selection_smoothing_window=3,
            max_iterations=5,
            min_iterations=2,
            random_forest_n_estimators=50,
            random_forest_min_node_size=1,
            random_state=123,
        ),
    )

    assert fitted.selection_rule_ == "best_oob"
    assert 1 <= fitted.selected_iteration_ <= 5
    assert fitted.selection_metric_ in {"smoothed_oob_prediction_mse", "oob_prediction_mse"}
    assert fitted.diagnostics_["is_selected"].sum() == 1
    assert "oob_prediction_mse" in fitted.diagnostics_.columns
    assert "smoothed_oob_prediction_mse" in fitted.diagnostics_.columns
    assert "selection_metric" in fitted.diagnostics_.columns
    assert "selection_score" in fitted.diagnostics_.columns


def test_fit_mace_initialization_supports_shrinkage_on_singular_covariance() -> None:
    dates = pd.date_range("2020-01-01", periods=8)
    shared_returns = [0.01, 0.02, -0.01, 0.015, -0.005, 0.01, 0.0, 0.012]
    panel = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "asset_id": ["A"] * len(dates) + ["B"] * len(dates),
            "return": shared_returns + shared_returns,
        }
    )
    matrix = build_mace_panel_matrix(panel=panel, config=PreprocessingConfig(lags=2))

    fitted = fit_mace(
        panel_matrix=matrix,
        preprocessing_config=PreprocessingConfig(lags=2),
        model_config=MaceModelConfig(
            max_iterations=2,
            random_forest_n_estimators=20,
            random_forest_min_node_size=1,
            rhs_init_cov_sample=0.0,
            rhs_init_cov_sample_shrinkage="lw03",
        ),
    )

    assert len(fitted.weights_) == 2
    assert np.isfinite(fitted.weights_.to_numpy()).all()
    assert fitted.z2_.notna().all()
