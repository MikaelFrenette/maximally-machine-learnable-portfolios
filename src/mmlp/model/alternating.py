"""
Core alternating ridge/random-forest estimator for MMLP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmlp.config.model import MaceModelConfig
from mmlp.model.backends import resolve_model_backend
from mmlp.model.fitted import AlternatingFitResult, FittedAlternatingMaceModel
from mmlp.model.input import ModelInput

__all__ = ["fit_alternating_mace"]


def fit_alternating_mace(
    model_input: ModelInput,
    config: MaceModelConfig,
) -> FittedAlternatingMaceModel:
    """
    Fit the core non-bagged alternating ridge/random-forest estimator.

    Parameters
    ----------
    model_input : ModelInput
        Training-ready model input.
    config : MaceModelConfig
        Model settings for the estimator and backend.

    Returns
    -------
    FittedAlternatingMaceModel
        Fitted alternating estimator and training diagnostics.
    """

    backend = resolve_model_backend(config=config)
    features = model_input.features
    latent = _standardize_series(model_input.target)
    diagnostics: list[dict[str, float | int | bool]] = []
    converged = False

    ridge_model = backend.ridge_factory()
    random_forest_model = backend.random_forest_factory()

    for iteration in range(1, config.max_iterations + 1):
        random_forest_model = backend.random_forest_factory().fit(features, latent)
        rf_prediction = _standardize_series(random_forest_model.predict(features))

        ridge_model = backend.ridge_factory().fit(features, rf_prediction)
        ridge_prediction = _standardize_series(ridge_model.predict(features))

        updated_latent = _standardize_series(
            (1.0 - config.learning_rate) * latent + config.learning_rate * ridge_prediction
        )
        delta = float(np.sqrt(np.mean((updated_latent - latent) ** 2)))

        diagnostics.append(
            {
                "iteration": iteration,
                "latent_delta_rmse": delta,
                "rf_prediction_std": float(rf_prediction.std(ddof=0)),
                "ridge_prediction_std": float(ridge_prediction.std(ddof=0)),
            }
        )

        latent = updated_latent
        if (
            config.stopping_rule == "tolerance"
            and iteration >= config.min_iterations
            and delta <= config.tolerance
        ):
            converged = True
            break

    fit_result = AlternatingFitResult(
        iteration_history=pd.DataFrame(diagnostics),
        converged=converged,
        n_iterations=len(diagnostics),
    )
    return FittedAlternatingMaceModel(
        ridge_model=ridge_model,
        random_forest_model=random_forest_model,
        fit_result=fit_result,
    )


def _standardize_series(values: pd.Series) -> pd.Series:
    """
    Standardize a series with deterministic zero-variance handling.
    """

    series = pd.Series(values, index=values.index, copy=True)
    std = float(series.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index, name=series.name)
    mean = float(series.mean())
    return pd.Series((series - mean) / std, index=series.index, name=series.name)
