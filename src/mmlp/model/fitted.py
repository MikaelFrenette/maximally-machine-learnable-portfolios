"""
Fitted model objects for MMLP estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mmlp.model.backends import RegressorAdapter

__all__ = ["AlternatingFitResult", "FittedAlternatingMaceModel"]


@dataclass(slots=True)
class AlternatingFitResult:
    """
    Training diagnostics for the alternating estimator.

    Parameters
    ----------
    iteration_history : pandas.DataFrame
        Per-iteration diagnostics.
    converged : bool
        Whether the stopping criterion was reached.
    n_iterations : int
        Number of executed iterations.
    """

    iteration_history: pd.DataFrame
    converged: bool
    n_iterations: int


@dataclass(slots=True)
class FittedAlternatingMaceModel:
    """
    Fitted alternating ridge/random-forest estimator.

    Parameters
    ----------
    ridge_model : RegressorAdapter
        Final fitted ridge adapter.
    random_forest_model : RegressorAdapter
        Final fitted random-forest adapter.
    fit_result : AlternatingFitResult
        Training diagnostics and convergence metadata.
    """

    ridge_model: RegressorAdapter
    random_forest_model: RegressorAdapter
    fit_result: AlternatingFitResult

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict using the final fitted ridge model.
        """

        return self.ridge_model.predict(features)
