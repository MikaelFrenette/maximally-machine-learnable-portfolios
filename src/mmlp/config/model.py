"""
Model configuration models for MMLP workflows.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from mmlp.config.models import BaseConfigModel

__all__ = ["MaceModelConfig"]


class MaceModelConfig(BaseConfigModel):
    """
    High-level MACE model settings.

    Parameters
    ----------
    variant : {"mace"}
        Model variant identifier.
    device : {"cpu", "cuda"}, default="cpu"
        Requested execution device. ``"cpu"`` uses sklearn. ``"cuda"``
        requests a cuML-backed implementation and fails clearly when cuML is
        unavailable.
    stopping_rule : {"max_iterations", "tolerance"}, default="max_iterations"
        Stopping policy for the alternating fit. ``"max_iterations"`` matches
        the fixed-iteration R-style behavior. ``"tolerance"`` enables
        convergence stopping based on the latent update size.
    selection_rule : {"last_iteration", "best_oob"}, default="last_iteration"
        Post-fit iteration-selection rule. ``"last_iteration"`` keeps the
        final iterate. ``"best_oob"`` selects the iteration minimizing a
        centered rolling mean of a Python-defined OOB MSE computed from
        sklearn's ``oob_prediction_`` output.
    selection_smoothing_window : int, default=5
        Centered rolling window used to smooth OOB errors before selecting the
        best iteration.
    max_iterations : int, default=100
        Maximum number of alternating optimization iterations.
    min_iterations : int, default=5
        Minimum number of iterations before convergence is checked when
        ``stopping_rule="tolerance"``.
    learning_rate : float, default=0.1
        Blending weight used when updating the latent target.
    tolerance : float, default=1e-4
        Convergence tolerance applied to the latent-target update.
    ridge_alpha : float, default=0.0168
        Fixed ridge regularization strength used when
        ``ridge_alpha_selection="fixed"``.
    ridge_alpha_selection : {"fixed", "cv", "target_r2"}, default="fixed"
        Policy for choosing the ridge penalty. ``"fixed"`` uses the provided
        ``ridge_alpha``. ``"cv"`` uses ``sklearn.linear_model.RidgeCV`` over a
        configured alpha grid when coefficient bounds are disabled.
        ``"target_r2"`` selects the alpha whose in-sample ridge fit is closest
        to ``ridge_target_r2``, which is the closest Python analogue to the
        R code's lambda-targeting path.
    ridge_alpha_grid : tuple of float or None, default=None
        Optional explicit alpha grid override. When omitted, the grid is
        generated as a log-spaced path from ``ridge_alpha_min`` to
        ``ridge_alpha_max`` with ``ridge_alpha_grid_size`` points.

        This is a practical approximation of ``glmnet``-style lambda
        selection, not a replication of the full ``glmnet`` path machinery.
    ridge_alpha_min : float, default=1e-4
        Minimum alpha for the generated log-spaced ridge path.
    ridge_alpha_max : float, default=10.0
        Maximum alpha for the generated log-spaced ridge path.
    ridge_alpha_grid_size : int, default=100
        Number of points in the generated log-spaced ridge path.
    ridge_target_r2 : float, default=0.01
        Target in-sample ``R2`` used when ``ridge_alpha_selection="target_r2"``.
        This mirrors the paper's plain-MACE lambda-targeting default.
    lambda_tranquilizer : float, default=1.0
        Post-selection multiplier applied to the chosen ridge penalty. The R
        driver exposes this as ``lambda_tranquilizer`` but does not wire it
        through the inspected fit path. We implement the intended behavior
        explicitly: choose a penalty via the configured rule, then scale it by
        ``lambda_tranquilizer`` before the final ridge fit.
    ridge_n_jobs : int, default=-1
        CPU parallelism used for ridge alpha selection. ``-1`` means use all
        available cores.
    ridge_lower_bound : float or None, default=-3.0
        Lower bound applied to ridge coefficients. This mirrors the paper's
        default ``glmnet`` lower limit more closely than unconstrained
        sklearn/cuml ridge. ``None`` disables the lower bound.
    ridge_upper_bound : float or None, default=None
        Optional upper bound applied to ridge coefficients. ``None`` means no
        upper bound, which matches the paper default.
    ridge_stock_specific_penalty : bool, default=False
        Whether to scale the ridge penalty per asset using the in-sample
        standard deviation of each return series, mirroring the R code's
        ``penalty.factor = apply(Y_train, 2, sd)`` behavior.
    rhs_init_cov_sample : float, default=0.0
        Fraction of in-sample rows used to estimate the initialization
        covariance for the inverse-covariance portfolio. ``0.0`` matches the
        R default and means full-sample covariance.
    rhs_init_cov_sample_shrinkage : {"none", "lw03"}, default="lw03"
        Shrinkage fallback used only when the initialization covariance is
        singular or numerically unstable. ``"lw03"`` maps to Ledoit-Wolf
        shrinkage, which is the closest Python analogue to the R setup.
    random_forest_n_estimators : int, default=1500
        Number of trees used by the random forest regressor.
    random_forest_mtry_denom : int, default=10
        Denominator used to resolve the integer ``mtry`` rule
        ``floor(n_features / random_forest_mtry_denom)``.

        We intentionally avoid relying on sklearn's fractional
        ``max_features`` shortcut here. The R implementation thinks in terms
        of an integer candidate-feature count per split, not a floating-point
        fraction interpreted by backend-specific heuristics. Resolving the
        integer ``mtry`` ourselves keeps the intent stable across sklearn and
        cuML and makes diagnostics easier to interpret.
    random_forest_max_depth : int or None, default=None
        Optional maximum tree depth.
    random_forest_min_node_size : int, default=200
        Parity-oriented public knob for the paper's large terminal-node size.

        In the R implementation this concept is expressed as
        ``ranger::min.node.size``. sklearn does not expose the same control
        directly, so we approximate it with:

        - ``min_samples_leaf = random_forest_min_node_size``
        - ``min_samples_split = 2 * random_forest_min_node_size``

        This is not exact ``ranger`` parity, but it preserves the intent
        better than exposing sklearn's leaf-size parameter directly as the
        public contract.
    random_state : int, default=1234
        Random seed used across estimator components.
    log_every_n_iterations : int, default=1
        Logging cadence for the alternating fit loop.
    """

    variant: Literal["mace"] = "mace"
    device: Literal["cpu", "cuda"] = "cpu"
    stopping_rule: Literal["max_iterations", "tolerance"] = "max_iterations"
    selection_rule: Literal["last_iteration", "best_oob"] = "last_iteration"
    selection_smoothing_window: int = Field(default=5, ge=1)
    max_iterations: int = Field(default=100, ge=1)
    min_iterations: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    tolerance: float = Field(default=1e-4, gt=0)
    ridge_alpha: float = Field(default=0.02**2 * 42, gt=0)
    ridge_alpha_selection: Literal["fixed", "cv", "target_r2"] = "fixed"
    ridge_alpha_grid: tuple[float, ...] | None = None
    ridge_alpha_min: float = Field(default=1e-4, gt=0.0)
    ridge_alpha_max: float = Field(default=10.0, gt=0.0)
    ridge_alpha_grid_size: int = Field(default=100, ge=2)
    ridge_target_r2: float = Field(default=0.01, ge=0.0, le=1.0)
    lambda_tranquilizer: float = Field(default=1.0, gt=0.0)
    ridge_n_jobs: int = Field(default=-1)
    ridge_lower_bound: float | None = -3.0
    ridge_upper_bound: float | None = None
    ridge_stock_specific_penalty: bool = False
    rhs_init_cov_sample: float = Field(default=0.0, ge=0.0, le=1.0)
    rhs_init_cov_sample_shrinkage: Literal["none", "lw03"] = "lw03"
    random_forest_n_estimators: int = Field(default=1500, ge=1)
    random_forest_mtry_denom: int = Field(default=10, ge=1)
    random_forest_max_depth: int | None = Field(default=None, ge=1)
    random_forest_min_node_size: int = Field(default=200, ge=1)
    random_state: int = 1234
    log_every_n_iterations: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_ridge_alpha_path(self) -> "MaceModelConfig":
        if self.ridge_alpha_max <= self.ridge_alpha_min:
            raise ValueError("ridge_alpha_max must be strictly greater than ridge_alpha_min.")
        return self

    def resolved_ridge_alpha_grid(self) -> tuple[float, ...]:
        if self.ridge_alpha_grid is not None:
            return tuple(float(alpha) for alpha in self.ridge_alpha_grid)
        return tuple(
            float(alpha)
            for alpha in np.geomspace(
                self.ridge_alpha_min,
                self.ridge_alpha_max,
                self.ridge_alpha_grid_size,
            )
        )
