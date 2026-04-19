"""
Backend selection and estimator adapters for MMLP model components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mmlp.config.model import MaceModelConfig

__all__ = [
    "RegressorAdapter",
    "ModelBackend",
    "resolve_model_backend",
]


class RegressorAdapter(Protocol):
    """
    Minimal regressor interface used by the alternating estimator.
    """

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RegressorAdapter":
        """
        Fit the regressor on the provided feature matrix and target.
        """

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate predictions for the provided feature matrix.
        """


@dataclass(slots=True)
class ModelBackend:
    """
    Concrete estimator constructors for a configured execution backend.

    Parameters
    ----------
    ridge_factory : callable
        Factory producing a ridge-style regressor adapter.
    random_forest_factory : callable
        Factory producing a random-forest regressor adapter.
    name : str
        Human-readable backend identifier.
    """

    ridge_factory: callable
    random_forest_factory: callable
    name: str


def resolve_model_backend(config: MaceModelConfig) -> ModelBackend:
    """
    Resolve estimator factories for the configured execution device.

    Parameters
    ----------
    config : MaceModelConfig
        Model settings including the requested device.

    Returns
    -------
    ModelBackend
        Backend-specific estimator factories.
    """

    if config.device == "cpu":
        return _build_sklearn_backend(config=config)
    if config.device == "cuda":
        return _build_cuml_backend(config=config)
    raise ValueError(f"Unsupported device: {config.device}")


def _build_sklearn_backend(config: MaceModelConfig) -> ModelBackend:
    """
    Build sklearn-based estimator factories.
    """

    from joblib import Parallel, delayed
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV

    class SklearnRidgeAdapter:
        def __init__(self) -> None:
            self._model: object | None = None

        def fit(self, features: pd.DataFrame, target: pd.Series) -> "SklearnRidgeAdapter":
            self._model = _fit_cpu_ridge_with_optional_bounds(
                features=features,
                target=target,
                alpha=config.ridge_alpha,
                alpha_selection=config.ridge_alpha_selection,
                alpha_grid=config.resolved_ridge_alpha_grid(),
                target_r2=config.ridge_target_r2,
                lambda_tranquilizer=config.lambda_tranquilizer,
                n_jobs=config.ridge_n_jobs,
                lower_bound=config.ridge_lower_bound,
                upper_bound=config.ridge_upper_bound,
                stock_specific_penalty=config.ridge_stock_specific_penalty,
                random_state=config.random_state,
                parallel_cls=Parallel,
                delayed_fn=delayed,
                sklearn_ridge_cls=Ridge,
                sklearn_grid_search_cv_cls=GridSearchCV,
            )
            return self

        def predict(self, features: pd.DataFrame) -> pd.Series:
            if self._model is None:  # pragma: no cover - defensive
                raise ValueError("Ridge adapter must be fit before calling predict.")
            return pd.Series(
                self._model.predict(features),
                index=features.index,
                name="ridge_prediction",
            )

    class SklearnRandomForestAdapter:
        def __init__(self) -> None:
            # We resolve the integer mtry rule explicitly instead of using a
            # fractional max_features shortcut. That keeps the paper intent
            # backend-agnostic: number of candidate features per split.
            #
            # We also keep the public node-size knob in paper terms
            # (min_node_size) rather than exposing sklearn's leaf-specific API
            # directly. The mapping below is a parity-oriented approximation of
            # ranger::min.node.size, not a claim of exact equivalence.
            self._model = RandomForestRegressor(
                n_estimators=config.random_forest_n_estimators,
                bootstrap=True,
                oob_score=True,
                max_depth=config.random_forest_max_depth,
                min_samples_leaf=config.random_forest_min_node_size,
                min_samples_split=2 * config.random_forest_min_node_size,
                random_state=config.random_state,
                n_jobs=-1,
            )

        def fit(
            self,
            features: pd.DataFrame,
            target: pd.Series,
        ) -> "SklearnRandomForestAdapter":
            self._model.set_params(
                max_features=_resolve_random_forest_mtry(
                    n_features=features.shape[1],
                    mtry_denom=config.random_forest_mtry_denom,
                )
            )
            self._model.fit(features, target)
            return self

        def predict(self, features: pd.DataFrame) -> pd.Series:
            return pd.Series(
                self._model.predict(features),
                index=features.index,
                name="rf_prediction",
            )

    return ModelBackend(
        ridge_factory=SklearnRidgeAdapter,
        random_forest_factory=SklearnRandomForestAdapter,
        name="sklearn",
    )


def _build_cuml_backend(config: MaceModelConfig) -> ModelBackend:
    """
    Build cuML-based estimator factories.

    Raises
    ------
    RuntimeError
        If cuML is unavailable in the current environment.
    """

    try:
        import cudf
        from cuml.ensemble import RandomForestRegressor as CuMLRandomForestRegressor
        from cuml.linear_model import Ridge as CuMLRidge
    except ImportError as error:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "device='cuda' requires cuML on a supported Linux CUDA environment. "
            "cuML is not available in the current environment."
        ) from error

    class CuMLRidgeAdapter:
        def __init__(self) -> None:
            if config.ridge_alpha_selection in {"cv", "target_r2"}:
                raise RuntimeError(
                    "device='cuda' does not currently support adaptive ridge_alpha_selection. "
                    "Use ridge_alpha_selection='fixed' or switch to device='cpu'."
                )
            if config.ridge_stock_specific_penalty:
                raise RuntimeError(
                    "device='cuda' does not currently support ridge_stock_specific_penalty=True. "
                    "Use device='cpu' for that parity feature."
                )
            if _ridge_bounds_requested(config):
                raise RuntimeError(
                    "device='cuda' does not currently support bounded ridge coefficients. "
                    "Set ridge_lower_bound and ridge_upper_bound to null-equivalent unconstrained "
                    "values or use device='cpu'."
                )
            self._model = CuMLRidge(alpha=config.ridge_alpha)

        def fit(self, features: pd.DataFrame, target: pd.Series) -> "CuMLRidgeAdapter":
            clean_features = _ensure_finite_frame(features)
            clean_target = _ensure_finite_series(target).loc[clean_features.index]
            self._model.fit(cudf.DataFrame.from_pandas(clean_features), cudf.Series(clean_target))
            return self

        def predict(self, features: pd.DataFrame) -> pd.Series:
            clean_features = _ensure_finite_frame(features)
            predictions = self._model.predict(
                cudf.DataFrame.from_pandas(clean_features)
            ).to_pandas()
            return pd.Series(predictions, index=clean_features.index, name="ridge_prediction")

    class CuMLRandomForestAdapter:
        def __init__(self) -> None:
            # Same rationale as sklearn: resolve integer mtry directly rather
            # than letting each backend interpret a float proportion.
            #
            # For node size we keep the public knob in paper terms and map it
            # into the closest backend controls available.
            self._model = CuMLRandomForestRegressor(
                n_estimators=config.random_forest_n_estimators,
                max_depth=config.random_forest_max_depth or 16,
                min_samples_leaf=config.random_forest_min_node_size,
                min_samples_split=2 * config.random_forest_min_node_size,
                random_state=config.random_state,
            )

        def fit(self, features: pd.DataFrame, target: pd.Series) -> "CuMLRandomForestAdapter":
            clean_features = _ensure_finite_frame(features)
            clean_target = _ensure_finite_series(target).loc[clean_features.index]
            self._model.set_params(
                max_features=_resolve_random_forest_mtry(
                    n_features=clean_features.shape[1],
                    mtry_denom=config.random_forest_mtry_denom,
                )
            )
            self._model.fit(cudf.DataFrame.from_pandas(clean_features), cudf.Series(clean_target))
            return self

        def predict(self, features: pd.DataFrame) -> pd.Series:
            clean_features = _ensure_finite_frame(features)
            predictions = self._model.predict(
                cudf.DataFrame.from_pandas(clean_features)
            ).to_pandas()
            return pd.Series(predictions, index=clean_features.index, name="rf_prediction")

    return ModelBackend(
        ridge_factory=CuMLRidgeAdapter,
        random_forest_factory=CuMLRandomForestAdapter,
        name="cuml",
    )


def _ensure_finite_frame(values: pd.DataFrame) -> pd.DataFrame:
    clean = values.apply(pd.to_numeric, errors="coerce").astype(float)
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(axis=0, how="any")
    if clean.empty:
        raise ValueError("Estimator features are empty after removing non-finite rows.")
    return clean


def _ensure_finite_series(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").astype(float)
    clean = clean.replace([np.inf, -np.inf], np.nan)
    finite = clean.dropna()
    if finite.empty:
        return pd.Series(np.zeros(len(clean)), index=clean.index, name=values.name, dtype=float)
    fill_value = float(finite.mean())
    return pd.Series(
        clean.fillna(fill_value).to_numpy(dtype=float),
        index=clean.index,
        name=values.name,
        dtype=float,
    )


def _resolve_random_forest_mtry(n_features: int, mtry_denom: int) -> int:
    if n_features <= 0:
        raise ValueError("n_features must be positive when resolving random forest mtry.")
    return max(1, n_features // mtry_denom)


@dataclass(slots=True)
class _BoundedRidgeModel:
    coef_: np.ndarray
    intercept_: float
    alpha_: float | None = None
    target_r2_: float | None = None
    selected_r2_: float | None = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.intercept_ + features.to_numpy(dtype=float) @ self.coef_


def _ridge_bounds_requested(config: MaceModelConfig) -> bool:
    return config.ridge_lower_bound is not None or config.ridge_upper_bound is not None


def _fit_cpu_ridge_with_optional_bounds(
    features: pd.DataFrame,
    target: pd.Series,
    alpha: float,
    alpha_selection: str,
    alpha_grid: tuple[float, ...],
    target_r2: float,
    lambda_tranquilizer: float,
    n_jobs: int,
    lower_bound: float | None,
    upper_bound: float | None,
    stock_specific_penalty: bool,
    random_state: int,
    parallel_cls: type,
    delayed_fn: callable,
    sklearn_ridge_cls: type,
    sklearn_grid_search_cv_cls: type,
) -> object:
    clean_features = _ensure_finite_frame(features)
    clean_target = _ensure_finite_series(target).loc[clean_features.index]
    penalty_factors = _compute_penalty_factors(
        clean_features,
        use_stock_specific_penalty=stock_specific_penalty,
    )

    if alpha_selection == "target_r2":
        return _fit_ridge_by_target_r2(
            features=clean_features,
            target=clean_target,
            alpha_grid=alpha_grid,
            target_r2=target_r2,
            lambda_tranquilizer=lambda_tranquilizer,
            n_jobs=n_jobs,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            penalty_factors=penalty_factors,
            random_state=random_state,
            parallel_cls=parallel_cls,
            delayed_fn=delayed_fn,
            sklearn_ridge_cls=sklearn_ridge_cls,
        )

    if lower_bound is None and upper_bound is None:
        if stock_specific_penalty:
            raise RuntimeError(
                "ridge_stock_specific_penalty=True currently requires the bounded CPU ridge path. "
                "Enable coefficient bounds or disable stock-specific penalties."
            )
        if alpha_selection == "cv":
            cv_folds = max(2, min(5, len(clean_features)))
            selector = sklearn_grid_search_cv_cls(
                estimator=sklearn_ridge_cls(random_state=random_state),
                param_grid={"alpha": list(alpha_grid)},
                scoring="neg_mean_squared_error",
                cv=cv_folds,
                n_jobs=n_jobs,
            )
            selector.fit(clean_features, clean_target)
            selected_alpha = float(selector.best_params_["alpha"]) * lambda_tranquilizer
            model = sklearn_ridge_cls(alpha=selected_alpha, random_state=random_state)
            model.fit(clean_features, clean_target)
            model.alpha_ = selected_alpha  # type: ignore[attr-defined]
            return model
        model = sklearn_ridge_cls(
            alpha=float(alpha) * lambda_tranquilizer,
            random_state=random_state,
        )
        model.fit(clean_features, clean_target)
        return model

    if alpha_selection == "cv":
        raise RuntimeError(
            "ridge_alpha_selection='cv' is not currently supported when ridge coefficient "
            "bounds are enabled. Disable bounds or use ridge_alpha_selection='fixed'."
        )

    return _fit_bounded_ridge_model(
        features=clean_features,
        target=clean_target,
        alpha=float(alpha) * lambda_tranquilizer,
        lower_bound=(-np.inf if lower_bound is None else float(lower_bound)),
        upper_bound=upper_bound,
        penalty_factors=penalty_factors,
    )


def _fit_ridge_by_target_r2(
    features: pd.DataFrame,
    target: pd.Series,
    alpha_grid: tuple[float, ...],
    target_r2: float,
    lambda_tranquilizer: float,
    n_jobs: int,
    lower_bound: float | None,
    upper_bound: float | None,
    penalty_factors: np.ndarray,
    random_state: int,
    parallel_cls: type,
    delayed_fn: callable,
    sklearn_ridge_cls: type,
) -> object:
    if not alpha_grid:
        raise ValueError(
            "ridge_alpha_grid must not be empty when using "
            "ridge_alpha_selection='target_r2'."
        )

    y = target.to_numpy(dtype=float)
    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))

    def evaluate_alpha(alpha: float) -> tuple[float, float, object]:
        effective_alpha = float(alpha) * lambda_tranquilizer
        if lower_bound is None and upper_bound is None:
            if np.any(np.abs(penalty_factors - 1.0) > 1e-12):
                model = _fit_bounded_ridge_model(
                    features=features,
                    target=target,
                    alpha=effective_alpha,
                    lower_bound=-np.inf,
                    upper_bound=None,
                    penalty_factors=penalty_factors,
                )
            else:
                model = sklearn_ridge_cls(alpha=effective_alpha, random_state=random_state)
                model.fit(features, target)
        else:
            model = _fit_bounded_ridge_model(
                features=features,
                target=target,
                alpha=effective_alpha,
                lower_bound=(-np.inf if lower_bound is None else float(lower_bound)),
                upper_bound=upper_bound,
                penalty_factors=penalty_factors,
            )

        predictions = np.asarray(model.predict(features), dtype=float)
        if sst <= 1e-12:
            r2 = 0.0
        else:
            ssr = float(np.sum((y - predictions) ** 2))
            r2 = 1.0 - ssr / sst
        return abs(r2 - target_r2), float(r2), model

    results = parallel_cls(n_jobs=n_jobs)(
        delayed_fn(evaluate_alpha)(alpha) for alpha in alpha_grid
    )
    best_index = min(range(len(results)), key=lambda idx: results[idx][0])
    best_distance, best_r2, best_model = results[best_index]
    best_alpha = float(alpha_grid[best_index]) * lambda_tranquilizer

    if best_model is None:  # pragma: no cover - defensive
        raise RuntimeError("Failed to select a ridge model by target R2.")

    if hasattr(best_model, "alpha_"):
        best_model.alpha_ = best_alpha  # type: ignore[attr-defined]
    elif hasattr(best_model, "alpha"):
        best_model.alpha = best_alpha  # type: ignore[attr-defined]
        if not hasattr(best_model, "alpha_"):
            best_model.alpha_ = best_alpha  # type: ignore[attr-defined]
    else:
        setattr(best_model, "alpha_", best_alpha)
    if hasattr(best_model, "target_r2_"):
        best_model.target_r2_ = target_r2  # type: ignore[attr-defined]
    if hasattr(best_model, "selected_r2_"):
        best_model.selected_r2_ = best_r2  # type: ignore[attr-defined]
    return best_model


def _fit_bounded_ridge_model(
    features: pd.DataFrame,
    target: pd.Series,
    alpha: float,
    lower_bound: float,
    upper_bound: float | None,
    penalty_factors: np.ndarray,
) -> _BoundedRidgeModel:
    """
    Fit a ridge objective with coefficient bounds on CPU.

    This is a parity-oriented approximation of the paper's ``glmnet`` setup.
    We are not reproducing the whole lambda path or penalty machinery; the
    goal is only to preserve fixed-lambda ridge behavior plus coefficient
    bounds when the config requests them.
    """

    X = features.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    n_obs, n_features = X.shape

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = float(params[0])
        coef = params[1:]
        residual = y - intercept - X @ coef
        loss = 0.5 * float(np.mean(residual**2)) + 0.5 * alpha * float(
            np.sum(penalty_factors * coef**2)
        )
        grad_intercept = -float(np.mean(residual))
        grad_coef = -(X.T @ residual) / n_obs + alpha * penalty_factors * coef
        gradient = np.concatenate(([grad_intercept], grad_coef))
        return loss, gradient

    initial_intercept = float(np.mean(y))
    initial_params = np.zeros(n_features + 1, dtype=float)
    initial_params[0] = initial_intercept

    coef_upper_bound = np.inf if upper_bound is None else float(upper_bound)
    bounds = [(None, None)] + [(float(lower_bound), coef_upper_bound) for _ in range(n_features)]

    result = minimize(
        fun=lambda params: objective(params)[0],
        x0=initial_params,
        jac=lambda params: objective(params)[1],
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success:
        raise RuntimeError(f"Bounded ridge optimization failed: {result.message}")

    fitted_params = np.asarray(result.x, dtype=float)
    return _BoundedRidgeModel(
        intercept_=float(fitted_params[0]),
        coef_=fitted_params[1:],
    )


def _compute_penalty_factors(
    features: pd.DataFrame,
    use_stock_specific_penalty: bool,
) -> np.ndarray:
    if not use_stock_specific_penalty:
        return np.ones(features.shape[1], dtype=float)
    penalty_factors = features.std(axis=0, ddof=1).to_numpy(dtype=float)
    penalty_factors[~np.isfinite(penalty_factors)] = 1.0
    penalty_factors[penalty_factors <= 1e-12] = 1.0
    return penalty_factors
