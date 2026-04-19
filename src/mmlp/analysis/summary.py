"""
Summary metrics for MACE run artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmlp.model.mace import FittedMaceModel, MacePanelMatrix

__all__ = ["summarize_mace_run"]


def summarize_mace_run(
    fitted_model: FittedMaceModel,
    panel_matrix: MacePanelMatrix,
    normalized_weights: pd.Series | None = None,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Build a compact run summary for a fitted MACE model.

    Parameters
    ----------
    fitted_model : FittedMaceModel
        Fitted MACE model.
    panel_matrix : MacePanelMatrix
        Wide return matrix used during fitting.
    normalized_weights : pandas.Series or None, default=None
        Optional sum-normalized portfolio weights used for performance-facing
        summary statistics. If omitted, the fitted weights are normalized by
        their plain sum before evaluation.
    annualization_factor : int, default=252
        Annualization factor for daily returns.

    Returns
    -------
    pandas.DataFrame
        One-row summary table with convergence, return, and weight diagnostics.
    """

    weights_raw = fitted_model.weights_.astype(float)
    if normalized_weights is None:
        weight_sum = float(weights_raw.sum())
        if abs(weight_sum) <= 1e-12:
            weights = pd.Series(0.0, index=weights_raw.index, dtype=float)
        else:
            weights = weights_raw / weight_sum
    else:
        weights = normalized_weights.astype(float).reindex(weights_raw.index).fillna(0.0)

    portfolio_returns = pd.Series(
        panel_matrix.returns.to_numpy() @ weights.loc[panel_matrix.returns.columns].to_numpy(),
        index=panel_matrix.returns.index,
        name="portfolio_return",
    )
    mean_return = float(portfolio_returns.mean())
    return_std = float(portfolio_returns.std(ddof=0))
    annual_return = mean_return * annualization_factor
    annual_volatility = return_std * np.sqrt(annualization_factor)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 1e-12 else np.nan

    gross_exposure = float(np.abs(weights).sum())
    net_exposure = float(weights.sum())
    weight_square_sum = float(np.sum(np.square(weights)))
    effective_n = float(1.0 / weight_square_sum) if weight_square_sum > 1e-12 else np.nan
    max_weight = float(weights.max())
    min_weight = float(weights.min())

    diagnostics = fitted_model.diagnostics_
    selected_delta = np.nan
    if not diagnostics.empty:
        selected_row = diagnostics.loc[
            diagnostics["iteration"] == fitted_model.selected_iteration_
        ]
        if not selected_row.empty:
            selected_delta = float(selected_row["latent_delta_rmse"].iloc[0])
        else:
            selected_delta = float(diagnostics["latent_delta_rmse"].iloc[-1])

    return pd.DataFrame(
        [
            {
                "n_observations": len(panel_matrix.returns),
                "n_assets": len(panel_matrix.asset_ids),
                "n_iterations": len(diagnostics),
                "selected_iteration": fitted_model.selected_iteration_,
                "selection_rule": fitted_model.selection_rule_,
                "selection_metric": fitted_model.selection_metric_,
                "selection_score": fitted_model.selection_score_,
                "selected_latent_delta_rmse": selected_delta,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "effective_n": effective_n,
                "max_weight": max_weight,
                "min_weight": min_weight,
            }
        ]
    )
