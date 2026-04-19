"""
Panel preprocessing helpers for MMLP workflows.
"""

from __future__ import annotations

import pandas as pd

from mmlp.config.preprocessing import PreprocessingConfig

__all__ = ["build_lagged_feature_panel"]


def build_lagged_feature_panel(
    panel: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """
    Build a model-ready panel with autoregressive lag features.

    Parameters
    ----------
    panel : pandas.DataFrame
        Canonical long-format return panel.
    config : PreprocessingConfig
        Preprocessing settings controlling sort order and lag generation.

    Returns
    -------
    pandas.DataFrame
        Long-format feature panel with lag columns added.
    """

    required_columns = {
        config.date_column,
        config.id_column,
        config.return_column,
    }
    missing_columns = sorted(required_columns.difference(panel.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Panel is missing required preprocessing columns: {missing_text}")

    frame = panel.copy()
    if config.sort_panel:
        frame = frame.sort_values([config.id_column, config.date_column]).reset_index(drop=True)

    grouped_returns = frame.groupby(config.id_column, sort=False)[config.return_column]
    for lag in range(1, config.lags + 1):
        column_name = f"{config.lag_feature_prefix}_{lag}"
        frame[column_name] = grouped_returns.shift(lag)

    if config.drop_rows_with_missing_lags:
        lag_columns = [f"{config.lag_feature_prefix}_{lag}" for lag in range(1, config.lags + 1)]
        frame = frame.dropna(subset=lag_columns).reset_index(drop=True)

    return frame
