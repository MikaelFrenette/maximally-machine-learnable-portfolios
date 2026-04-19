"""
Model input assembly utilities for MMLP.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mmlp.config.preprocessing import PreprocessingConfig
from mmlp.model.schema import validate_model_input_panel

__all__ = ["ModelInput", "build_model_input"]


@dataclass(slots=True)
class ModelInput:
    """
    Training-ready model input assembled from a preprocessed panel.

    Parameters
    ----------
    keys : pandas.DataFrame
        Observation keys, typically date and asset identifier.
    target : pandas.Series
        Supervised learning target.
    features : pandas.DataFrame
        Feature matrix for model fitting.
    feature_columns : tuple of str
        Feature column names in model order.
    target_column : str
        Target column name.
    """

    keys: pd.DataFrame
    target: pd.Series
    features: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str


def build_model_input(
    panel: pd.DataFrame,
    config: PreprocessingConfig,
) -> ModelInput:
    """
    Build a typed model input object from a preprocessed feature panel.

    Parameters
    ----------
    panel : pandas.DataFrame
        Preprocessed long-format feature panel.
    config : PreprocessingConfig
        Preprocessing settings defining keys, target, and lag features.

    Returns
    -------
    ModelInput
        Typed model input object for downstream estimators.
    """

    validate_model_input_panel(panel=panel, config=config)

    feature_columns = tuple(
        f"{config.lag_feature_prefix}_{lag}"
        for lag in range(1, config.lags + 1)
    )
    keys = panel.loc[:, [config.date_column, config.id_column]].copy()
    target = panel.loc[:, config.return_column].copy()
    features = panel.loc[:, list(feature_columns)].copy()

    return ModelInput(
        keys=keys,
        target=target,
        features=features,
        feature_columns=feature_columns,
        target_column=config.return_column,
    )
