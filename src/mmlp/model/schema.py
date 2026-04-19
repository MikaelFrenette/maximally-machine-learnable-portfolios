"""
Model input schema validation for MMLP.
"""

from __future__ import annotations

import pandas as pd

from mmlp.config.preprocessing import PreprocessingConfig

__all__ = ["validate_model_input_panel"]


def validate_model_input_panel(
    panel: pd.DataFrame,
    config: PreprocessingConfig,
) -> None:
    """
    Validate that a preprocessed panel is suitable for model input assembly.

    Parameters
    ----------
    panel : pandas.DataFrame
        Preprocessed long-format feature panel.
    config : PreprocessingConfig
        Preprocessing settings defining required columns and lag semantics.

    Returns
    -------
    None
        Validation succeeds silently and raises on failure.
    """

    required_columns = {
        config.date_column,
        config.id_column,
        config.return_column,
        *(
            f"{config.lag_feature_prefix}_{lag}"
            for lag in range(1, config.lags + 1)
        ),
    }
    missing_columns = sorted(required_columns.difference(panel.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Model input panel is missing required columns: {missing_text}")

    duplicate_mask = panel.duplicated(
        subset=[config.date_column, config.id_column],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "Model input panel contains duplicate (date, asset_id) observations."
        )

    if panel.empty:
        raise ValueError("Model input panel must contain at least one observation.")
