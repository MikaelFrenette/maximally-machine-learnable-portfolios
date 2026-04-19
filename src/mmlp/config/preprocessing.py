"""
Preprocessing configuration models for MMLP workflows.
"""

from __future__ import annotations

from pydantic import Field

from mmlp.config.models import BaseConfigModel

__all__ = ["PreprocessingConfig"]


class PreprocessingConfig(BaseConfigModel):
    """
    Preprocessing and feature-engineering settings.

    Parameters
    ----------
    return_column : str, default="return"
        Canonical return column name in long-format panels.
    date_column : str, default="date"
        Canonical date column name.
    id_column : str, default="asset_id"
        Canonical asset identifier column name.
    ticker_column : str, default="ticker"
        Canonical ticker column name.
    sector_column : str, default="sector"
        Canonical sector column name.
    industry_column : str, default="industry"
        Canonical industry column name.
    lags : int, default=5
        Maximum autoregressive lag count. A value of ``4`` means create
        lag features 1 through 4.
    lag_feature_prefix : str, default="lag"
        Prefix used when naming generated lag features.
    drop_rows_with_missing_lags : bool, default=True
        Whether to drop rows that do not yet have a complete lag history.
    sort_panel : bool, default=True
        Whether canonical panels should be sorted by asset and date.
    """

    return_column: str = Field(default="return", min_length=1)
    date_column: str = Field(default="date", min_length=1)
    id_column: str = Field(default="asset_id", min_length=1)
    ticker_column: str = Field(default="ticker", min_length=1)
    sector_column: str = Field(default="sector", min_length=1)
    industry_column: str = Field(default="industry", min_length=1)
    lags: int = Field(default=5, ge=1)
    lag_feature_prefix: str = Field(default="lag", min_length=1)
    drop_rows_with_missing_lags: bool = True
    sort_panel: bool = True
