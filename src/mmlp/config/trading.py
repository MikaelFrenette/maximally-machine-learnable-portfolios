"""
Trading configuration models for MMLP workflows.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mmlp.config.models import BaseConfigModel

__all__ = ["TradingConfig"]


class TradingConfig(BaseConfigModel):
    """
    Mean-variance trading overlay settings.

    Parameters
    ----------
    enabled : bool, default=True
        Whether to compute trading-overlay artifacts.
    vol_model : {"ewma"}, default="ewma"
        Volatility model used in the overlay.
    lookback : int, default=2520
        Lookback window used for prevailing-mean and variance estimation.
    gamma : float, default=5.0
        Relative risk aversion parameter.
    alpha : float, default=0.94
        EWMA decay parameter.
    position_min : float, default=-1.0
        Minimum allowed position.
    position_max : float, default=2.0
        Maximum allowed position.
    horizon : int, default=1
        Forecast horizon in periods.
    """

    enabled: bool = True
    vol_model: Literal["ewma"] = "ewma"
    lookback: int = Field(default=120 * 21, ge=1)
    gamma: float = Field(default=5.0, gt=0)
    alpha: float = Field(default=0.94, gt=0, lt=1)
    position_min: float = -1.0
    position_max: float = 2.0
    horizon: int = Field(default=1, ge=1)
