"""
Plotting configuration models for MMLP workflows.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mmlp.config.models import BaseConfigModel

__all__ = ["PlottingConfig"]


class PlottingConfig(BaseConfigModel):
    """
    Plot generation settings for run artifacts.
    """

    enabled: bool = True
    image_format: Literal["png"] = "png"
    dpi: int = Field(default=150, ge=72, le=600)
    benchmark_ticker: str | None = None
    benchmark_label: str = "S&P 500"
