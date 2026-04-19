"""
Extraction configuration models for raw feature generation workflows.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from mmlp.config.io import load_yaml_config
from mmlp.config.models import BaseConfigModel

__all__ = ["ExtractConfig", "load_extract_config"]


class ExtractConfig(BaseConfigModel):
    """
    User-facing configuration for Yahoo-based raw feature extraction.

    Parameters
    ----------
    tickers : tuple of str
        Ticker symbols to download.
    start_date : datetime.date
        Inclusive start date.
    end_date : datetime.date
        Inclusive end date.
    output_path : pathlib.Path
        Destination CSV path for the extracted feature panel.
    calendar : str, default="XNYS"
        Master calendar identifier used to normalize sessions.
    drop_missing : bool, default=False
        Whether to drop rows with any missing returns.
    auto_adjust : bool, default=False
        Whether Yahoo download requests should auto-adjust prices.
    progress : bool, default=False
        Whether to show Yahoo download progress output.
    """

    tickers: tuple[str, ...]
    start_date: date
    end_date: date
    output_path: Path
    calendar: str = "XNYS"
    drop_missing: bool = False
    auto_adjust: bool = False
    progress: bool = False
    price_field: str = Field(default="Adj Close")

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """
        Validate ticker symbols for extraction.
        """

        if not value:
            raise ValueError("tickers must contain at least one symbol.")
        normalized = tuple(ticker.strip().upper() for ticker in value)
        if any(not ticker for ticker in normalized):
            raise ValueError("tickers must not contain empty values.")
        return normalized

    @field_validator("calendar")
    @classmethod
    def validate_calendar(cls, value: str) -> str:
        """
        Normalize the configured market calendar identifier.
        """

        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("calendar must be a non-empty string.")
        return normalized

    @field_validator("output_path")
    @classmethod
    def validate_output_path(cls, value: Path) -> Path:
        """
        Validate the configured output path.
        """

        if not value.suffix:
            raise ValueError("output_path must include a file extension.")
        return value

    @field_validator("price_field")
    @classmethod
    def validate_price_field(cls, value: str) -> str:
        """
        Validate the requested Yahoo price field.
        """

        if value != "Adj Close":
            raise ValueError("Only 'Adj Close' is currently supported.")
        return value

    @model_validator(mode="after")
    def validate_dates(self) -> "ExtractConfig":
        """
        Validate the extraction date bounds.
        """

        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date.")
        return self


def load_extract_config(path: Path | str) -> ExtractConfig:
    """
    Load and validate an extraction configuration file.
    """

    return load_yaml_config(path=path, model_type=ExtractConfig)
