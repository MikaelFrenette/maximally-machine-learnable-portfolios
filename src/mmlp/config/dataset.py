"""
Dataset configuration models for MMLP workflows.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from mmlp.config.models import BaseConfigModel

__all__ = ["DatasetConfig", "UniverseConfig"]


class UniverseConfig(BaseConfigModel):
    """
    Universe selection settings for a run.

    Parameters
    ----------
    path : Path or None, default=None
        Optional path to an ordered universe definition file such as
        ``mace_universe.json``. This is required for providers that source the
        universe externally, such as Yahoo-based extraction.
    size : int or None, default=None
        Optional ordered subset size. This can be used to define variants such
        as MACE20 from the top-ranked names.
    """

    path: Path | None = None
    size: int | None = Field(default=None, ge=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: Path | None) -> Path | None:
        """
        Validate the universe file path.
        """

        if value is None:
            return value
        if not str(value).strip():
            raise ValueError("universe.path must be a non-empty path.")
        return value


class DatasetConfig(BaseConfigModel):
    """
    Dataset extraction settings for a run.

    Parameters
    ----------
    provider : {"yahoo", "mace_paper_csv"}
        Data provider used for extraction.
    path : pathlib.Path or None, default=None
        Local dataset path used by providers that read from disk, such as
        ``"mace_paper_csv"``.
    calendar : str, default="XNYS"
        Master market calendar identifier.
    price_field : str, default="Adj Close"
        Price field used to derive returns.
    start_date : datetime.date
        Inclusive start date.
    end_date : datetime.date
        Inclusive end date.
    drop_missing : bool, default=False
        Whether to drop rows with missing values after return computation.
    auto_adjust : bool, default=False
        Whether price downloads should be auto-adjusted at source.
    progress : bool, default=False
        Whether to show provider progress output.
    universe : UniverseConfig
        Universe selection settings.
    """

    provider: Literal["yahoo", "mace_paper_csv"] = "yahoo"
    path: Path | None = None
    calendar: str = "XNYS"
    price_field: str = Field(default="Adj Close")
    start_date: date
    end_date: date
    drop_missing: bool = False
    auto_adjust: bool = False
    progress: bool = False
    universe: UniverseConfig

    @field_validator("calendar")
    @classmethod
    def validate_calendar(cls, value: str) -> str:
        """
        Normalize the market calendar identifier.
        """

        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("calendar must be a non-empty string.")
        return normalized

    @field_validator("price_field")
    @classmethod
    def validate_price_field(cls, value: str) -> str:
        """
        Validate the requested price field.
        """

        if value != "Adj Close":
            raise ValueError("Only 'Adj Close' is currently supported.")
        return value

    @model_validator(mode="after")
    def validate_dates(self) -> "DatasetConfig":
        """
        Validate the configured date bounds.
        """

        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date.")
        if self.provider == "yahoo":
            if self.universe.path is None:
                raise ValueError("dataset.universe.path is required when provider='yahoo'.")
            if self.universe.path.suffix.lower() != ".json":
                raise ValueError("universe.path must point to a JSON universe file.")
        if self.provider == "mace_paper_csv":
            if self.path is None:
                raise ValueError("dataset.path is required when provider='mace_paper_csv'.")
            if self.path.suffix.lower() != ".csv":
                raise ValueError("dataset.path must point to a CSV file.")
        return self
