"""
Date-driven train/test split configuration.
"""

from __future__ import annotations

from datetime import date

from pydantic import model_validator

from mmlp.config.models import BaseConfigModel

__all__ = ["SplitConfig"]


class SplitConfig(BaseConfigModel):
    """
    Explicit date-driven train/test split settings.
    """

    train_start: date
    train_end: date
    test_start: date
    test_end: date

    @model_validator(mode="after")
    def validate_dates(self) -> "SplitConfig":
        if self.train_start > self.train_end:
            raise ValueError("train_start must be earlier than or equal to train_end.")
        if self.test_start > self.test_end:
            raise ValueError("test_start must be earlier than or equal to test_end.")
        if self.train_end >= self.test_start:
            raise ValueError("train_end must be earlier than test_start.")
        return self
