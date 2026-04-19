"""
Pydantic configuration models for the MMLP rewrite.

All user-facing YAML knobs should be represented by explicit models in this
module or in submodules imported from here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["BaseConfigModel"]


class BaseConfigModel(BaseModel):
    """
    Base class for validated public configuration models.

    Parameters
    ----------
    BaseModel
        Pydantic base model with strict repository defaults.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
