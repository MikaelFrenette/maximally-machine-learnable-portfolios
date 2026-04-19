"""
Top-level run configuration for MMLP workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from mmlp.config.dataset import DatasetConfig
from mmlp.config.io import load_yaml_config
from mmlp.config.model import MaceModelConfig
from mmlp.config.models import BaseConfigModel
from mmlp.config.plotting import PlottingConfig
from mmlp.config.preprocessing import PreprocessingConfig
from mmlp.config.split import SplitConfig
from mmlp.config.trading import TradingConfig

__all__ = ["OutputConfig", "RunConfig", "load_run_config"]


class OutputConfig(BaseConfigModel):
    """
    Output settings for a run.

    Parameters
    ----------
    log_level : {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}, default="INFO"
        Repository logger level.
    verbosity : int, default=1
        User-facing verbosity preset.
    """

    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    verbosity: int = Field(default=1, ge=0, le=3)


class RunConfig(BaseConfigModel):
    """
    Top-level run configuration for end-to-end workflows.

    Parameters
    ----------
    run_name : str
        Human-readable run identifier.
    dataset : DatasetConfig
        Dataset extraction settings.
    preprocessing : PreprocessingConfig
        Preprocessing settings.
    split : SplitConfig
        Explicit train/test date split.
    model : MaceModelConfig
        Model settings.
    trading : TradingConfig
        Trading-overlay settings.
    plotting : PlottingConfig
        Plot-generation settings.
    outputs : OutputConfig
        Output and logging settings.
    """

    run_name: str = Field(min_length=1)
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    split: SplitConfig
    model: MaceModelConfig = MaceModelConfig()
    trading: TradingConfig = TradingConfig()
    plotting: PlottingConfig = PlottingConfig()
    outputs: OutputConfig

    @model_validator(mode="after")
    def validate_split_within_dataset(self) -> "RunConfig":
        if self.split.train_start < self.dataset.start_date:
            raise ValueError("split.train_start must be within dataset date range.")
        if self.split.test_end > self.dataset.end_date:
            raise ValueError("split.test_end must be within dataset date range.")
        return self


def load_run_config(path: Path | str) -> RunConfig:
    """
    Load and validate a run configuration file.
    """

    return load_yaml_config(path=path, model_type=RunConfig)
