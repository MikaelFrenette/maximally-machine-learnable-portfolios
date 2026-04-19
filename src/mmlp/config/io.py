"""
YAML configuration loading helpers for MMLP.

Configuration files loaded through this module must be validated against
explicit pydantic models before being used elsewhere in the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml

from mmlp.config.models import BaseConfigModel

__all__ = ["load_yaml_config"]

ConfigModelT = TypeVar("ConfigModelT", bound=BaseConfigModel)


def load_yaml_config(path: Path | str, model_type: type[ConfigModelT]) -> ConfigModelT:
    """
    Load a YAML configuration file and validate it with pydantic.

    Parameters
    ----------
    path : Path or str
        Path to the YAML file.
    model_type : type[ConfigModelT]
        Pydantic model used to validate the loaded configuration.

    Returns
    -------
    ConfigModelT
        Validated configuration model instance.
    """

    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload is None:
        payload = {}

    return model_type.model_validate(payload)
