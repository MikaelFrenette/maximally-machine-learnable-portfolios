from __future__ import annotations

import logging

from mmlp.logger import LogLevel, Verbosity, get_logger


def test_get_logger_uses_verbosity_mapping() -> None:
    logger = get_logger("mmlp.tests.logger", verbosity=Verbosity.DEBUG)
    assert logger.level == logging.DEBUG


def test_get_logger_respects_explicit_level() -> None:
    logger = get_logger("mmlp.tests.level", level=LogLevel.ERROR, verbosity=Verbosity.DEBUG)
    assert logger.level == logging.ERROR
