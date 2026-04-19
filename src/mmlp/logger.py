"""
Logging utilities for the MMLP rewrite.

This module provides a small, consistent logging interface with explicit
log levels and a separate verbosity concept for user-facing control.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path

__all__ = ["LogLevel", "Verbosity", "get_logger"]


class LogLevel(IntEnum):
    """
    Supported logging levels for the repository logger.

    Parameters
    ----------
    IntEnum
        Integer-backed enum compatible with the standard library logger.
    """

    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


class Verbosity(IntEnum):
    """
    User-facing verbosity presets.

    Parameters
    ----------
    IntEnum
        Integer-backed enum that maps high-level verbosity to log levels.
    """

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3

    def to_log_level(self) -> LogLevel:
        """
        Convert verbosity to a concrete log level.

        Returns
        -------
        LogLevel
            Standard logging level associated with this verbosity preset.
        """

        mapping = {
            Verbosity.QUIET: LogLevel.WARNING,
            Verbosity.NORMAL: LogLevel.INFO,
            Verbosity.VERBOSE: LogLevel.INFO,
            Verbosity.DEBUG: LogLevel.DEBUG,
        }
        return mapping[self]


def get_logger(
    name: str = "mmlp",
    *,
    level: LogLevel | int | None = None,
    verbosity: Verbosity | int = Verbosity.NORMAL,
    log_path: Path | str | None = None,
) -> logging.Logger:
    """
    Build or retrieve a configured package logger.

    Parameters
    ----------
    name : str, default="mmlp"
        Logger name.
    level : LogLevel or int or None, default=None
        Explicit log level. When omitted, the value is derived from
        ``verbosity``.
    verbosity : Verbosity or int, default=Verbosity.NORMAL
        User-facing verbosity preset.
    log_path : pathlib.Path or str or None, default=None
        Optional file path receiving a copy of the logger output.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    resolved_verbosity = Verbosity(verbosity)
    resolved_level = int(level if level is not None else resolved_verbosity.to_log_level())

    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_path is not None:
        resolved_log_path = Path(log_path)
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        has_matching_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")) == resolved_log_path.resolve()
            for handler in logger.handlers
        )
        if not has_matching_file_handler:
            file_handler = logging.FileHandler(resolved_log_path, mode="w", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.setLevel(resolved_level)
    logger.propagate = False

    return logger
