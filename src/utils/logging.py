"""Logging utilities for scripts and training loops."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: str | Path,
    log_name: str = "run.log",
    level: int = logging.INFO,
    overwrite_handlers: bool = False,
) -> logging.Logger:
    """Create a logger that writes to both console and a file.

    Parameters
    ----------
    log_dir:
        Directory for log file.
    log_name:
        Name of the log file and logger.
    level:
        Logging level.
    overwrite_handlers:
        If ``True``, removes existing handlers before attaching new ones.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    if overwrite_handlers:
        logger.handlers.clear()

    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_dir / log_name, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def get_logger(name: str = "cast_gnn") -> logging.Logger:
    """Return a standard logger by name."""
    return logging.getLogger(name)
