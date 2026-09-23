"""Logging utilities for the project."""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Create or retrieve a logger configured for the project."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


__all__ = ["get_logger"]
