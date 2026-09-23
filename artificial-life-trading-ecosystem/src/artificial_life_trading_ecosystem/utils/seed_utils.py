"""Deterministic random seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:  # pragma: no cover - optional until forecasting model is implemented
    tf = None
    keras = None


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and TensorFlow/Keras when available."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if keras is not None:
        keras.utils.set_random_seed(seed)
    elif tf is not None:
        tf.random.set_seed(seed)


__all__ = ["set_deterministic_seed"]
