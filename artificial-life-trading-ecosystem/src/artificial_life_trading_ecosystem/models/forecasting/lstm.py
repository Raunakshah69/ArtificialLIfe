"""LSTM forecasting model implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from tensorflow import keras
except ImportError:  # pragma: no cover
    keras = None

from artificial_life_trading_ecosystem.models.forecasting.base_forecaster import ForecastingModelConfig, SharedForecaster
from artificial_life_trading_ecosystem.utils.seed_utils import set_deterministic_seed


class LSTM(SharedForecaster):
    """Long short-term memory model for next-return forecasting."""

    def __init__(self, config: ForecastingModelConfig | None = None, *, input_shape: tuple[int, int] | None = None, feature_names: list[str] | None = None) -> None:
        super().__init__(config=config, input_shape=input_shape, feature_names=feature_names)
        self.model_type = "LSTM"
        self.config.model_type = "LSTM"

    def build(self, input_shape: tuple[int, int] | None = None) -> Any:
        if keras is None:
            raise ImportError("TensorFlow is required to build an LSTM forecasting model.")

        shape = input_shape or self.input_shape
        if shape[1] == 0:
            raise ValueError("Input feature count is required to build the forecasting model.")

        set_deterministic_seed(self.random_seed_)
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(shape[0], shape[1])),
                keras.layers.LSTM(self.config.hidden_units, dropout=self.config.dropout),
                keras.layers.Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=self.config.loss,
            metrics=["mae"],
        )
        self.model = model
        return self.model


__all__ = ["LSTM"]
