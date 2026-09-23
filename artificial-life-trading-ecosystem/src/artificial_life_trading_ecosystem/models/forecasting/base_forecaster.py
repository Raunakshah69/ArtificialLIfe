"""Shared forecasting backbone interface and common serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:  # pragma: no cover - TensorFlow is required for forecasting work.
    tf = None
    keras = None

from artificial_life_trading_ecosystem.config import settings
from artificial_life_trading_ecosystem.utils.seed_utils import set_deterministic_seed


@dataclass
class ForecastingModelConfig:
    """Configuration for the shared forecasting backbone."""

    model_type: str = "SimpleRNN"
    sequence_length: int = 30
    hidden_units: int = 64
    dropout: float = 0.20
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss: str = "mse"
    epochs: int = 20
    batch_size: int = 32
    early_stopping_patience: int = 5
    random_seed: int = 42
    target_mode: str = "regression"
    feature_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SharedForecaster:
    """Abstract shared forecasting interface used by all recurrent model variants."""

    def __init__(
        self,
        config: ForecastingModelConfig | None = None,
        *,
        input_shape: tuple[int, int] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.config = config or ForecastingModelConfig()
        self.model_type = self.config.model_type
        self.input_shape = input_shape or (self.config.sequence_length, 0)
        self.feature_names = feature_names or self.config.feature_names
        self.forecaster_mode = "trained"
        self.model = None
        self.training_history_: dict[str, list[float]] = {}
        self.validation_metrics_: dict[str, float] = {}
        self.training_started_at_: str | None = None
        self.training_completed_at_: str | None = None
        self.random_seed_ = self.config.random_seed or settings.seed
        set_deterministic_seed(self.random_seed_)

    def build(self, input_shape: tuple[int, int] | None = None) -> Any:
        raise NotImplementedError("Each recurrent architecture must implement build().")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray | None = None,
        y_validation: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        if keras is None or tf is None:
            raise ImportError("TensorFlow is required to train the forecasting backbone.")

        if self.model is None:
            self.build((X_train.shape[1], X_train.shape[2]))

        self.training_started_at_ = datetime.now(timezone.utc).isoformat()
        callbacks = []
        if X_validation is not None and y_validation is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                )
            )

        fit_kwargs: dict[str, Any] = {
            "x": X_train,
            "y": y_train,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "verbose": 0,
        }
        if X_validation is not None and y_validation is not None:
            fit_kwargs["validation_data"] = (X_validation, y_validation)
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        history = self.model.fit(**fit_kwargs)
        self.training_history_ = history.history
        self.training_completed_at_ = datetime.now(timezone.utc).isoformat()
        return self.training_history_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("The model has not been built before prediction.")
        predictions = self.model.predict(X, verbose=0)
        return np.asarray(predictions, dtype=np.float32).reshape(-1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        if self.model is None:
            raise ValueError("The model has not been built before evaluation.")
        metrics = self.model.evaluate(X, y, verbose=0, return_dict=True)
        self.validation_metrics_ = {
            "mae": float(metrics.get("mae", 0.0)),
            "rmse": float(np.sqrt(metrics.get("mse", 0.0))),
            "mse": float(metrics.get("mse", 0.0)),
            "loss": float(metrics.get("loss", 0.0)),
        }
        return self.validation_metrics_

    def parameter_count(self) -> dict[str, Any]:
        if self.model is None:
            return {
                "model_type": self.model_type,
                "total": 0,
                "trainable": 0,
                "non_trainable": 0,
            }

        return {
            "model_type": self.model_type,
            "total": int(self.model.count_params()),
            "trainable": int(sum(np.prod(v.shape) for v in self.model.trainable_variables)),
            "non_trainable": int(sum(np.prod(v.shape) for v in self.model.non_trainable_variables)),
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("The model must be built before it can be saved.")

        target_dir = Path(path)
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / "model.keras"
        metadata_path = target_dir / "metadata.json"
        self.model.save(model_path)

        payload = {
            "model_type": self.model_type,
            "architecture": self.model.to_json(),
            "feature_names": self.feature_names,
            "target_mode": self.config.target_mode,
            "sequence_length": self.config.sequence_length,
            "scaler_metadata": {"scaler_type": "StandardScaler"},
            "random_seed": self.random_seed_,
            "training_started_at": self.training_started_at_,
            "training_completed_at": self.training_completed_at_,
            "parameter_count": self.parameter_count(),
            "validation_metrics": self.validation_metrics_,
            "training_history": self.training_history_,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def load(self, path: str | Path) -> "SharedForecaster":
        target_dir = Path(path)
        model_path = target_dir / "model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model was found at {model_path}.")

        if keras is None or tf is None:
            raise ImportError("TensorFlow is required to load the forecasting backbone.")

        self.model = keras.models.load_model(model_path)
        metadata_path = target_dir / "metadata.json"
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.model_type = payload.get("model_type", self.model_type)
            self.feature_names = payload.get("feature_names", self.feature_names)
            self.config.target_mode = payload.get("target_mode", self.config.target_mode)
            self.config.sequence_length = int(payload.get("sequence_length", self.config.sequence_length))
            self.validation_metrics_ = payload.get("validation_metrics", self.validation_metrics_)
            self.training_history_ = payload.get("training_history", self.training_history_)
        return self


__all__ = ["ForecastingModelConfig", "SharedForecaster"]
