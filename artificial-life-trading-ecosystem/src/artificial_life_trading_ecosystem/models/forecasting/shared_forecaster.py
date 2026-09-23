"""Interface for the shared forecasting backbone."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SharedForecaster(ABC):
    """Abstract interface for a population-shared recurrent forecast model."""

    @abstractmethod
    def fit(self, features: Any, targets: Any, **kwargs: Any) -> Any:
        """Train the shared model with gradient descent."""

    @abstractmethod
    def predict(self, features: Any, **kwargs: Any) -> Any:
        """Produce market forecasts for a given feature set."""


__all__ = ["SharedForecaster"]
