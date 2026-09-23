"""Base interfaces for simulated trading agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Minimal state held by a trading agent."""

    capital: float = 0.0
    cash: float = 0.0
    position: float = 0.0
    generation: int = 0
    alive: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract trading agent interface."""

    @abstractmethod
    def decide(self, market_features: Any, forecast: Any) -> dict[str, Any]:
        """Return a decision payload for the current market snapshot."""

    @abstractmethod
    def update_state(self, state: AgentState) -> None:
        """Apply the latest simulation state to the agent."""


__all__ = ["AgentState", "BaseAgent"]
