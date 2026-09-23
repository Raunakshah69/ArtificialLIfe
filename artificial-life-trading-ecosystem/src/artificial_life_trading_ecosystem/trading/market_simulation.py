"""Interfaces and contracts for the trading simulation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TradeAction:
    """A normalized trade decision emitted by an agent."""

    side: str = "hold"
    quantity: float = 0.0
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketState:
    """Minimal market state for simulation steps."""

    timestamp: Any = None
    price: float = 0.0
    volume: float = 0.0
    forecast: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["TradeAction", "MarketState"]
