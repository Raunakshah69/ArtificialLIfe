"""Manual deterministic baseline agents."""

from __future__ import annotations

from typing import Any

from artificial_life_trading_ecosystem.agents.agent import Agent
from artificial_life_trading_ecosystem.config import settings


class ManualBaselineAgent(Agent):
    """Simple deterministic threshold baseline using the frozen forecast only."""

    def decide(self, forecast: float, context: list[float] | None = None) -> dict[str, Any]:
        threshold = float(self.config.get("signal_threshold", settings.signal_threshold))
        if forecast > threshold:
            action = "BUY"
        elif forecast < -threshold:
            action = "SELL"
        else:
            action = "HOLD"
        return {"action": action, "score": float(forecast), "threshold": threshold}


__all__ = ["ManualBaselineAgent"]
