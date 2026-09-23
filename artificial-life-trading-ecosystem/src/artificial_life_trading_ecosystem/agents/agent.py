"""Concrete long-only trading agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from artificial_life_trading_ecosystem.config import settings
from artificial_life_trading_ecosystem.models.genomes.decision_genome import DecisionGenome


@dataclass
class Agent:
    """Deterministic single trading organism.

    The decision head is reconstructed from the genome and a frozen forecaster
    prediction plus a short recent normalization context. No gradient descent or
    mutation is used during ordinary inference or trading.
    """

    agent_id: str = "agent_0"
    generation: int = 0
    parent_a: str | None = None
    parent_b: str | None = None
    genome: DecisionGenome | None = None
    starting_capital: float = 100000.0
    cash: float | None = None
    current_capital: float | None = None
    current_position: float = 0.0
    entry_price: float | None = None
    position_quantity: float = 0.0
    trade_history: list[dict[str, Any]] = field(default_factory=list)
    equity_history: list[float] = field(default_factory=list)
    alive: bool = True
    statistics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.genome is None:
            self.genome = DecisionGenome.from_default(
                context_length=int(self.config.get("context_length", settings.context_length)),
                hidden_units=int(self.config.get("decision_hidden_units", settings.decision_hidden_units)),
                activation=str(self.config.get("decision_activation", settings.decision_activation)),
                signal_threshold=float(self.config.get("signal_threshold", settings.signal_threshold)),
                position_size=float(self.config.get("position_size", settings.position_size)),
                stop_loss=float(self.config.get("stop_loss", settings.stop_loss)),
                take_profit=float(self.config.get("take_profit", settings.take_profit)),
                transaction_cost=float(self.config.get("transaction_cost", settings.transaction_cost)),
                seed=int(self.config.get("seed", settings.seed)),
            )

        self.starting_capital = float(self.starting_capital)
        self.cash = float(self.starting_capital if self.cash is None else self.cash)
        self.current_capital = float(self.cash if self.current_capital is None else self.current_capital)
        self.current_position = float(self.current_position)
        self.position_quantity = float(self.position_quantity)
        self._validate_configuration()
        self._sync_position_capital()
        if not self.equity_history:
            self.equity_history = [self.current_capital]
        self.statistics.setdefault("trade_count", 0)
        self.statistics.setdefault("winning_trades", 0)
        self.statistics.setdefault("losing_trades", 0)
        self.statistics.setdefault("realized_pnl", 0.0)
        self.statistics.setdefault("unrealized_pnl", 0.0)
        self.statistics.setdefault("transaction_costs", 0.0)

    def _validate_configuration(self) -> None:
        params = self.genome.trading_params if self.genome else {}
        if params.get("signal_threshold", 0.0) < 0.0 or params.get("signal_threshold", 0.0) > 1.0:
            raise ValueError("signal_threshold must be in [0, 1].")
        if params.get("position_size", 0.0) <= 0.0 or params.get("position_size", 0.0) > 1.0:
            raise ValueError("position_size must be in (0, 1].")
        if params.get("stop_loss", 0.0) <= 0.0 or params.get("stop_loss", 0.0) >= 1.0:
            raise ValueError("stop_loss must be in (0, 1).")
        if params.get("take_profit", 0.0) <= 0.0 or params.get("take_profit", 0.0) >= 1.0:
            raise ValueError("take_profit must be in (0, 1).")
        if params.get("transaction_cost", 0.0) < 0.0 or params.get("transaction_cost", 0.0) >= 1.0:
            raise ValueError("transaction_cost must be in [0, 1).")

    @property
    def context_length(self) -> int:
        return int(self.genome.context_length if self.genome else self.config.get("context_length", settings.context_length))

    @property
    def decision_hidden_units(self) -> int:
        return int(self.genome.hidden_units if self.genome else self.config.get("decision_hidden_units", settings.decision_hidden_units))

    @property
    def threshold(self) -> float:
        return float(self.genome.trading_params.get("signal_threshold", self.config.get("signal_threshold", settings.signal_threshold)))

    @property
    def position_size_fraction(self) -> float:
        return float(self.genome.trading_params.get("position_size", self.config.get("position_size", settings.position_size)))

    @property
    def stop_loss(self) -> float:
        return float(self.genome.trading_params.get("stop_loss", self.config.get("stop_loss", settings.stop_loss)))

    @property
    def take_profit(self) -> float:
        return float(self.genome.trading_params.get("take_profit", self.config.get("take_profit", settings.take_profit)))

    @property
    def transaction_cost(self) -> float:
        return float(self.genome.trading_params.get("transaction_cost", self.config.get("transaction_cost", settings.transaction_cost)))

    def _sync_position_capital(self) -> None:
        if self.entry_price is None:
            self.current_position = 0.0
            self.position_quantity = 0.0
        self.current_capital = float(self.cash + self.position_quantity * (self.entry_price if self.entry_price is not None else 0.0))

    def _decision_input(self, forecast: float, context: list[float] | np.ndarray) -> np.ndarray:
        ctx = np.asarray(context, dtype=np.float32).reshape(-1)
        if ctx.size < self.context_length:
            pad = np.zeros(self.context_length - ctx.size, dtype=np.float32)
            ctx = np.concatenate([pad, ctx])
        elif ctx.size > self.context_length:
            ctx = ctx[-self.context_length:]
        return np.concatenate([[float(forecast)], ctx.astype(np.float32)])

    def decision_score(self, forecast: float, context: list[float] | np.ndarray) -> float:
        x = self._decision_input(forecast, context)
        input_dim = self.context_length + 1
        w_in = np.asarray(self.genome.input_to_hidden, dtype=np.float32).reshape(input_dim, self.decision_hidden_units)
        hidden_bias = np.asarray(self.genome.hidden_bias, dtype=np.float32)
        hidden_out = np.asarray(self.genome.hidden_to_output, dtype=np.float32)
        output_bias = float(self.genome.output_bias)

        hidden_linear = x @ w_in + hidden_bias
        hidden = np.tanh(hidden_linear)
        score = float(hidden @ hidden_out + output_bias)
        return float(np.clip(score, -1.0, 1.0))

    def decide(self, forecast: float, context: list[float] | np.ndarray) -> dict[str, Any]:
        score = self.decision_score(forecast, context)
        ctx = np.asarray(context, dtype=np.float32).reshape(-1)
        trend_component = float(np.mean(ctx[-min(5, ctx.size):])) if ctx.size else 0.0
        signal = float(forecast) + 0.25 * score + 0.05 * trend_component

        rising = bool(ctx.size >= 2 and float(ctx[-1]) >= float(np.mean(ctx[-min(3, ctx.size):])))
        uptrend = bool(ctx.size >= 3 and float(np.mean(ctx[-3:])) > 0.0 and rising)
        downtrend = bool(ctx.size >= 3 and float(np.mean(ctx[-3:])) < 0.0 and not rising)

        if signal >= self.threshold or (forecast > 0.0 and uptrend and signal > 0.0):
            action = "BUY"
        elif signal <= -self.threshold or (forecast < 0.0 and downtrend and signal < 0.0):
            action = "SELL"
        else:
            action = "HOLD"
        return {"action": action, "score": score, "threshold": self.threshold, "signal": signal}

    def position_value(self, price: float) -> float:
        return self.position_quantity * float(price)

    def desired_position_quantity(self, price: float) -> float:
        if price <= 0:
            return 0.0
        max_buyable = (self.cash / (price * (1.0 + self.transaction_cost))) if self.cash > 0 else 0.0
        target_cash = max(self.cash * self.position_size_fraction, 0.0)
        return max(0.0, min(int(np.floor(target_cash / (price * (1.0 + self.transaction_cost)))), int(np.floor(max_buyable))))

    def record_trade(
        self,
        *,
        timestamp: Any,
        action: str,
        price: float,
        quantity: float,
        transaction_cost: float,
        realized_pnl: float,
        cash_after: float,
        equity_after: float,
        reason: str,
    ) -> None:
        entry = {
            "agent_id": self.agent_id,
            "timestamp": timestamp,
            "action": action,
            "price": float(price),
            "quantity": float(quantity),
            "transaction_cost": float(transaction_cost),
            "realized_pnl": float(realized_pnl),
            "cash_after": float(cash_after),
            "equity_after": float(equity_after),
            "reason": reason,
        }
        self.trade_history.append(entry)
        self.statistics["trade_count"] = int(len(self.trade_history))
        self.statistics["transaction_costs"] = float(sum(item["transaction_cost"] for item in self.trade_history))
        if realized_pnl > 0:
            self.statistics["winning_trades"] = int(self.statistics.get("winning_trades", 0) + 1)
        elif realized_pnl < 0:
            self.statistics["losing_trades"] = int(self.statistics.get("losing_trades", 0) + 1)
        self.statistics["realized_pnl"] = float(sum(item["realized_pnl"] for item in self.trade_history))

    def _append_equity(self, price: float | None = None) -> None:
        if price is None:
            price = self.entry_price if self.entry_price is not None else 0.0
        self.current_capital = float(self.cash + self.position_quantity * price)
        self.equity_history.append(self.current_capital)

    def __repr__(self) -> str:
        return f"Agent(agent_id={self.agent_id}, cash={self.cash:.2f}, position={self.position_quantity})"


__all__ = ["Agent"]
