"""Evaluation interfaces for ecosystem metrics and trading scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np


@dataclass
class PerformanceMetrics:
    """Summary metrics for a generation or agent run."""

    starting_capital: float = 0.0
    ending_capital: float = 0.0
    total_return: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    transaction_costs: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    maximum_drawdown: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    sharpe_ratio: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_agent(cls, agent: Any) -> "PerformanceMetrics":
        equity = list(agent.equity_history)
        starting = float(agent.starting_capital)
        ending = float(equity[-1]) if equity else starting
        pnl = float(agent.statistics.get("realized_pnl", 0.0))
        tx = float(agent.statistics.get("transaction_costs", 0.0))
        drawdown = 0.0
        peak = starting
        for value in equity:
            if value > peak:
                peak = float(value)
            drawdown = max(drawdown, (peak - value) / peak if peak else 0.0)
        trade_count = int(len(agent.trade_history))
        winning_trades = int(agent.statistics.get("winning_trades", 0))
        losing_trades = int(agent.statistics.get("losing_trades", 0))
        win_rate = (winning_trades / trade_count) if trade_count else 0.0
        returns = np.diff(np.asarray(equity, dtype=np.float64), prepend=starting)
        sharpe = None
        if returns.size > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * sqrt(252))
        return cls(
            starting_capital=starting,
            ending_capital=ending,
            total_return=((ending - starting) / starting) if starting else 0.0,
            realized_pnl=pnl,
            unrealized_pnl=float(agent.current_capital - agent.cash),
            transaction_costs=tx,
            trade_count=trade_count,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            maximum_drawdown=drawdown,
            equity_curve=equity,
            sharpe_ratio=sharpe,
        )


__all__ = ["PerformanceMetrics"]
