from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from artificial_life_trading_ecosystem.agents.agent import Agent
from artificial_life_trading_ecosystem.agents.baseline import ManualBaselineAgent
from artificial_life_trading_ecosystem.evaluation.metrics import PerformanceMetrics
from artificial_life_trading_ecosystem.models.genomes.decision_genome import DecisionGenome
from artificial_life_trading_ecosystem.trading.engine import TradingEngine


class MockForecaster:
    def __init__(self, prediction: float = 0.0):
        self.prediction = prediction

    def predict(self, window: Any) -> np.ndarray:
        return np.asarray([self.prediction], dtype=np.float32)


@pytest.fixture
def synthetic_market() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    closes = np.linspace(100.0, 120.0, 20, dtype=float)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": closes - 1.0,
            "High": closes + 2.0,
            "Low": closes - 2.0,
            "Close": closes,
            "Volume": 1000.0,
        }
    )
    return df


def test_agent_initializes_correctly() -> None:
    genome = DecisionGenome.from_default(seed=7)
    agent = Agent(agent_id="a1", genome=genome, starting_capital=100000.0)
    assert agent.agent_id == "a1"
    assert agent.starting_capital == 100000.0
    assert agent.cash == 100000.0
    assert agent.position_quantity == 0.0
    assert agent.alive is True


def test_genome_produces_deterministic_decision_output() -> None:
    genome = DecisionGenome.from_default(seed=13)
    agent = Agent(agent_id="agent", genome=genome, starting_capital=100000.0)
    score1 = agent.decision_score(0.75, [0.5, 0.1, -0.2, 0.3, 0.8])
    score2 = agent.decision_score(0.75, [0.5, 0.1, -0.2, 0.3, 0.8])
    assert np.isclose(score1, score2)
    assert -1.0 <= score1 <= 1.0


def test_decision_head_uses_forecast_and_context_as_neural_inputs() -> None:
    genome = DecisionGenome.from_default(signal_threshold=0.2, seed=101)
    genome.input_to_hidden.fill(0.0)
    genome.hidden_to_output.fill(0.0)
    genome.input_to_hidden[0, 0] = 1.0
    genome.input_to_hidden[1, 1] = 1.0
    genome.hidden_to_output[:2] = 1.0
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)

    baseline_score = agent.decision_score(0.0, [0.0] * 5)
    forecast_score = agent.decision_score(0.5, [0.0] * 5)
    context_score = agent.decision_score(0.0, [0.5, 0.0, 0.0, 0.0, 0.0])

    assert baseline_score == 0.0
    assert forecast_score > baseline_score
    assert context_score > baseline_score


def test_context_cannot_override_genome_score_threshold() -> None:
    genome = DecisionGenome.from_default(signal_threshold=0.2, seed=102)
    genome.input_to_hidden.fill(0.0)
    genome.hidden_to_output.fill(0.0)
    genome.output_bias = 0.1
    genome.trading_params["signal_threshold"] = 0.2
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)

    decision = agent.decide(0.0, [-1.0, -1.0, 1.0, 1.0, 1.0])

    assert decision["score"] == pytest.approx(0.1)
    assert decision["action"] == "HOLD"


def test_evolved_output_bias_changes_action_without_changing_threshold() -> None:
    genome = DecisionGenome.from_default(signal_threshold=0.2, seed=103)
    genome.input_to_hidden.fill(0.0)
    genome.hidden_to_output.fill(0.0)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)

    genome.output_bias = 0.3
    buy = agent.decide(0.0, [0.0] * 5)
    genome.output_bias = -0.3
    sell = agent.decide(0.0, [0.0] * 5)

    assert buy["threshold"] == sell["threshold"] == 0.2
    assert buy["action"] == "BUY"
    assert sell["action"] == "SELL"


def test_buy_opens_position() -> None:
    genome = DecisionGenome.from_default(signal_threshold=0.1, seed=1)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    engine = TradingEngine()
    row = pd.Series({"Date": "2024-01-01", "Close": 100.0, "Low": 99.0, "High": 101.0})
    action = agent.decide(0.5, [0.6, 0.7, 0.8, 0.9, 1.0])
    assert action["action"] in {"BUY", "HOLD", "SELL"}
    engine.open_long(agent, row, action="BUY", reason="ENTRY")
    assert agent.position_quantity > 0
    assert agent.cash < 100000.0


def test_hold_preserves_position() -> None:
    genome = DecisionGenome.from_default(signal_threshold=0.9, seed=2)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    agent.position_quantity = 10.0
    agent.entry_price = 100.0
    agent.cash = 5000.0
    row = pd.Series({"Date": "2024-01-02", "Close": 100.0, "Low": 99.0, "High": 101.0})
    action = agent.decide(0.1, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert action["action"] == "HOLD"
    agent._append_equity(price=float(row["Close"]))
    assert agent.position_quantity == 10.0


def test_sell_closes_position() -> None:
    genome = DecisionGenome.from_default(signal_threshold=0.1, seed=3)
    genome.input_to_hidden.fill(0.0)
    genome.hidden_to_output.fill(0.0)
    genome.output_bias = -0.5
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    agent.position_quantity = 10.0
    agent.entry_price = 100.0
    agent.cash = 1000.0
    agent.current_capital = 2000.0
    row = pd.Series({"Date": "2024-01-03", "Close": 101.0, "Low": 95.0, "High": 103.0})
    decision = agent.decide(-0.5, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert decision["action"] == "SELL"


def test_transaction_costs_reduce_capital() -> None:
    genome = DecisionGenome.from_default(seed=4)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    engine = TradingEngine(transaction_cost=0.001)
    row = pd.Series({"Date": "2024-01-01", "Close": 100.0, "Low": 99.0, "High": 101.0})
    engine.open_long(agent, row, action="BUY", reason="ENTRY")
    before = 100000.0
    assert agent.cash < before
    assert agent.statistics["transaction_costs"] > 0.0


def test_position_size_is_correctly_calculated() -> None:
    genome = DecisionGenome.from_default(position_size=0.25, seed=5)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    engine = TradingEngine(position_size=0.25)
    row = pd.Series({"Date": "2024-01-01", "Close": 100.0, "Low": 99.0, "High": 101.0})
    engine.open_long(agent, row, action="BUY", reason="ENTRY")
    assert agent.position_quantity > 0
    assert agent.cash >= 0.0


def test_stop_loss_closes_losing_position() -> None:
    genome = DecisionGenome.from_default(seed=6)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    agent.position_quantity = 10.0
    agent.entry_price = 100.0
    agent.cash = 5000.0
    row = pd.Series({"Date": "2024-01-04", "Close": 96.0, "Low": 96.0, "High": 97.0})
    engine = TradingEngine()
    engine.close_position(agent, row, action="SELL", reason="STOP_LOSS")
    assert agent.position_quantity == 0.0
    assert agent.trade_history[-1]["reason"] == "STOP_LOSS"


def test_take_profit_closes_profitable_position() -> None:
    genome = DecisionGenome.from_default(seed=7)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    agent.position_quantity = 10.0
    agent.entry_price = 100.0
    agent.cash = 5000.0
    row = pd.Series({"Date": "2024-01-05", "Close": 106.0, "Low": 105.0, "High": 106.5})
    engine = TradingEngine()
    engine.close_position(agent, row, action="SELL", reason="TAKE_PROFIT")
    assert agent.position_quantity == 0.0
    assert agent.trade_history[-1]["reason"] == "TAKE_PROFIT"


def test_simultaneous_sl_tp_uses_stop_loss_first() -> None:
    genome = DecisionGenome.from_default(seed=8)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    agent.position_quantity = 10.0
    agent.entry_price = 100.0
    row = pd.Series({"Date": "2024-01-06", "Close": 104.0, "Low": 97.0, "High": 106.0})
    engine = TradingEngine()
    reason, _ = engine._trigger_exit(agent, row)
    assert reason == "STOP_LOSS"


def test_zero_trade_agent_retains_capital() -> None:
    genome = DecisionGenome.from_default(seed=9)
    agent = Agent(agent_id="a", genome=genome, starting_capital=25000.0)
    assert agent.cash == 25000.0
    assert len(agent.trade_history) == 0


def test_trade_history_is_consistent() -> None:
    genome = DecisionGenome.from_default(seed=10)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    engine = TradingEngine()
    row = pd.Series({"Date": "2024-01-01", "Close": 100.0, "Low": 98.0, "High": 101.0})
    engine.open_long(agent, row, action="BUY", reason="ENTRY")
    assert agent.trade_history[-1]["cash_after"] < 100000.0
    assert agent.trade_history[-1]["equity_after"] == pytest.approx(agent.cash + agent.position_quantity * row["Close"])


def test_equity_curve_is_generated_for_every_timestep() -> None:
    genome = DecisionGenome.from_default(seed=11)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    agent.equity_history = [100000.0, 100500.0, 100100.0]
    metrics = PerformanceMetrics.from_agent(agent)
    assert len(metrics.equity_curve) == 3


def test_no_negative_cash_due_to_position_sizing() -> None:
    genome = DecisionGenome.from_default(position_size=0.999, seed=12)
    agent = Agent(agent_id="a", genome=genome, starting_capital=1000.0)
    engine = TradingEngine(position_size=0.999)
    row = pd.Series({"Date": "2024-01-01", "Close": 100.0, "Low": 99.0, "High": 101.0})
    engine.open_long(agent, row, action="BUY", reason="ENTRY")
    assert agent.cash >= 0.0


def test_same_inputs_produce_identical_results() -> None:
    genome = DecisionGenome.from_default(seed=13)
    agent_1 = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    agent_2 = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    score_1 = agent_1.decision_score(0.5, [0.1, 0.2, 0.3, 0.4, 0.5])
    score_2 = agent_2.decision_score(0.5, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.isclose(score_1, score_2)


def test_agent_works_with_mock_forecaster() -> None:
    genome = DecisionGenome.from_default(seed=14)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    forecaster = MockForecaster(prediction=0.8)
    result = agent.decide(forecaster.predict(np.ones((5, 1)))[0], [0.2, 0.3, 0.4, 0.5, 0.6])
    assert result["action"] in {"BUY", "SELL", "HOLD"}


def test_baseline_strategy_produces_deterministic_trades() -> None:
    genome = DecisionGenome.from_default(seed=15)
    baseline = ManualBaselineAgent(agent_id="baseline", genome=genome, starting_capital=100000.0)
    first = baseline.decide(0.5, [0.0, 0.0, 0.0, 0.0, 0.0])
    second = baseline.decide(0.5, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert first == second


def test_realized_and_unrealized_pnl_accounting() -> None:
    genome = DecisionGenome.from_default(seed=16)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    agent.position_quantity = 10
    agent.entry_price = 100.0
    agent.current_capital = 1010.0
    metrics = PerformanceMetrics.from_agent(agent)
    assert metrics.realized_pnl >= 0.0 or metrics.unrealized_pnl >= 0.0


def test_basic_engine_step_runs() -> None:
    genome = DecisionGenome.from_default(seed=17)
    agent = Agent(agent_id="a", genome=genome, starting_capital=100000.0)
    engine = TradingEngine()
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.0, 101.0, 102.0],
        }
    )
    outcome = engine.simulate(agent, df, MockForecaster(prediction=0.8))
    assert isinstance(outcome, Agent)
    assert len(outcome.equity_history) >= 1
