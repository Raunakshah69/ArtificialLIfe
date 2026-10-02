from __future__ import annotations

import pandas as pd
import pytest

from artificial_life_trading_ecosystem.agents.agent import Agent
from artificial_life_trading_ecosystem.evaluation.experiment import _ensure_forecaster, run_development_experiment
from artificial_life_trading_ecosystem.models.genomes.decision_genome import DecisionGenome
from artificial_life_trading_ecosystem.trading.engine import TradingEngine


class DummyForecaster:
    def __init__(self, mode: str = "trained") -> None:
        self.model_type = "Dummy"
        self.config = type("Config", (), {"sequence_length": 5, "epochs": 1})()
        self.feature_names = ["Open", "High", "Low", "Close", "Volume"]
        self.validation_metrics_ = {"mae": 0.0, "rmse": 0.0, "loss": 0.0}
        self.forecaster_mode = mode

    def predict(self, window):
        return [0.01]


def test_run_experiment_rejects_silent_heuristic_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load(_model_name: str):
        raise FileNotFoundError("missing trained model")

    monkeypatch.setattr("artificial_life_trading_ecosystem.evaluation.experiment._load_saved_forecaster", fail_load)
    with pytest.raises(RuntimeError, match="trained forecaster"):
        _ensure_forecaster("SimpleRNN", allow_heuristic=False)


def test_heuristic_mode_is_explicitly_reported() -> None:
    forecaster = DummyForecaster(mode="heuristic")
    assert getattr(forecaster, "forecaster_mode", "heuristic") == "heuristic"


def test_generation_chunks_are_non_overlapping_and_10_day_default() -> None:
    market = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=25, freq="D"),
            "Open": range(25),
            "High": [x + 1 for x in range(25)],
            "Low": [max(0, x - 1) for x in range(25)],
            "Close": range(25),
            "Volume": [1000 + x for x in range(25)],
        }
    )
    engine = TradingEngine()
    chunks = engine.chunk_market_by_generation(market)
    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [10, 10, 5]
    assert chunks[0].iloc[-1]["Date"] < chunks[1].iloc[0]["Date"]


def test_daily_simulation_closes_open_positions_by_end_of_session() -> None:
    market = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000] * 5,
        }
    )
    agent = Agent(
        agent_id="A001",
        genome=DecisionGenome.from_default(seed=7),
        starting_capital=10000.0,
        cash=10000.0,
        current_capital=10000.0,
    )
    agent.genome.input_to_hidden.fill(0.0)
    agent.genome.hidden_to_output.fill(0.0)
    agent.genome.output_bias = 0.5
    engine = TradingEngine()
    result = engine.simulate_daily(agent, market, DummyForecaster(mode="trained"))
    assert result is agent
    assert agent.position_quantity == 0.0
    assert len(agent.trade_history) >= 1
