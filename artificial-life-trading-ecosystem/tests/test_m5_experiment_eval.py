from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from artificial_life_trading_ecosystem.evaluation.experiment import (
    buy_and_hold_benchmark,
    compute_drawdown,
    compute_generation_metrics,
    compute_lineage_extinction,
    compute_sharpe_ratio,
    compute_strategy_comparison_table,
    make_experiment_id,
    run_development_experiment,
)
from artificial_life_trading_ecosystem.evolution.population import Population, PopulationConfig


@pytest.fixture
def synthetic_epoch() -> pd.DataFrame:
    rows = []
    price = 100.0
    for idx in range(25):
        price = price * (1.0 + ((idx % 5) - 2) * 0.01)
        rows.append(
            {
                "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx),
                "Open": round(price, 4),
                "High": round(price * 1.02, 4),
                "Low": round(price * 0.98, 4),
                "Close": round(price, 4),
                "Volume": 1000 + idx * 25,
            }
        )
    return pd.DataFrame(rows)


def test_generation_metrics_calculation(synthetic_epoch: pd.DataFrame) -> None:
    population = Population(PopulationConfig(population_size=4, seed=11))
    for agent in population.agents:
        agent.statistics["ending_capital"] = 100000.0 + agent.agent_id.__hash__() % 1000
        agent.alive = True
        agent.trade_history = [{"action": "BUY"}]
    metrics = compute_generation_metrics(population, 0, synthetic_epoch, starting_capital=100000.0)
    assert metrics["generation"] == 0
    assert metrics["population_size"] == 4
    assert metrics["mean_capital"] >= 0.0
    assert "elite_agent_id" in metrics


def test_lineage_depth_calculation() -> None:
    records = {
        "A": type("R", (), {"parent_ids": [], "generation": 0})(),
        "B": type("R", (), {"parent_ids": ["A"], "generation": 1})(),
        "C": type("R", (), {"parent_ids": ["A"], "generation": 1})(),
        "D": type("R", (), {"parent_ids": ["B", "C"], "generation": 2})(),
    }
    depths = compute_lineage_extinction(records)
    assert depths["total_lineage_records"] == 4
    assert depths["maximum_lineage_depth"] >= 2


def test_lineage_extinction_calculation() -> None:
    records = {
        "root": type("R", (), {"parent_ids": [], "generation": 0})(),
        "child": type("R", (), {"parent_ids": ["root"], "generation": 1})(),
    }
    summary = compute_lineage_extinction(records, living_agents=[])
    assert summary["unique_root_lineages"] == 1
    assert summary["extinct_lineages"] == 1
    assert summary["extinction_rate"] == 1.0


def test_strategy_comparison_table() -> None:
    table = compute_strategy_comparison_table(
        evolved_metrics={"strategy": "evolved", "starting_capital": 100000.0, "ending_capital": 120000.0, "cumulative_return": 0.2, "trade_count": 10, "win_rate": 0.5, "maximum_drawdown": 0.1, "transaction_costs": 25.0, "sharpe_ratio": 1.2, "sharpe_valid": True, "generation_discovered": 3, "lineage_depth": 2, "lineage_root": "A1", "number_of_descendants": 5},
        manual_metrics={"strategy": "manual", "starting_capital": 100000.0, "ending_capital": 110000.0, "cumulative_return": 0.1, "trade_count": 20, "win_rate": 0.4, "maximum_drawdown": 0.08, "transaction_costs": 35.0, "sharpe_ratio": 0.8, "sharpe_valid": True},
        buy_hold_metrics={"strategy": "buy_hold", "starting_capital": 100000.0, "ending_capital": 115000.0, "cumulative_return": 0.15, "trade_count": 1, "win_rate": 1.0, "maximum_drawdown": 0.12, "transaction_costs": 0.0, "sharpe_ratio": None, "sharpe_valid": False},
    )
    assert list(table["strategy"]) == ["evolved", "manual", "buy_hold"]
    assert len(table) == 3


def test_buy_and_hold_benchmark() -> None:
    market = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "Open": [100, 101, 102, 103, 104],
        "High": [101, 102, 103, 104, 105],
        "Low": [99, 100, 101, 102, 103],
        "Close": [100, 101, 102, 103, 104],
        "Volume": [1000, 1000, 1000, 1000, 1000],
    })
    result = buy_and_hold_benchmark(market, starting_capital=100000.0, transaction_cost=0.001)
    assert result["trade_count"] == 1
    assert result["starting_capital"] == 100000.0
    assert result["ending_capital"] > 0.0


def test_drawdown_calculation() -> None:
    drawdown = compute_drawdown([100, 110, 105, 90, 95])
    assert drawdown > 0.0
    assert drawdown <= 1.0


def test_sharpe_edge_cases() -> None:
    assert compute_sharpe_ratio([100.0, 100.0, 100.0]) is None
    assert compute_sharpe_ratio([100.0, 105.0, 110.0]) is not None


def test_make_experiment_id_is_unique_and_stable() -> None:
    first = make_experiment_id(seed=42)
    second = make_experiment_id(seed=42)
    assert first != second
    assert first.startswith("exp-")


def test_experiment_artifact_creation(tmp_path: Path) -> None:
    result = run_development_experiment(
        population_size=4,
        generations=2,
        seed=11,
        model="SimpleRNN",
        target_mode="regression",
        development_only=True,
        results_dir=tmp_path,
    )
    experiment_root = Path(result["experiment_root"])
    assert experiment_root.exists()
    assert (experiment_root / "config.yaml").exists()
    assert (experiment_root / "generation_metrics.csv").exists()
    assert (experiment_root / "lineage_metrics.json").exists()
    assert (experiment_root / "strategy_comparison.csv").exists()
    assert (experiment_root / "final_test").exists() is False


def test_development_only_mode_has_no_final_test_dir(tmp_path: Path) -> None:
    result = run_development_experiment(
        population_size=4,
        generations=2,
        seed=12,
        model="SimpleRNN",
        target_mode="regression",
        development_only=True,
        results_dir=tmp_path,
    )
    assert result["final_test_dir"] is None
    assert not (Path(result["experiment_root"]) / "final_test").exists()


def test_required_configuration_metadata_is_recorded(tmp_path: Path) -> None:
    result = run_development_experiment(
        population_size=4,
        generations=2,
        seed=13,
        model="SimpleRNN",
        target_mode="regression",
        development_only=True,
        results_dir=tmp_path,
    )
    config = json.loads((Path(result["experiment_root"]) / "manifest.json").read_text(encoding="utf-8"))
    assert config["seed"] == 13
    assert config["ticker"]
    assert config["population_size"] == 4
    assert config["generations"] == 2
