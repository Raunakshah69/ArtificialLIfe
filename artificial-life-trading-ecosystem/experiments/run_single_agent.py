"""Run a single deterministic trading organism on the evolution-development segment."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from artificial_life_trading_ecosystem.agents.agent import Agent
from artificial_life_trading_ecosystem.agents.baseline import ManualBaselineAgent
from artificial_life_trading_ecosystem.config import settings
from artificial_life_trading_ecosystem.data import build_feature_set, chronological_split, load_raw_market_data, validate_market_data
from artificial_life_trading_ecosystem.evaluation.metrics import PerformanceMetrics
from artificial_life_trading_ecosystem.models.forecasting import SimpleRNN
from artificial_life_trading_ecosystem.models.genomes.decision_genome import DecisionGenome
from artificial_life_trading_ecosystem.trading.engine import TradingEngine


def _find_saved_forecaster() -> Path:
    forecast_root = REPO_ROOT / "results" / "forecasts"
    candidates = [p for p in forecast_root.glob("*/model.keras") if p.exists()]
    if candidates:
        return candidates[0].parent
    raise FileNotFoundError(f"No saved forecaster was found under {forecast_root}.")


def _ensure_prepared_artifacts() -> None:
    data_path = REPO_ROOT / "data" / "processed" / "market_data.csv"
    if not data_path.exists():
        raw_df = load_raw_market_data()
        cleaned = validate_market_data(raw_df)
        prepared = build_feature_set(cleaned)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(data_path, index=False)
    if not _find_saved_forecaster().exists():
        runpy.run_path(str(REPO_ROOT / "experiments" / "train_forecasters.py"), run_name="__main__")


def run_demo() -> None:
    _ensure_prepared_artifacts()
    data_path = REPO_ROOT / "data" / "processed" / "market_data.csv"
    model_dir = _find_saved_forecaster()

    market = pd.read_csv(data_path)
    if len(market) < 2:
        raise ValueError("Prepared market data must contain at least two rows.")

    splits = chronological_split(
        market,
        train_proportion=settings.train_proportion,
        validation_proportion=settings.validation_proportion,
        evolution_proportion=settings.evolution_proportion,
        test_proportion=settings.test_proportion,
    )
    dev_market = splits["evolution"].copy()

    if dev_market.empty:
        raise ValueError("Development evolution segment is empty; adjust split proportions.")

    forecaster = SimpleRNN()
    forecaster.load(model_dir)

    feature_columns = [col for col in market.columns if col != "Date"]
    genome = DecisionGenome.from_default(seed=42)
    agent = Agent(agent_id="agent_demo", generation=0, genome=genome, starting_capital=settings.starting_capital)
    baseline = ManualBaselineAgent(agent_id="baseline_demo", generation=0, genome=genome, starting_capital=settings.starting_capital)

    engine = TradingEngine()
    engine.simulate(agent, dev_market, forecaster, feature_columns=feature_columns)
    engine.simulate(baseline, dev_market, forecaster, feature_columns=feature_columns)

    agent_metrics = PerformanceMetrics.from_agent(agent)
    baseline_metrics = PerformanceMetrics.from_agent(baseline)

    output_dir = REPO_ROOT / "results" / "single_agent_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "agent_trade_history.json").write_text(json.dumps(agent.trade_history, indent=2), encoding="utf-8")
    (output_dir / "agent_equity_curve.json").write_text(json.dumps(agent.equity_history, indent=2), encoding="utf-8")
    (output_dir / "agent_metrics.json").write_text(json.dumps(agent_metrics.__dict__, indent=2), encoding="utf-8")
    (output_dir / "baseline_metrics.json").write_text(json.dumps(baseline_metrics.__dict__, indent=2), encoding="utf-8")

    print({
        "segment": "evolution",
        "rows": len(dev_market),
        "starting_capital": round(settings.starting_capital, 2),
        "agent_final_capital": round(agent.current_capital, 2),
        "baseline_final_capital": round(baseline.current_capital, 2),
        "agent_total_return": round(((agent.current_capital - settings.starting_capital) / settings.starting_capital) if settings.starting_capital else 0.0, 6),
        "baseline_total_return": round(((baseline.current_capital - settings.starting_capital) / settings.starting_capital) if settings.starting_capital else 0.0, 6),
        "agent_trades": len(agent.trade_history),
        "baseline_trades": len(baseline.trade_history),
        "agent_tx_cost": round(agent.statistics.get("transaction_costs", 0.0), 6),
        "max_drawdown": round(agent_metrics.maximum_drawdown, 6),
        "win_rate": round(agent_metrics.win_rate, 6),
    })


if __name__ == "__main__":
    run_demo()
