"""Run the M4 evolution pipeline on the chronological development segment only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from artificial_life_trading_ecosystem.config import settings
from artificial_life_trading_ecosystem.data import build_feature_set, chronological_split, load_raw_market_data, validate_market_data
from artificial_life_trading_ecosystem.evolution.population import Population, PopulationConfig
from artificial_life_trading_ecosystem.models.forecasting import SimpleRNN

OUTPUT_DIR = REPO_ROOT / "results" / "population"


def _find_saved_forecaster() -> Path:
    forecast_root = REPO_ROOT / "results" / "forecasts"
    candidates = sorted(forecast_root.glob("*/model.keras"), key=lambda path: str(path))
    if not candidates:
        raise FileNotFoundError(f"No saved forecaster was found under {forecast_root}.")
    preferred = [path for path in candidates if "simplernn" in str(path).lower()]
    return (preferred[0] if preferred else candidates[0]).parent


def _ensure_prepared_market() -> Path:
    data_path = REPO_ROOT / "data" / "processed" / "market_data.csv"
    if not data_path.exists():
        raw_df = load_raw_market_data()
        cleaned = validate_market_data(raw_df)
        prepared = build_feature_set(cleaned)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(data_path, index=False)
    return data_path


def _load_forecaster() -> SimpleRNN:
    model_dir = _find_saved_forecaster()
    forecaster = SimpleRNN()
    forecaster.load(model_dir)
    if hasattr(forecaster, "model") and forecaster.model is not None:
        forecaster.model.trainable = False
        for layer in forecaster.model.layers:
            layer.trainable = False
    return forecaster


def _evolution_epochs(market: pd.DataFrame, *, generation_count: int) -> list[pd.DataFrame]:
    splits = chronological_split(
        market,
        train_proportion=settings.train_proportion,
        validation_proportion=settings.validation_proportion,
        evolution_proportion=settings.evolution_proportion,
        test_proportion=settings.test_proportion,
    )
    development = splits["evolution"].copy()
    if development.empty:
        raise ValueError("The evolution-development segment is empty. Adjust the data split proportions.")
    if generation_count <= 0:
        raise ValueError("The number of generations must be positive.")
    epoch_count = min(int(generation_count), max(1, len(development)))
    indices = np.array_split(np.arange(len(development)), epoch_count)
    return [development.iloc[idx].reset_index(drop=True).copy() for idx in indices if len(idx) > 0]


def _summarize_generation(population: Population, generation: int) -> dict[str, float | int | str]:
    stats = population.compute_generation_statistics(population.agents, generation)
    return {
        "generation": int(generation),
        "population_size": int(len(population.agents)),
        "alive_count": int(stats.get("alive_count", 0)),
        "dead_count": int(stats.get("dead_count", 0)),
        "survival_rate": float(stats.get("survival_rate", 0.0)),
        "zero_trade_count": int(stats.get("zero_trade_count", 0)),
        "mean_capital": float(stats.get("mean_ending_capital", stats.get("mean_capital", 0.0))),
        "median_capital": float(stats.get("median_ending_capital", stats.get("median_capital", 0.0))),
        "best_capital": float(stats.get("best_ending_capital", stats.get("best_capital", 0.0))),
        "worst_capital": float(stats.get("worst_ending_capital", stats.get("worst_capital", 0.0))),
        "genetic_diversity": float(stats.get("genetic_diversity", 0.0)),
        "immigrant_count": int(stats.get("immigrant_count", 0)),
        "sexual_children": int(sum(1 for record in population.lineage.values() if int(getattr(record, "generation", 0)) == generation and getattr(record, "reproduction_method", "") == "sexual")),
        "asexual_children": int(sum(1 for record in population.lineage.values() if int(getattr(record, "generation", 0)) == generation and getattr(record, "reproduction_method", "") == "asexual")),
        "mutation_count": int(sum(1 for record in population.lineage.values() if int(getattr(record, "generation", 0)) == generation and bool(getattr(record, "mutation_applied", False)))),
        "elite_agent_id": str(stats.get("elite_agent_id", "")),
    }


def _advance_population(population: Population, epoch: pd.DataFrame, *, feature_columns: list[str]) -> None:
    population.generation = int(population.generation)
    population.evaluate_generation(epoch, feature_columns=feature_columns)
    population.mark_dead_agents()
    survivors = population.surviving_agents()

    if not survivors:
        population.agents = population.zero_survivor_fallback(target_count=population.population_size)
        population.generation = int(population.generation) + 1
        return

    elite_count = min(population.elite_count, len(survivors))
    next_agents = population.preserve_elite(survivors[:elite_count]) if elite_count else []
    eligible_parents = population.eligible_parents() or list(population.agents)

    while len(next_agents) < population.population_size:
        parents = population.capital_weighted_selection(eligible_parents)
        if len(parents) >= 2:
            parent_a, parent_b = parents[0], parents[1]
            child = population.create_child(parent_a=parent_a, parent_b=parent_b, generation=population.generation + 1, method="sexual")
            next_agents.append(child)
            continue
        if parents:
            child = population.create_child(parent_a=parents[0], parent_b=None, generation=population.generation + 1, method="asexual")
            next_agents.append(child)
            continue
        immigrants = population.make_immigrants(count=max(1, population.population_size - len(next_agents)))
        next_agents.extend(immigrants)
        break

    if len(next_agents) < population.population_size:
        immigrants = population.make_immigrants(count=population.population_size - len(next_agents))
        next_agents.extend(immigrants)

    population.agents = next_agents[: population.population_size]
    population.generation = int(population.generation) + 1


def _save_population_outputs(population: Population, generation_history: list[dict[str, float | int | str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_df = pd.DataFrame(generation_history)
    stats_df = stats_df[
        [
            "generation",
            "population_size",
            "alive_count",
            "dead_count",
            "survival_rate",
            "zero_trade_count",
            "mean_capital",
            "median_capital",
            "best_capital",
            "worst_capital",
            "genetic_diversity",
            "immigrant_count",
            "sexual_children",
            "asexual_children",
            "mutation_count",
            "elite_agent_id",
        ]
    ]
    stats_df.to_csv(OUTPUT_DIR / "generation_stats.csv", index=False)

    snapshot = {
        "generation": int(population.generation),
        "population_size": int(len(population.agents)),
        "starting_capital": float(population.starting_capital),
        "seed": int(population.seed),
        "agents": [
            {
                "agent_id": agent.agent_id,
                "generation": int(agent.generation),
                "alive": bool(agent.alive),
                "starting_capital": float(agent.starting_capital),
                "cash": float(agent.cash),
                "current_capital": float(agent.current_capital),
                "position_quantity": float(agent.position_quantity),
                "parent_a": agent.parent_a,
                "parent_b": agent.parent_b,
                "immigrant": bool(agent.statistics.get("immigrant", False)),
                "elite": bool(agent.statistics.get("elite", False)),
                "trade_count": int(agent.statistics.get("trade_count", len(agent.trade_history))),
                "genome": {
                    "signal_threshold": float(agent.genome.signal_threshold),
                    "position_size": float(agent.genome.position_size),
                    "stop_loss": float(agent.genome.stop_loss),
                    "take_profit": float(agent.genome.take_profit),
                    "transaction_cost": float(agent.genome.transaction_cost),
                },
            }
            for agent in population.agents
        ],
    }
    (OUTPUT_DIR / "population_state.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    lineage_rows = []
    for record in population.lineage.values():
        lineage_rows.append(
            {
                "agent_id": record.agent_id,
                "parent_a": record.parent_a,
                "parent_b": record.parent_b,
                "parent_ids": ";".join(record.parent_ids),
                "generation": int(record.generation),
                "reproduction_method": record.reproduction_method,
                "mutation_applied": bool(record.mutation_applied),
                "immigrant": bool(record.immigrant),
                "metadata": json.dumps(record.metadata, sort_keys=True),
            }
        )
    pd.DataFrame(lineage_rows).to_csv(OUTPUT_DIR / "lineage.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M4 population evolution loop on the approved development segment.")
    parser.add_argument("--population-size", type=int, default=int(settings.population_size))
    parser.add_argument("--generations", type=int, default=int(settings.generations))
    parser.add_argument("--seed", type=int, default=int(settings.seed))
    args = parser.parse_args()

    data_path = _ensure_prepared_market()
    market = pd.read_csv(data_path)
    forecaster = _load_forecaster()

    if hasattr(forecaster, "model") and forecaster.model is not None:
        forecaster.model.trainable = False
        for layer in forecaster.model.layers:
            layer.trainable = False

    development_epochs = _evolution_epochs(market, generation_count=args.generations)
    feature_columns = [col for col in market.columns if col != "Date"]

    population_cfg = PopulationConfig(
        population_size=args.population_size,
        generation=0,
        generations=max(1, args.generations),
        starting_capital=settings.starting_capital,
        survival_threshold=settings.survival_threshold,
        mutation_rate=settings.mutation_rate,
        mutation_sigma=settings.mutation_sigma,
        immigrant_fraction=settings.immigrant_fraction,
        elite_count=settings.elite_count,
        seed=args.seed,
    )
    population = Population(population_cfg, forecaster=forecaster, seed=args.seed)

    generation_history: list[dict[str, float | int | str]] = []
    print({
        "dataset": settings.project_name,
        "instrument": settings.ticker,
        "development_epoch_count": len(development_epochs),
        "population_size": args.population_size,
        "generations": min(args.generations, len(development_epochs)),
        "starting_capital": settings.starting_capital,
    })

    for generation_index, epoch in enumerate(development_epochs[: max(1, min(args.generations, len(development_epochs)) )]):
        population.generation = generation_index
        _advance_population(population, epoch, feature_columns=feature_columns)
        summary = _summarize_generation(population, generation_index)
        generation_history.append(summary)

        print({
            "generation": summary["generation"],
            "alive_dead": f"{summary['alive_count']}/{summary['dead_count']}",
            "mean_capital": round(float(summary["mean_capital"]), 2),
            "median_capital": round(float(summary["median_capital"]), 2),
            "best_capital": round(float(summary["best_capital"]), 2),
            "survival_rate": round(float(summary["survival_rate"]), 4),
            "immigrants": int(summary["immigrant_count"]),
            "mutations": int(summary["mutation_count"]),
            "elite": summary["elite_agent_id"],
        })

    _save_population_outputs(population, generation_history)

    final_stats = generation_history[-1] if generation_history else _summarize_generation(population, population.generation)
    print({
        "final_generation": int(population.generation),
        "final_population_size": int(len(population.agents)),
        "best_final_generation_capital": round(float(final_stats["best_capital"]), 2),
        "lineage_records": len(population.lineage),
        "number_of_immigrants": sum(1 for record in population.lineage.values() if record.immigrant),
        "number_of_mutations": sum(1 for record in population.lineage.values() if record.mutation_applied),
    })


if __name__ == "__main__":
    main()
