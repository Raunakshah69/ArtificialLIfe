"""FastAPI endpoints for persisted, development-only interactive evolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from artificial_life_trading_ecosystem.config import ROOT_DIR, settings
from artificial_life_trading_ecosystem.data.pipeline import build_feature_set, load_raw_market_data, validate_market_data
from artificial_life_trading_ecosystem.evaluation.experiment import _ensure_forecaster
from artificial_life_trading_ecosystem.runtime.simulation import InteractiveSimulation


class RunGenerationsRequest(BaseModel):
    count: int = Field(ge=1, le=100)


class SelectExperimentRequest(BaseModel):
    model_name: Literal["SimpleRNN", "LSTM", "GRU"]


def available_experiments() -> list[dict[str, Any]]:
    models = []
    for model_name in ("SimpleRNN", "LSTM", "GRU"):
        artifact_path = ROOT_DIR / "results" / "forecasts" / model_name.lower() / "model.keras"
        if artifact_path.exists():
            models.append(
                {
                    "experiment_id": f"local-evolution-development:{model_name}",
                    "model_name": model_name,
                    "label": f"Evolution-development / {model_name}",
                    "data_split": "evolution-development",
                    "final_test_access": False,
                }
            )
    return models


def build_development_simulation(model_name: str = "SimpleRNN", *, state_path: str | Path | None = None) -> InteractiveSimulation:
    raw_market = validate_market_data(load_raw_market_data())
    proportions = (
        settings.train_proportion,
        settings.validation_proportion,
        settings.evolution_proportion,
        settings.test_proportion,
    )
    if not np.isclose(sum(proportions), 1.0):
        raise ValueError("Chronological split proportions must sum to 1.0.")
    total_rows = len(raw_market)
    validation_end = int(total_rows * (settings.train_proportion + settings.validation_proportion))
    evolution_end = int(
        total_rows
        * (settings.train_proportion + settings.validation_proportion + settings.evolution_proportion)
    )
    development_source = raw_market.iloc[:evolution_end].copy()
    features = build_feature_set(development_source)
    features.insert(0, "Date", development_source["Date"].to_numpy())
    history_prefix = features.iloc[:validation_end].copy().reset_index(drop=True)
    development_market = features.iloc[validation_end:evolution_end].copy().reset_index(drop=True)
    if history_prefix.empty or development_market.empty:
        raise ValueError("Train, validation, and evolution-development segments must be non-empty.")
    forecaster = _ensure_forecaster(model_name, allow_heuristic=False)
    feature_columns = [name for name in forecaster.feature_names if name in development_market.columns]
    return InteractiveSimulation(
        development_market=development_market,
        history_prefix=history_prefix,
        forecaster=forecaster,
        state_path=state_path or ROOT_DIR / "results" / "interactive" / model_name.lower() / "simulation-state.json",
        population_size=min(settings.population_size, 10),
        starting_capital=settings.starting_capital,
        survival_threshold=settings.survival_threshold,
        mutation_rate=settings.mutation_rate,
        mutation_sigma=settings.mutation_sigma,
        seed=settings.seed,
        feature_columns=feature_columns,
        experiment_id=f"local-evolution-development:{model_name}",
    )


def create_app(
    simulation: InteractiveSimulation | None = None,
    *,
    simulation_factory: Callable[..., InteractiveSimulation] = build_development_simulation,
) -> FastAPI:
    app = FastAPI(title="Artificial Life Trading Ecosystem API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.simulation = simulation
    app.state.selected_model = "SimpleRNN"

    def get_simulation() -> InteractiveSimulation:
        if app.state.simulation is None:
            try:
                app.state.simulation = simulation_factory(app.state.selected_model)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Development simulation could not start: {exc}") from exc
        return app.state.simulation

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "data_split": "evolution-development", "final_test_access": "false"}

    @app.get("/api/simulation")
    def get_state() -> dict[str, Any]:
        return get_simulation().get_state()

    @app.get("/api/experiments")
    def get_experiments() -> dict[str, Any]:
        return {"experiments": available_experiments(), "final_test_access": False}

    @app.post("/api/simulation/select-experiment")
    def select_experiment(request: SelectExperimentRequest) -> dict[str, Any]:
        available = {entry["model_name"] for entry in available_experiments()}
        if request.model_name not in available:
            raise HTTPException(status_code=404, detail="The trained forecasting experiment is not available.")
        try:
            app.state.simulation = simulation_factory(request.model_name)
            app.state.selected_model = request.model_name
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Selected development experiment could not start: {exc}") from exc
        return app.state.simulation.get_state()

    @app.post("/api/simulation/next-day")
    def next_day() -> dict[str, Any]:
        try:
            return get_simulation().next_day()
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/simulation/run-generation")
    def run_generation() -> dict[str, Any]:
        try:
            return get_simulation().run_current_generation()
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/simulation/run-generations")
    def run_generations(request: RunGenerationsRequest) -> dict[str, Any]:
        try:
            return get_simulation().run_generations(request.count)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/simulation/reset")
    def reset() -> dict[str, Any]:
        return get_simulation().reset()

    @app.get("/api/generations/{generation}")
    def get_generation(generation: int) -> dict[str, Any]:
        state = get_simulation().get_state()
        selected = next((entry for entry in state["generations"] if entry["generation"] == generation), None)
        if selected is None:
            raise HTTPException(status_code=404, detail="Generation is not present in interactive development state.")
        return selected

    @app.get("/api/replay/{generation}")
    def replay_generation(generation: int) -> dict[str, Any]:
        state = get_simulation().get_state()
        selected = next((entry for entry in state["generations"] if entry["generation"] == generation), None)
        if selected is None:
            raise HTTPException(status_code=404, detail="No saved daily events exist for this generation.")
        return {"generation": generation, "events": selected.get("daily_events", [])}

    @app.get("/api/backtest/{generation}")
    def backtest_generation(generation: int, agent_id: str | None = None) -> dict[str, Any]:
        state = get_simulation().get_state()
        selected = next((entry for entry in state["generations"] if entry["generation"] == generation), None)
        if selected is None or selected.get("status") != "EVALUATED":
            raise HTTPException(status_code=409, detail="Backtest curves are available only for evaluated development generations.")
        agents = selected.get("agents", [])
        evolved = next((agent for agent in agents if agent["agent_id"] == agent_id), None) if agent_id else None
        best_evolved = max(agents, key=lambda agent: float(agent.get("ending_capital", agent.get("cash", 0.0))), default=None)
        if evolved is None:
            evolved = best_evolved
        selected_agent = {**evolved, "equity_curve": list(evolved.get("equity_curve", evolved.get("equity_history", [])))} if evolved else None
        best_lineage = {**best_evolved, "equity_curve": list(best_evolved.get("equity_curve", best_evolved.get("equity_history", [])))} if best_evolved else None
        return {
            "generation": generation,
            "data_split": "evolution-development",
            "selected_agent": selected_agent,
            "best_lineage": best_lineage,
            "manual_baseline": selected.get("strategy_curves", {}).get("manual_baseline", []),
            "buy_and_hold": selected.get("strategy_curves", {}).get("buy_and_hold", []),
            "final_test_enabled": False,
        }

    return app


app = create_app()