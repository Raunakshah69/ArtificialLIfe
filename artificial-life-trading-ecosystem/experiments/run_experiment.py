"""CLI for reproducible M5 experimental evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from artificial_life_trading_ecosystem.evaluation.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the artificial-life trading ecosystem experimental evaluation.")
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["SimpleRNN", "LSTM", "GRU"], default="SimpleRNN")
    parser.add_argument("--target-mode", choices=["regression", "classification"], default="regression")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()

    result = run_experiment(
        population_size=args.population_size,
        generations=args.generations,
        seed=args.seed,
        model=args.model,
        target_mode=args.target_mode,
        experiment_name=args.experiment_name,
        skip_training=args.skip_training,
        development_only=args.development_only,
        final_test=args.final_test,
    )
    print(json.dumps({
        "experiment_id": result.get("experiment_id"),
        "experiment_root": result.get("experiment_root"),
        "development_dir": result.get("development_dir"),
        "final_test_dir": result.get("final_test_dir"),
        "generation_count": len(result.get("generation_history", [])),
    }, indent=2))


if __name__ == "__main__":
    main()
