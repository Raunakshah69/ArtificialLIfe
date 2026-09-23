# Evolution runner

The M4 population runner uses the approved M1 prepared market data, the saved M2 forecasting backbone, and the M3 agent/trading contracts. It evaluates the evolution-development segment only and does not touch the final held-out test period.

## Full run

```bash
python experiments/run_population.py
```

## Lightweight smoke test

```bash
python experiments/run_population.py --population-size 10 --generations 3
```

The script loads the saved shared forecaster once, freezes it, and applies each generation to a different chronological, non-overlapping development epoch from the evolution split. The output artifacts are saved under `results/population/`:

- `generation_stats.csv`
- `population_state.json`
- `lineage.csv`

This runner intentionally does not evaluate the final test period.
