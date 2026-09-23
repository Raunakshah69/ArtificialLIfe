# Experimental evaluation framework

This document describes the M5 evaluation layer for the artificial-life trading ecosystem. The goal is to measure the frozen architecture scientifically without modifying the core trading or evolution rules.

## Scope

The experiments cover:

- forecasting model comparison across SimpleRNN, LSTM, and GRU
- population evolution across chronological development epochs
- lineage analysis from recorded ancestry
- manual baseline comparison
- buy-and-hold benchmark comparison
- final held-out evaluation after the development experiment is frozen

## Development vs final test

The development experiment measures what happened during the search process. It is used for model comparison, population evolution, lineage tracking, and benchmark comparison. The final held-out test is a separate out-of-sample evaluation stage that is never allowed to feed back into model selection or evolution.

The final test stage may only be run after a completed development experiment exists. It is not permitted in development-only mode.

## Reproducibility

Each experiment records:

- random seed
- ticker
- date range
- interval
- target mode
- feature names
- sequence length
- forecaster type
- population size
- generation count
- mutation rate
- mutation sigma
- survival threshold
- immigrant fraction
- transaction cost
- starting capital
- experiment metadata

The dataset manifest and raw-data hash are reused when available to ensure that the evaluation is traceable to the correct market data snapshot.

## Metrics

### Forecasting

For each model, the experiment records:

- model name
- parameter count
- training time
- validation MAE
- validation RMSE
- number of epochs
- best validation loss

### Population

Each generation records:

- generation
- epoch start/end
- population size
- alive/dead counts
- survival rate
- zero-trade count
- mean, median, best, and worst capital
- mean and best return
- genetic diversity
- immigrant count
- sexual/asexual child counts
- mutation count
- elite information

### Lineage

The lineage analysis records root lineages, surviving vs extinct lineages, maximum depth, average depth, breadth, number of generations survived, and the best lineage path based on the final best agent and its ancestry.

### Trading benchmarks

Manual baseline and buy-and-hold are evaluated on the same period and use the same transaction costs, starting capital, and market data.

Sharpe is only reported when the return series is sufficiently informative; otherwise it is set to null with an explanatory flag.

## Baselines and benchmarks

The manual baseline is a deterministic non-evolving agent using the same frozen forecast and transaction costs as the main system.

The buy-and-hold benchmark is a simple purchase of the instrument at the beginning of the evaluation window and hold through the end of the period.

## Artifact layout

Experiments are saved under results/experiments/<experiment_id> with:

- config.yaml
- manifest.json
- forecasting_metrics.csv
- generation_metrics.csv
- lineage_metrics.json
- strategy_comparison.csv
- computational_metrics.json
- development/
- final_test/

No experiment silently overwrites an earlier one. The experiment ID is generated deterministically from the timestamp and seed.

## How to reproduce

Run a lightweight development experiment:

python experiments/run_experiment.py --population-size 10 --generations 3 --development-only

Run the full default experiment:

python experiments/run_experiment.py --population-size 100

A final held-out test can be run after the development experiment has completed:

python experiments/run_experiment.py --final-test --development-experiment results/experiments/<experiment_id>

## Limitations

- Development results are not proof of future profitability.
- Final test results are out-of-sample and must remain separated from learning/evolution.
- The experiments are designed around the existing architecture and do not add new evolutionary operators.
- The shared forecaster remains frozen during population evolution.
