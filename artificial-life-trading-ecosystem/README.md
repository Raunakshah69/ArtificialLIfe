# Artificial-Life Trading Ecosystem

This repository provides the foundational structure for an artificial-life trading ecosystem in which a shared, gradient-trained forecasting backbone supports a population of evolving trading agents.

## Core architectural principle

The recurrent forecasting model is shared and trained through gradient descent. The evolutionary genome is intentionally small and contains only the agent-specific decision head and trading parameters. The full forecasting backbone is not evolved.

## Repository layout

- `src/` contains the Python package and module skeletons.
- `configs/` stores project configuration and environment defaults.
- `data/` holds raw and processed datasets.
- `models/` stores forecasting and genome model artifacts.
- `app/` is reserved for the eventual dashboard or Streamlit app.
- `tests/` contains project validation and smoke tests.
- `docs/` contains architecture and design notes.

## Scope of this milestone

This repository foundation intentionally avoids implementing:

- data processing pipelines
- forecasting training logic
- trading strategies
- evolutionary algorithms
- real brokerage or market execution features

The focus here is a clean package layout, configuration system, utilities, and import-safe skeletons that can support Milestone 1 work.

## Quick start

Python 3.11 is the supported project environment. TensorFlow's forecasting tests are not available in the default Python 3.14 setup; running pytest there skips those tests. Use the Python 3.11 launcher explicitly on Windows:

```bash
cd artificial-life-trading-ecosystem
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests -q
```

On macOS or Linux:

```bash
cd artificial-life-trading-ecosystem
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Confirm the active interpreter with `python --version`; it should report Python 3.11.x. Do not use Python 3.14 for the full suite because the TensorFlow-dependent forecast tests will be skipped.

## Planned layers

- Data layer
- Forecasting layer
- Agent layer
- Trading simulation layer
- Evolutionary population layer
- Evaluation layer
- Lineage layer
- Dashboard layer
