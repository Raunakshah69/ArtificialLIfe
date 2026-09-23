# Architecture Overview

## Data layer

The data layer is responsible for ingesting and validating market observations, time-series feature sets, and simulation metadata. It defines the canonical schema for historical prices, volumes, technical indicators, and environment state. This layer is intentionally decoupled from trading logic and forecasting logic so that market data can be swapped without altering the remainder of the system.

## Forecasting layer

The forecasting layer contains the shared recurrent model used to predict near-future market behavior. This model is implemented with TensorFlow/Keras and is trained with gradient descent across the full population of training epochs. The critical design rule is that only the decision head and trading parameters are evolvable. The shared recurrent backbone remains a population-wide asset, not an individual agent trait.

## Agent layer

The agent layer defines each simulated trading agent. Each agent owns private capital, inventory state, and an individual genome that controls its decision-policy head and trading parameters. This keeps the evolutionary search space compact while preserving interpretability and reproducibility across market epochs.

## Trading simulation layer

The trading simulation layer models the interaction between market observations, forecast output, and each agent’s policy decisions. It tracks order execution, transaction costs, cash balances, portfolio value, and step-by-step market state transitions. This layer provides the mechanics by which agent decisions produce performance outcomes over time.

## Evolutionary population layer

The evolutionary population layer manages the population of agents, selection pressure, mutation, reproduction, and survival rules. It uses agent fitness derived from capital growth and risk-adjusted performance. This layer makes the decision head mutable while keeping the forecasting backbone fixed and shared.

## Evaluation layer

The evaluation layer measures outcomes across epochs, including return, drawdown, survival rate, and lineage-based performance. It integrates forecasting quality, agent decisions, and market conditions to score each generation and provide population-level summaries for analysis.

## Lineage layer

The lineage layer records the ancestry of each agent, its descendants, and its historical performance. This structure supports long-term evolutionary analysis across generations and enables later inspection of successful trait patterns without requiring mutations to be tracked in a global state.

## Dashboard layer

The dashboard layer is responsible for rendering population statistics, market summaries, lineage graphs, and forecast diagnostics. It uses Streamlit and Plotly and remains a thin visualization layer that depends on evaluation and simulation outputs rather than owning business logic.

## Dependencies between layers

The layers interact in a directed flow:

1. Data layer provides market features to the forecasting layer and simulation layer.
2. Forecasting layer provides shared market predictions to the trading simulation and agent decision processes.
3. Agent layer consumes forecasts and emits trading actions.
4. Trading simulation layer updates capital and portfolio state based on those actions.
5. Evolutionary population layer uses outcomes to select survivors and generate offspring.
6. Evaluation layer summarizes the generation and exposes metrics to dashboards and reporting.
7. Lineage layer records ancestry and fitness history for later inspection.

This separation maintains a clean boundary between market signal generation, agent decision-making, evolutionary adaptation, and visualization.
