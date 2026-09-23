# Trading Engine and Single-Agent Execution

## Execution timing convention

The simulation uses a close-to-close convention to avoid look-ahead bias.

At the close of session `t`:

1. information available through `t` is used
2. the frozen shared forecaster predicts the next period `t+1`
3. the decision head converts the forecast and recent normalized context into an action
4. the trade is executed at the close price of `t`
5. transaction cost is charged
6. the resulting position is then exposed in session `t+1` and later

This convention ensures no future information is used in the decision at time `t`.

## Long-only model

This milestone implements a long-only trading model. The agent may open or maintain long positions only. No shorting, leverage, or margin is allowed.

## Position sizing

The position size is a configurable fraction of available cash. The default is `0.25`.

The engine never allows negative cash through accidental oversizing and keeps all position sizing within available capital constraints.

## Transaction cost

Each entry and exit incurs a transaction-cost rate. The default is `0.001`.

Costs are handled explicitly in capital accounting and recorded in the trade ledger.

## Stop loss and take profit

For a long position:

- stop-loss price = `entry_price * (1 - stop_loss)`
- take-profit price = `entry_price * (1 + take_profit)`

During future market sessions, the engine inspects the next OHLC bar to determine whether either boundary was reached.

If both stop-loss and take-profit appear to be triggered within the same OHLC candle, the engine uses a conservative deterministic rule:

- STOP LOSS FIRST

This is required because daily OHLC data does not reveal intraday ordering, so the exact entry/exit sequence cannot be inferred.

## Capital accounting

The agent tracks:

- cash
- position quantity
- position market value
- total equity
- realized P&L
- unrealized P&L
- transaction costs

At every simulation step, total equity is appended to the equity history.

## Trade record schema

Each trade entry includes:

- agent_id
- timestamp/date
- action
- price
- quantity
- transaction_cost
- realized_pnl
- cash_after
- equity_after
- reason

Valid reasons are:

- ENTRY
- EXIT_SIGNAL
- STOP_LOSS
- TAKE_PROFIT

## Leakage prevention

The agent receives only information available through the current bar and uses the shared forecaster prediction without any retraining or mutation. The forecasting backbone is frozen and reused by the agent; the decision head is reconstructed from the genome and does not learn during inference.
