# Constraint-Aware Hedge Planning and Futures Execution Findings

## Objective

A constraint-aware planning layer was implemented to convert predicted market risk and a covariance-based target hedge ratio into explainable NIFTY futures hedging actions.

## Architecture

The implemented workflow is:

1. GJR-GARCH produces the next-day volatility forecast.
2. The forecast is mapped to a LOW, MEDIUM, or HIGH risk regime.
3. The rolling 60-day minimum-variance model provides the core hedge ratio.
4. Unified Planning and ENHSP generate a constraint-aware action plan.
5. The futures execution calculator converts the approved ratio into whole NIFTY futures contracts.
6. Exact margin validation determines whether the recommendation can be executed.

## Planning Actions

The original proposal category for adjusting a hedge ratio was refined into separate actions for improved explainability.

The implemented PDDL planning actions are:

1. OPEN_HEDGE
2. INCREASE_HEDGE
3. REDUCE_HEDGE
4. CLOSE_HEDGE
5. HOLD_POSITION
6. ROLL_FUTURES_FORWARD

## Constraints Modelled

The planner currently evaluates:

- Current hedge position
- Predicted volatility-risk regime
- Core target hedge ratio
- Maximum permitted hedge ratio
- No-trade band
- Rehedging-frequency control
- Transaction-cost threshold
- Margin availability
- Futures expiry and rolling requirement

## Planner Validation

Unified Planning and the ENHSP numeric planner successfully generated plans for:

- Opening a new hedge position
- Increasing an existing hedge
- Reducing an existing hedge
- Closing an existing hedge
- Holding the current position
- Rolling an expiring futures contract forward and maintaining the position
- Holding an increase request when additional margin is unavailable

The deterministic fallback remains available if ENHSP does not return a usable plan within the configured timeout.

## Futures Execution Layer

The execution calculator converts the target hedge ratio into a valid whole number of short NIFTY futures contracts:

required contracts =
portfolio value × target hedge ratio
------------------------------------
futures price × lot size

The calculator also reports:

- Target futures notional
- Contract notional
- Current and recommended short-contract quantities
- Contract change
- Achieved hedge ratio after whole-contract rounding
- Residual hedge-ratio error
- Additional margin requirement
- Final execution instruction

## Integrated Workflow Validation

The integrated workflow successfully produced the following outcomes:

- OPEN_HEDGE → SELL_TO_OPEN_SHORT_FUTURES
- INCREASE_HEDGE → SELL_ADDITIONAL_SHORT_FUTURES
- REDUCE_HEDGE → BUY_TO_REDUCE_SHORT_FUTURES
- CLOSE_HEDGE → BUY_TO_CLOSE_SHORT_FUTURES
- HOLD_POSITION → HOLD_EXISTING_SHORT_POSITION
- ROLL_FUTURES_FORWARD → maintain the required contract count after rolling
- Insufficient exact margin → DO_NOT_EXECUTE_INSUFFICIENT_MARGIN

## Lot Size Handling

The execution module uses a configurable lot-size input. The demonstration currently uses a NIFTY futures lot size of 65.

In the final live integration, the applicable lot size, nearest valid futures expiry, live futures price, current position, and available margin should be obtained dynamically from broker API data rather than hardcoded.

## Decision

Retain the implemented Unified Planning and ENHSP architecture as the constraint-aware planning layer.

Retain the separate futures execution calculator for whole-contract conversion and exact margin validation.

The next stage will connect the workflow to the existing prediction outputs and broker-data integration.
