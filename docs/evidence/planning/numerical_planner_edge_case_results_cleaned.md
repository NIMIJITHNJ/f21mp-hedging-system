# Numerical Planner Edge Case Results - Cleaned

This report separates action correctness from fallback usage.

Total cases: 12
Correct decisions: 12
Fallback cases: 0

| Scenario | Expected | Actual | Sequence | Fallback | Status | Current | Target | Executable | Gap | Correct |
|---|---|---|---|---|---|---:|---:|---:|---:|---|
| edge_open_new_hedge | OPEN_HEDGE | OPEN_HEDGE | OPEN_HEDGE | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 0 | 3 | 3 | 0 | True |
| edge_hold_no_hedge_needed | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 0 | 0 | 0 | 0 | True |
| edge_hold_same_position | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 3 | 3 | 3 | 0 | True |
| edge_increase_existing_hedge | INCREASE_HEDGE | INCREASE_HEDGE | INCREASE_HEDGE | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 1 | 4 | 4 | 0 | True |
| edge_reduce_existing_hedge | REDUCE_HEDGE | REDUCE_HEDGE | REDUCE_HEDGE | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 4 | 1 | 1 | 0 | True |
| edge_close_existing_hedge | CLOSE_HEDGE | CLOSE_HEDGE | CLOSE_HEDGE | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 3 | 0 | 0 | 0 | True |
| edge_margin_insufficient | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 1 | 4 | 1 | 3 | True |
| edge_transaction_cost_too_high | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 1 | 4 | 1 | 3 | True |
| edge_rehedge_not_allowed | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 1 | 4 | 1 | 3 | True |
| edge_roll_then_hold | HOLD_POSITION | HOLD_POSITION | ROLL_FUTURES_FORWARD -> HOLD_POSITION | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 3 | 3 | 3 | 0 | True |
| edge_roll_then_increase | INCREASE_HEDGE | INCREASE_HEDGE | ROLL_FUTURES_FORWARD -> INCREASE_HEDGE | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 1 | 4 | 4 | 0 | True |
| edge_roll_then_close | CLOSE_HEDGE | CLOSE_HEDGE | ROLL_FUTURES_FORWARD -> CLOSE_HEDGE | False | PlanGenerationResultStatus.SOLVED_SATISFICING | 3 | 0 | 0 | 0 | True |