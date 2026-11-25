# Autonomous Symbolic Planner Edge Case Results

Total cases: 12
Passed: 12
Failed: 0

| Scenario | Expected | Actual | Status | Fallback | Passed |
|---|---|---|---|---|---|
| edge_open_new_hedge | OPEN_HEDGE | OPEN_HEDGE | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_hold_no_hedge_needed | HOLD_POSITION | HOLD_POSITION | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_hold_same_position | HOLD_POSITION | HOLD_POSITION | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_increase_existing_hedge | INCREASE_HEDGE | INCREASE_HEDGE | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_reduce_existing_hedge | REDUCE_HEDGE | REDUCE_HEDGE | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_close_existing_hedge | CLOSE_HEDGE | CLOSE_HEDGE | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_margin_insufficient | HOLD_POSITION | HOLD_POSITION | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_transaction_cost_too_high | HOLD_POSITION | HOLD_POSITION | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_rehedge_not_allowed | HOLD_POSITION | HOLD_POSITION | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_roll_then_hold | ROLL_FUTURES_FORWARD -> HOLD_POSITION | ROLL_FUTURES_FORWARD -> HOLD_POSITION | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_roll_then_increase | ROLL_FUTURES_FORWARD -> INCREASE_HEDGE | ROLL_FUTURES_FORWARD -> INCREASE_HEDGE | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |
| edge_roll_then_close | ROLL_FUTURES_FORWARD -> CLOSE_HEDGE | ROLL_FUTURES_FORWARD -> CLOSE_HEDGE | PlanGenerationResultStatus.SOLVED_SATISFICING | False | True |