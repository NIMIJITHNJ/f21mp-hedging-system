# Three Planner Common Scenario Comparison with VAL Validation

All three planner variants were evaluated on the same common scenario suite. Each generated action sequence was additionally validated using VAL against the exported PDDL domain and problem files.

| Scenario | Planner | Expected | Actual | Sequence | ENHSP Valid | VAL Status | VAL Valid | Fallback | Time ms | Gap | Constraint OK |
|---|---|---|---|---|---|---|---|---|---:|---:|---|
| common_open_new_hedge | current_rule_guided_up | OPEN_HEDGE | OPEN_HEDGE | OPEN_HEDGE | True | VALID | True | False | 404.973 | 0 | True |
| common_open_new_hedge | autonomous_symbolic_planner | OPEN_HEDGE | OPEN_HEDGE | OPEN_HEDGE | True | VALID | True | False | 225.197 | 0 | True |
| common_open_new_hedge | numerical_planner | OPEN_HEDGE | OPEN_HEDGE | OPEN_HEDGE | True | VALID | True | False | 1205.623 | 0 | True |
| common_hold_no_hedge_needed | current_rule_guided_up | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 258.735 | 0 | True |
| common_hold_no_hedge_needed | autonomous_symbolic_planner | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 1194.936 | 0 | True |
| common_hold_no_hedge_needed | numerical_planner | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 248.056 | 0 | True |
| common_hold_same_position | current_rule_guided_up | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 1220.174 | 0 | True |
| common_hold_same_position | autonomous_symbolic_planner | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 247.387 | 0 | True |
| common_hold_same_position | numerical_planner | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 1221.672 | 0 | True |
| common_increase_existing_hedge | current_rule_guided_up | INCREASE_HEDGE | INCREASE_HEDGE | INCREASE_HEDGE | True | VALID | True | False | 276.638 | 0 | True |
| common_increase_existing_hedge | autonomous_symbolic_planner | INCREASE_HEDGE | INCREASE_HEDGE | INCREASE_HEDGE | True | VALID | True | False | 1205.613 | 0 | True |
| common_increase_existing_hedge | numerical_planner | INCREASE_HEDGE | INCREASE_HEDGE | INCREASE_HEDGE | True | VALID | True | False | 268.462 | 0 | True |
| common_reduce_existing_hedge | current_rule_guided_up | REDUCE_HEDGE | REDUCE_HEDGE | REDUCE_HEDGE | True | VALID | True | False | 1238.271 | 0 | True |
| common_reduce_existing_hedge | autonomous_symbolic_planner | REDUCE_HEDGE | REDUCE_HEDGE | REDUCE_HEDGE | True | VALID | True | False | 238.648 | 0 | True |
| common_reduce_existing_hedge | numerical_planner | REDUCE_HEDGE | REDUCE_HEDGE | REDUCE_HEDGE | True | VALID | True | False | 1223.177 | 0 | True |
| common_close_existing_hedge | current_rule_guided_up | CLOSE_HEDGE | CLOSE_HEDGE | CLOSE_HEDGE | True | VALID | True | False | 267.67 | 0 | True |
| common_close_existing_hedge | autonomous_symbolic_planner | CLOSE_HEDGE | CLOSE_HEDGE | CLOSE_HEDGE | True | VALID | True | False | 1206.691 | 0 | True |
| common_close_existing_hedge | numerical_planner | CLOSE_HEDGE | CLOSE_HEDGE | CLOSE_HEDGE | True | VALID | True | False | 273.601 | 0 | True |
| common_margin_insufficient | current_rule_guided_up | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 1222.555 | 3 | True |
| common_margin_insufficient | autonomous_symbolic_planner | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 249.375 | 3 | True |
| common_margin_insufficient | numerical_planner | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 1218.18 | 3 | True |
| common_transaction_cost_too_high | current_rule_guided_up | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 263.453 | 3 | True |
| common_transaction_cost_too_high | autonomous_symbolic_planner | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 1218.985 | 3 | True |
| common_transaction_cost_too_high | numerical_planner | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 264.08 | 3 | True |
| common_rehedge_not_allowed | current_rule_guided_up | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 1205.489 | 3 | True |
| common_rehedge_not_allowed | autonomous_symbolic_planner | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 251.799 | 3 | True |
| common_rehedge_not_allowed | numerical_planner | HOLD_POSITION | HOLD_POSITION | HOLD_POSITION | True | VALID | True | False | 245.134 | 3 | True |
| common_roll_then_hold | current_rule_guided_up | HOLD_POSITION | HOLD_POSITION | ROLL_FUTURES_FORWARD -> HOLD_POSITION | True | VALID | True | False | 1220.874 | 0 | True |
| common_roll_then_hold | autonomous_symbolic_planner | HOLD_POSITION | HOLD_POSITION | ROLL_FUTURES_FORWARD -> HOLD_POSITION | True | VALID | True | False | 262.744 | 0 | True |
| common_roll_then_hold | numerical_planner | HOLD_POSITION | HOLD_POSITION | ROLL_FUTURES_FORWARD -> HOLD_POSITION | True | VALID | True | False | 1221.108 | 0 | True |
| common_roll_then_increase | current_rule_guided_up | INCREASE_HEDGE | INCREASE_HEDGE | ROLL_FUTURES_FORWARD -> INCREASE_HEDGE | True | VALID | True | False | 266.141 | 0 | True |
| common_roll_then_increase | autonomous_symbolic_planner | INCREASE_HEDGE | INCREASE_HEDGE | ROLL_FUTURES_FORWARD -> INCREASE_HEDGE | True | VALID | True | False | 1216.104 | 0 | True |
| common_roll_then_increase | numerical_planner | INCREASE_HEDGE | INCREASE_HEDGE | ROLL_FUTURES_FORWARD -> INCREASE_HEDGE | True | VALID | True | False | 273.513 | 0 | True |
| common_roll_then_close | current_rule_guided_up | CLOSE_HEDGE | CLOSE_HEDGE | CLOSE_HEDGE | True | VALID | True | False | 1212.066 | 0 | True |
| common_roll_then_close | autonomous_symbolic_planner | CLOSE_HEDGE | CLOSE_HEDGE | ROLL_FUTURES_FORWARD -> CLOSE_HEDGE | True | VALID | True | False | 262.293 | 0 | True |
| common_roll_then_close | numerical_planner | CLOSE_HEDGE | CLOSE_HEDGE | ROLL_FUTURES_FORWARD -> CLOSE_HEDGE | True | VALID | True | False | 1222.771 | 0 | True |