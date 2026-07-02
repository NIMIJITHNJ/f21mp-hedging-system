# Proposal-Aligned Three Planner Evaluation with VAL Validation

The proposal states that the planning layer should be evaluated using plan generation time, plan quality, and plan validity. It also specifies a five-second ENHSP timeout and fallback logging. This report evaluates the current rule-guided baseline, autonomous symbolic planner, and numerical planner on the same common scenario suite, and adds VAL validation for the exported PDDL domain, problem, and plan files.

| Planner | Cases | Correct | Correctness Rate | ENHSP Valid Rate | VAL Valid Rate | Fallbacks | Avg Time ms | Max Time ms | Within 5s | Avg Plan Length | Avg Txn Cost | Avg Turnover | Avg Hedge Gap | Max Gap | Constraint Success | PDDL Trace | Explanation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|
| current_rule_guided_up | 12 | 12 | 1.0 | 1.0 | 1.0 | 0 | 754.753 | 1238.271 | True | 1.167 | 88.167 | 1.5 | 0.75 | 3.0 | 6/6 | True | True |
| autonomous_symbolic_planner | 12 | 12 | 1.0 | 1.0 | 1.0 | 0 | 648.314 | 1218.985 | True | 1.25 | 450.0 | 1.5 | 0.75 | 3.0 | 6/6 | True | True |
| numerical_planner | 12 | 12 | 1.0 | 1.0 | 1.0 | 0 | 740.448 | 1223.177 | True | 1.25 | 450.0 | 1.5 | 0.75 | 3.0 | 6/6 | True | True |

## Interpretation

All planners are assessed using the same common scenario suite. Plan validity is measured in two ways: ENHSP solved status and external VAL validation of the exported PDDL domain, problem, and generated plan file.

The final planner should be selected by prioritising VAL-valid plans, constraint satisfaction, fallback rate, response time, and then plan quality measures such as transaction cost, turnover, plan length, and hedge gap.