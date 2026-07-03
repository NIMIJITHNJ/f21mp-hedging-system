# Proposal-Aligned Four Strategy Financial Evaluation

Period: `selection_2020_2024`

This report compares the four strategy structure required by the proposal: no hedge, static hedge, ML/rule-based hedge, and planning-based hedge.

| Strategy | Proposal Role | Total Return % | Annual Return % | Volatility % | Sharpe | Sortino | Downside Dev % | Max Drawdown % | CVaR 95% % | Hedge Ratio Std | Avg Hedge Ratio | Turnover | Txn Cost % | Hedging Effectiveness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_hedge | 1_no_hedge | 112.01 | 16.46 | 20.39 | 0.428 | 0.476 | 18.34 | -39.85 | -3.13 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| static_50 | 2_static_50 | 52.05 | 8.87 | 11.3 | 0.177 | 0.205 | 9.75 | -23.53 | -1.67 | 0.0 | 0.5 | 0.5 | 0.01 | 0.6929 |
| ml_rule_based_gjr | 3_ml_rule_based_gjr | 20.56 | 3.86 | 7.6 | -0.356 | -0.487 | 5.56 | -14.09 | -1.06 | 0.2748 | 0.678 | 44.9 | 0.898 | 0.861 |
| planning_based_numerical | 4_planning_based_numerical | -3.11 | -0.64 | 7.27 | -0.982 | -1.315 | 5.43 | -23.0 | -1.06 | 0.214 | 0.9882 | 41.5095 | 0.8302 | 0.8728 |

## Interpretation Guide

The no-hedge and static-50 strategies are direct baselines. The ML/rule-based strategy uses the GJR-GARCH volatility regime mapped into fixed hedge levels. The planning-based strategy applies the numerical planner to the dynamic scaled minimum-variance hedge target and converts it into executable futures contracts under margin, transaction-cost, expiry and re-hedging constraints.

The strongest planning claim should focus on risk reduction and operational feasibility, not only return. The key metrics are volatility, maximum drawdown, downside deviation, CVaR, hedge-ratio stability, turnover, transaction cost, and hedging effectiveness.