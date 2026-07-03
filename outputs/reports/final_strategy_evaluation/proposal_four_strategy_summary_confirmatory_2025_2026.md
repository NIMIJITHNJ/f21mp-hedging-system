# Proposal-Aligned Four Strategy Financial Evaluation

Period: `confirmatory_2025_2026`

This report compares the four strategy structure required by the proposal: no hedge, static hedge, ML/rule-based hedge, and planning-based hedge.

| Strategy | Proposal Role | Total Return % | Annual Return % | Volatility % | Sharpe | Sortino | Downside Dev % | Max Drawdown % | CVaR 95% % | Hedge Ratio Std | Avg Hedge Ratio | Turnover | Txn Cost % | Hedging Effectiveness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_hedge | 1_no_hedge | -10.27 | -8.46 | 13.22 | -1.16 | -1.687 | 9.09 | -17.61 | -1.95 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| static_50 | 2_static_50 | -7.33 | -6.02 | 7.32 | -1.737 | -2.58 | 4.93 | -10.74 | -1.04 | 0.0 | 0.5 | 0.5 | 0.01 | 0.6937 |
| ml_rule_based_gjr | 3_ml_rule_based_gjr | -7.87 | -6.47 | 6.15 | -2.145 | -3.387 | 3.89 | -8.81 | -0.84 | 0.2452 | 0.5785 | 13.6 | 0.272 | 0.7838 |
| planning_based_numerical | 4_planning_based_numerical | -8.15 | -6.7 | 5.21 | -2.581 | -3.816 | 3.52 | -10.85 | -0.75 | 0.1729 | 0.8334 | 10.7807 | 0.2156 | 0.8449 |

## Interpretation Guide

The no-hedge and static-50 strategies are direct baselines. The ML/rule-based strategy uses the GJR-GARCH volatility regime mapped into fixed hedge levels. The planning-based strategy applies the numerical planner to the dynamic scaled minimum-variance hedge target and converts it into executable futures contracts under margin, transaction-cost, expiry and re-hedging constraints.

The strongest planning claim should focus on risk reduction and operational feasibility, not only return. The key metrics are volatility, maximum drawdown, downside deviation, CVaR, hedge-ratio stability, turnover, transaction cost, and hedging effectiveness.