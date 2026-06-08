# Hedge-Performance Baseline Findings

## Objective

The selected 1-day GJR-GARCH volatility forecast was tested as part of a practical Nifty 50 futures hedging system. The goal was to determine whether volatility forecasts improve portfolio risk reduction after turnover and transaction costs.

## Portfolio and Hedge Instrument

- Portfolio: equal-weight portfolio of 10 Nifty 50 constituent stocks
- Hedge instrument: Nifty 50 continuous futures
- Backtest selection period: January 2020 to December 2024
- Confirmatory period: January 2025 to March 2026
- Transaction-cost assumption: 2 basis points per one-way futures turnover
- Forecast timing rule: a forecast generated at the end of day t is applied only to the futures position for day t+1

## Strategies Compared

1. No hedge
2. Static 50% hedge
3. Literature VECM static reference ratio: 0.8824
4. Portfolio-specific static minimum-variance ratio: 1.0205
5. Rolling 60-day minimum-variance hedge ratio
6. GJR-GARCH regime-based dynamic hedge
7. GJR-GARCH scaled minimum-variance hedge

## Selection-Period Findings: 2020 to 2024

The portfolio-specific static minimum-variance ratio achieved the lowest volatility:

- Hedge ratio: 1.0205
- Volatility: 5.72%
- Hedging effectiveness: 92.13%
- Maximum drawdown: -10.95%
- Total transaction cost: 0.02%

The rolling minimum-variance strategy also performed strongly:

- Average hedge ratio: 0.9845
- Volatility: 5.79%
- Hedging effectiveness: 91.93%
- Maximum drawdown: -12.00%

## Confirmatory Findings: January 2025 to March 2026

The strongest risk reduction came from:

- Literature VECM static reference: 4.81% volatility
- Rolling minimum-variance hedge: 4.82% volatility
- Portfolio-specific static minimum-variance hedge: 5.05% volatility

## Finding

GJR-GARCH is useful as a volatility-risk forecasting model, but volatility alone is not sufficient to determine the optimal futures hedge ratio.

The covariance-based hedge strategies outperformed the initial GJR-GARCH regime and scaling rules. The GJR-based rules also created higher turnover.

## Decision

Retain GJR-GARCH as a volatility-risk signal.

Use the covariance-based hedge ratios as the current practical baseline.

The next stage will evaluate DCC-GARCH as a candidate dynamic hedge-ratio engine because it models time-varying covariance between portfolio returns and futures returns.
