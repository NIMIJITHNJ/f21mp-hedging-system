# DCC-GJR-GARCH Hedge-Ratio Benchmark Findings

## Objective

A DCC-GJR-GARCH model was evaluated as an advanced dynamic hedge-ratio engine for the Nifty 50 futures hedging system.

## Method

- Portfolio: equal-weight portfolio of 10 Nifty 50 constituent stocks
- Hedge instrument: Nifty 50 continuous futures
- Marginal volatility models: GJR-GARCH(1,1) with Student-t errors
- Dependence model: DCC(1,1)
- Hedge-ratio formula: conditional covariance divided by futures variance
- Annual expanding refits using data available only up to the previous year-end
- Selection period: January 2020 to December 2024
- Confirmatory period: January 2025 to March 2026
- Transaction-cost assumption: 2 basis points per one-way futures turnover

## Model Stability

The annual DCC persistence values remained below one:

- 2020: 0.9206
- 2021: 0.9845
- 2022: 0.9445
- 2023: 0.9677
- 2024: 0.9594
- 2025: 0.9418
- 2026: 0.9505

This indicates stable but persistent time-varying correlation.

## Selection-Period Results: 2020 to 2024

- Portfolio-specific static minimum-variance hedge: 5.72% volatility
- Rolling minimum-variance hedge: 5.79% volatility
- Literature VECM static reference: 6.13% volatility
- DCC-GJR-GARCH hedge: 6.37% volatility

DCC-GJR-GARCH turnover was 23.11, compared with 9.56 for the rolling minimum-variance hedge.

## Confirmatory Results: January 2025 to March 2026

- Literature VECM static reference: 4.81% volatility
- Rolling minimum-variance hedge: 4.82% volatility
- DCC-GJR-GARCH hedge: 4.87% volatility
- Portfolio-specific static minimum-variance hedge: 5.05% volatility

## Finding

DCC-GJR-GARCH successfully captures changing covariance and remains stable across annual refits. However, the unsmoothed daily DCC hedge ratio does not outperform the simpler rolling minimum-variance hedge after accounting for turnover and transaction costs.

## Decision

Retain DCC-GJR-GARCH as an evaluated advanced benchmark.

Use rolling minimum variance as the current practical hedge-ratio baseline.

The next stage will test a smoothed DCC hedge ratio with a no-trade band to reduce unnecessary rebalancing.
