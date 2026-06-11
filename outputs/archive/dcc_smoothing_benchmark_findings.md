# DCC Smoothing and No-Trade-Band Benchmark Findings

## Objective

The unsmoothed DCC-GJR-GARCH hedge ratio produced useful dynamic covariance estimates but generated excessive turnover. This stage tested whether EMA smoothing and a no-trade band could improve practical hedge execution.

## Method

- Core model: leakage-safe annual out-of-sample DCC-GJR-GARCH
- EMA smoothing spans tested: 1, 3, 5, 10, and 20 trading days
- No-trade bands tested: 0.00, 0.02, 0.05, and 0.10 hedge-ratio units
- Total DCC execution variants tested: 20
- Selection period: January 2020 to December 2024
- Confirmatory period: January 2025 to March 2026
- Transaction-cost assumption: 2 basis points per one-way futures turnover

## Best DCC Execution Variant

The strongest DCC execution rule on the 2020-2024 selection period was:

- Strategy: dcc_ema20_band0p10
- EMA smoothing window: 20 trading days
- No-trade band: 0.10
- Average hedge ratio: 1.0174
- Volatility: 6.05%
- Hedging effectiveness: 91.19%
- Maximum drawdown: -16.20%
- Total turnover: 3.28
- Total transaction cost: 0.07%

## Improvement Compared with Raw DCC

Raw DCC-GJR-GARCH:

- Volatility: 6.37%
- Maximum drawdown: -18.40%
- Turnover: 23.11
- Transaction cost: 0.46%

Smoothed DCC with no-trade band:

- Volatility: 6.05%
- Maximum drawdown: -16.20%
- Turnover: 3.28
- Transaction cost: 0.07%

The smoothing and no-trade-band layer reduced turnover by approximately 86%.

## Confirmatory Findings

For January 2025 to March 2026:

- Literature VECM static reference: 4.81% volatility
- Rolling minimum-variance hedge: 4.82% volatility
- Raw DCC-GJR-GARCH hedge: 4.87% volatility
- Smoothed DCC hedge: 4.93% volatility
- Portfolio-specific static minimum-variance hedge: 5.05% volatility

## Finding

Smoothing and a no-trade band substantially improve the practical efficiency of DCC-GJR-GARCH. However, the smoothed DCC hedge still does not outperform the simpler rolling minimum-variance hedge across the evaluation periods.

## Decision

Use the rolling 60-day minimum-variance hedge ratio as the core practical hedge-ratio engine.

Retain GJR-GARCH as the next-day volatility-prediction layer.

Retain DCC-GJR-GARCH and its smoothed variant as advanced benchmark evidence.

The next stage will build the planning layer that converts the predicted risk signal and core hedge ratio into actions: increase hedge, reduce hedge, or hold.
