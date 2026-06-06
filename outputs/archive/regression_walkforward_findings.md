# Volatility Regression Walk-Forward Findings

## Objective

The original multiclass LSTM approach was unstable across changing market regimes. A leakage-safe volatility-regression benchmark was therefore performed to identify a more stable risk-estimation model for the Nifty 50 hedging system.

## Data and Evaluation Design

- Market-data coverage: January 2012 to March 2026
- Usable feature coverage: February 2012 to March 2026
- Prediction horizons tested: 1, 5, and 10 trading days
- Walk-forward evaluation periods: 2020 to 2024
- Additional confirmatory period: January 2025 to March 2026
- Boundary purging applied using actual Nifty trading dates
- Statistical forecasts generated using annual expanding refits with prior-year data only
- Two complete runs performed using the same fixed protocol
- Core report files matched exactly using SHA-256 checks

## Models Compared

1. GARCH(1,1)
2. EGARCH(1,1) with Student-t errors
3. GJR-GARCH(1,1) with Student-t errors
4. HAR-inspired Ridge regression
5. XGBoost regression
6. LightGBM regression
7. GARCH-XGBoost hybrid
8. EGARCH-XGBoost hybrid
9. GJR-GARCH-XGBoost hybrid

## Selected Forecasting Model

The strongest walk-forward candidate was GJR-GARCH with a 1-day prediction horizon.

### Walk-Forward Performance: 2020 to 2024

- Mean QLIKE: -2.773523
- Mean MAE: 0.092283
- Mean RMSE: 0.125974
- Mean directional accuracy: 0.671157
- Mean stress-period MAE: 0.130014

### Confirmatory Performance: January 2025 to March 2026

- QLIKE: -3.247310
- MAE: 0.070428
- RMSE: 0.087429
- R-squared: 0.076308
- Directional accuracy: 0.714286
- Stress-period MAE: 0.088021

## Finding

The simpler asymmetric GJR-GARCH model outperformed the ML and hybrid regression models under the current benchmark. The result was reproduced exactly across two independent executions of the same fixed protocol.

The walk-forward mean R-squared value was negative, so the model should not be described as superior under every forecasting metric. Its strongest evidence is its QLIKE performance, directional accuracy, confirmatory-period performance, and reproducibility.

## Decision

Use GJR-GARCH as the primary short-term volatility-risk estimator for the next stage.

The next step is to evaluate whether its forecasts improve real portfolio-hedging outcomes after accounting for turnover and transaction costs.
