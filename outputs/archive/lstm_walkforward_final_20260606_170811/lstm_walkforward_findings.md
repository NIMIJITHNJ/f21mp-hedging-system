# Final LSTM Walk-Forward Findings

## Objective

The original proposed model used an LSTM-GARCH approach to classify future Nifty 50 volatility into Low, Medium, and High-risk regimes.

## Experiments Completed

- Dataset extended from January 2012 to March 2026
- Prediction horizons tested: 3, 5, 7, 10, and 14 trading days
- Six LSTM hyperparameter configurations tested
- Five random seeds tested: 7, 21, 42, 84, and 123
- Walk-forward evaluation completed across five market periods from 2020 to 2024
- Boundary purging applied to avoid future-label leakage
- A separate confirmatory evaluation was performed for January 2025 to March 2026

## Best Walk-Forward Candidate

- Prediction horizon: 7 days
- Configuration: C3
- LSTM units: 64
- Dense units: 32
- Dropout: 0.2
- Batch size: 32
- Learning rate: 0.001
- Mean macro-F1: 0.2946
- Mean High-risk recall: 0.3753
- Mean minimum class recall: 0.0990
- Collapse rate: 0.56

## Confirmatory Evaluation

- Accuracy: 0.5066
- Macro-F1: 0.2242
- High-risk recall: 0.0000
- Minimum class recall: 0.0000

The confirmatory model predicted only the Low-risk class and failed to detect Medium- and High-risk periods.

## Research Finding

No LSTM configuration passed the required stability criteria. The results show that standalone multiclass LSTM classification does not generalise reliably across changing Indian market regimes.

## Decision

The LSTM-GARCH approach will be retained as an evaluated benchmark and discussed in the dissertation. It will not be used as the primary forecasting model.

The next stage will evaluate volatility-regression models and hybrid GARCH-ML approaches using the same leakage-safe walk-forward framework.
