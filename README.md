# F21MP Dissertation: Dynamic Explainable Risk Hedging System

MSc Artificial Intelligence — Heriot-Watt University  
Supervised by Dr. Ron Petrick  
Submission: 23 July 2026

## Project
A three-layer hybrid architecture combining LSTM-GARCH risk prediction,
PDDL-based AI planning, and SHAP explainability for dynamic portfolio
hedging using Nifty 50 index futures (NSE, India).

## Stack
Python 3.14 · TensorFlow/Keras · arch · Unified Planning Library · SHAP · Kite Connect

## Data

Raw data is not committed to this repository. To regenerate:

1. Add your Kite Connect credentials to `.env`
2. Run `python src/data/fetch_data.py`

This fetches 557 trading days (Jan 2023 – Mar 2025) for:
- Nifty 50 spot index
- Nifty continuous futures (with OI)
- 10 Nifty 50 constituent stocks