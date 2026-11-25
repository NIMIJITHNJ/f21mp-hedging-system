# Hedra · Dynamic and explainable risk hedging

Hedra is the research prototype for the Heriot-Watt University MSc Artificial Intelligence dissertation *Integrating Machine Learning Forecasts with AI Planning for Dynamic, Explainable Risk Hedging in Financial Markets*. It evaluates how volatility forecasts and explicit planning constraints can help manage the market exposure of a ten-stock, equal-weight Indian equity portfolio using short NIFTY 50 futures.

## Research design and final system

The original proposal explored LSTM-GARCH risk classification for planning-driven hedging. The project evaluated fixed-split LSTMs, classical Random Forest and XGBoost classifiers, and an LSTM walk-forward experiment with boundary purging. The selected walk-forward LSTM predicted only the Low-risk class on the confirmatory sample. These runs remain in the repository as exploratory evidence. See the [LSTM walk-forward findings](outputs/archive/lstm_walkforward_final_20260606_170811/lstm_walkforward_findings.md).

The final pipeline uses a **one-day GJR-GARCH(1,1) forecast with Student-t errors**. It was selected by mean QLIKE in the 2020–2024 volatility-regression walk-forward study, with stress-period MAE, MAE and RMSE as secondary criteria. That study compared statistical, machine-learning and hybrid models at one-, five- and ten-day horizons. Statistical backtest forecasts use expanding annual refits with prior-year data. The separate January 2025–March 2026 period was reserved for confirmation. See the [regression study](outputs/archive/regression_walkforward_findings.md).

1. **Market inputs.** Daily equity, spot index and continuous futures data feed feature construction and return alignment. Macroeconomic series supply additional candidate forecasting features.
2. **Risk and hedge targets.** Forecasts provide volatility regimes. The rule-based comparator maps regimes to fixed ratios. The final planning strategy starts from a GJR-scaled rolling minimum-variance target.
3. **Planning and simulated execution.** The numerical planner converts the target into whole futures contracts and accounts for margin, transaction-cost limits, rehedging permissions and expiry. The research also compared a rule-guided Unified Planning model and an autonomous symbolic model. The implementation can export PDDL and record the action sequence, fallback, hedge gap, constraint state and explanation. The submitted common-scenario comparison includes separate VAL results. See the [planner comparison](outputs/reports/planning/three_planner_common_scenario_comparison_val.md).
4. **Dashboard.** Streamlit presents live monitoring, risk signals and hedge recommendations using read-only Kite data. Historical Backtesting shows a date-based replay, decision explanations and the stored benchmark. Live selector logic and replay sizing have their own rules. The published financial comparison uses the fixed runner settings below.

## Four-strategy historical benchmark

The figures below are the archived confirmatory results for January 2025 through March 2026. The **2020–2024 selection period** is reported separately. The planning benchmark models an INR 8 million portfolio, INR 2 million in available margin, a fixed lot size of 65, 12% margin and 2 bps transaction costs.

| Strategy | Total return | Annualised volatility | Max drawdown | Daily CVaR 95% |
| --- | ---: | ---: | ---: | ---: |
| No hedge | −10.27% | 13.22% | −17.61% | −1.95% |
| Static 50% futures hedge | −7.33% | 7.32% | −10.74% | −1.04% |
| GJR-GARCH regime rules | −7.87% | 6.15% | −8.81% | −0.84% |
| Numerical planning | −8.15% | 5.21% | −10.85% | −0.75% |

The numerical strategy had the lowest volatility and least negative daily CVaR in this sample. The static hedge had the highest total return, and the regime-rule strategy had the shallowest drawdown. See the [complete results for both periods](outputs/reports/final_strategy_evaluation/proposal_four_strategy_summary_all_periods.md).

## Setup and use

The research environment uses Python 3.12, NumPy, pandas, SciPy, arch, scikit-learn, XGBoost, LightGBM and TensorFlow/Keras. The application uses Streamlit and Kite Connect. Planning uses Unified Planning with the Java-based ENHSP engine. Java 17 was available in the Windows validation environment. VAL is an optional external validator. The repository includes its setup instructions and selected PDDL examples.

From the project root on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The dashboard entry point requires a Kite API key and secret, and both modes sit behind the authenticated session. Keep credentials in a local `.env` file as described in [Reproduction and evidence](docs/reproduction.md). The offline experiment scripts and stored reports can be used independently of the dashboard. Run reproduction commands in a disposable copy because the scripts write to fixed repository-relative paths.

| Location | Contents |
| --- | --- |
| `dashboard/` | Streamlit interface for live monitoring and historical replay |
| `src/data/`, `src/features/`, `scripts/` | Acquisition, checks, feature building and end-of-day refresh |
| `src/models/` | Classification, volatility regression and live risk signal |
| `src/planning/`, `src/broker/` | Planning, futures sizing and read-only Kite integration |
| `src/evaluation/` | Portfolio construction, baselines and historical evaluation |
| `data/external/`, `data/reference/` | Existing macro snapshots and replay lot-size schedule |
| `outputs/`, `docs/evidence/` | Historical reports, figures and selected synthetic PDDL evidence |

## Data availability and limitations

The historical experiment uses daily NIFTY 50 spot, continuous futures and ten constituent equities sourced through Zerodha Kite. The modelling history starts in 2012. Yahoo Finance provides the Brent, gold and USD/INR inputs through yfinance. The code reads the India 10-year yield from an existing CSV whose upstream provenance is undocumented. Raw market data, processed splits and private audit files are held locally. Access to the original datasets and permission to redistribute vendor inputs must be established independently. See the [data and reproduction notes](docs/reproduction.md).

The evidence comes from one market, one portfolio and one confirmatory window. The project uses daily proxies for realised volatility and simplified cost, margin and execution modelling. The financial backtest uses a fixed lot size and a last-Thursday expiry approximation. Historical replay uses a separate lot-size schedule. Archived results describe a simulation and do not establish expected live performance.

**Research prototype.** Hedra provides research analysis and simulated hedge recommendations. It does not submit orders. It is unsuitable for production trading or financial advice. Futures hedging can incur losses, basis risk and margin calls.

[Submission comparison](docs/submission-comparison.md) · [Reproduction and evidence](docs/reproduction.md) · [Reconciliation verification](docs/reconciliation-verification.md)
