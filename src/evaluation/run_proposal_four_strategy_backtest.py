from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    sharpe_ratio,
    sortino_ratio,
    downside_deviation,
    max_drawdown,
    cvar,
    var,
    volatility,
    calmar_ratio,
)
from src.evaluation.portfolio import load_futures_prices
from src.planning.numerical_planner import NumericalPlannerInput, solve_numerical_planning


REPORT_DIR = ROOT / "outputs" / "reports" / "final_strategy_evaluation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DAILY = ROOT / "outputs" / "reports" / "hedge_performance" / "hedge_performance_daily_details.csv"
SOURCE_RATIOS = ROOT / "outputs" / "reports" / "hedge_performance" / "hedge_ratio_series.csv"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065

PORTFOLIO_VALUE = 8_000_000.0
LOT_SIZE = 65
MARGIN_RATE = 0.12
AVAILABLE_MARGIN = 2_000_000.0
TRANSACTION_COST_BPS = 2.0
MAX_TRADE_COST_MICROS = 5000
ROLL_THRESHOLD_DAYS = 3

DEFAULT_PERIODS = ["selection_2020_2024", "confirmatory_2025_2026"]
COST_BPS = 2.0

PROPOSAL_STRATEGY_MAP = {
    "no_hedge": "1_no_hedge",
    "static_50": "2_static_50",
    "gjr_regime_dynamic": "3_ml_rule_based_gjr",
    "planning_based_numerical": "4_planning_based_numerical",
}


def clean_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def risk_regime_from_target_ratio(target_ratio: float) -> str:
    if target_ratio <= 0.35:
        return "Low"
    if target_ratio <= 0.70:
        return "Medium"
    return "High"


def last_thursday(year: int, month: int) -> pd.Timestamp:
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    while end.weekday() != 3:
        end -= pd.Timedelta(days=1)
    return end.normalize()


def days_to_expiry(date: pd.Timestamp) -> int:
    current_expiry = last_thursday(date.year, date.month)
    if date.normalize() <= current_expiry:
        expiry = current_expiry
    else:
        next_month = date + pd.DateOffset(months=1)
        expiry = last_thursday(next_month.year, next_month.month)

    return max(0, int((expiry - date.normalize()).days))


def compute_strategy_summary(
    label: str,
    proposal_strategy: str,
    period: str,
    cost_bps: float,
    daily: pd.DataFrame,
    unhedged_returns: pd.Series,
) -> dict[str, Any]:
    returns = daily["hedged_return_net"].astype(float)
    gross_returns = daily["hedged_return_gross"].astype(float)
    hedge_ratios = daily["hedge_ratio"].astype(float)
    turnover = daily["turnover"].astype(float)
    transaction_cost = daily["transaction_cost"].astype(float)

    total_return = float(np.exp(returns.sum()) - 1.0)
    gross_total_return = float(np.exp(gross_returns.sum()) - 1.0)
    annual_return = float(np.exp(returns.mean() * TRADING_DAYS) - 1.0)

    unhedged_var = float(unhedged_returns.var())
    hedged_var = float(returns.var())
    hedging_effectiveness = 0.0
    if unhedged_var > 0 and not np.isnan(unhedged_var):
        hedging_effectiveness = float(1.0 - (hedged_var / unhedged_var))

    return {
        "label": label,
        "proposal_strategy": proposal_strategy,
        "period": period,
        "cost_bps": cost_bps,
        "n_days": int(len(daily)),
        "total_return_pct": round(total_return * 100.0, 2),
        "annual_return_pct": round(annual_return * 100.0, 2),
        "volatility_pct": round(volatility(returns) * 100.0, 2),
        "sharpe": round(sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE), 3),
        "sortino": round(sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE), 3),
        "downside_deviation_pct": round(downside_deviation(returns, risk_free_rate=RISK_FREE_RATE) * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown(returns) * 100.0, 2),
        "cvar_95_pct": round(cvar(returns, confidence=0.95) * 100.0, 2),
        "var_95_pct": round(var(returns, confidence=0.95) * 100.0, 2),
        "calmar": round(calmar_ratio(returns), 3),
        "hedge_ratio_std": round(float(hedge_ratios.std()), 4),
        "average_hedge_ratio": round(float(hedge_ratios.mean()), 4),
        "min_hedge_ratio": round(float(hedge_ratios.min()), 4),
        "max_hedge_ratio": round(float(hedge_ratios.max()), 4),
        "average_daily_turnover": round(float(turnover.mean()), 6),
        "total_turnover": round(float(turnover.sum()), 4),
        "total_transaction_cost_pct": round(float(transaction_cost.sum()) * 100.0, 4),
        "gross_total_return_pct": round(gross_total_return * 100.0, 2),
        "hedging_effectiveness": round(hedging_effectiveness, 4),
    }


def build_existing_strategy_daily(source_daily: pd.DataFrame, strategy: str, period: str) -> pd.DataFrame:
    daily = source_daily[
        (source_daily["strategy"] == strategy)
        & (source_daily["period"] == period)
        & (source_daily["cost_bps"].astype(float) == COST_BPS)
    ].copy()

    if daily.empty:
        raise ValueError(f"No rows found for strategy={strategy}, period={period}, cost_bps={COST_BPS}")

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def build_planning_strategy_daily(
    base_daily: pd.DataFrame,
    ratio_series: pd.DataFrame,
    futures_prices: pd.Series,
    period: str,
    target_column: str = "gjr_scaled_minvar",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratio_frame = ratio_series[["date", target_column]].copy()
    ratio_frame["date"] = pd.to_datetime(ratio_frame["date"])
    ratio_frame[target_column] = pd.to_numeric(ratio_frame[target_column], errors="coerce")
    ratio_frame[target_column] = ratio_frame[target_column].ffill().fillna(0.0)

    panel = base_daily[["date", "period", "cost_bps", "portfolio", "futures"]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(ratio_frame, on="date", how="left")
    panel[target_column] = panel[target_column].ffill().fillna(0.0)
    panel[target_column] = panel[target_column].clip(lower=0.0, upper=1.50)

    futures_prices = futures_prices.copy()
    futures_prices.index = pd.to_datetime(futures_prices.index)
    futures_prices = futures_prices.sort_index()

    current_contracts = 0
    previous_hedge_ratio = 0.0

    daily_rows = []
    decision_rows = []

    total = len(panel)

    for i, row in panel.iterrows():
        date = pd.Timestamp(row["date"])
        target_ratio = clean_float(row[target_column])

        if date in futures_prices.index:
            futures_price = float(futures_prices.loc[date])
        else:
            prior_prices = futures_prices.loc[:date]
            if prior_prices.empty:
                futures_price = 24_000.0
            else:
                futures_price = float(prior_prices.iloc[-1])

        margin_per_contract = futures_price * LOT_SIZE * MARGIN_RATE
        dte = days_to_expiry(date)
        risk_regime = risk_regime_from_target_ratio(target_ratio)

        planning_input = NumericalPlannerInput(
            scenario_name=f"financial_{date.strftime('%Y%m%d')}",
            portfolio_value=PORTFOLIO_VALUE,
            target_hedge_ratio=target_ratio,
            predicted_volatility=0.0,
            risk_regime=risk_regime,
            futures_price=futures_price,
            lot_size=LOT_SIZE,
            current_short_contracts=current_contracts,
            margin_per_contract=margin_per_contract,
            available_margin=AVAILABLE_MARGIN,
            days_to_expiry=dte,
            rehedge_allowed=True,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            max_trade_cost_micros=MAX_TRADE_COST_MICROS,
            roll_threshold_days=ROLL_THRESHOLD_DAYS,
        )

        result = solve_numerical_planning(planning_input)
        state = result.prepared_state

        executable_contracts = int(state["executable_contracts"])
        contract_notional = futures_price * LOT_SIZE
        hedge_ratio = executable_contracts * contract_notional / PORTFOLIO_VALUE

        turnover = abs(hedge_ratio - previous_hedge_ratio)
        transaction_cost = turnover * (TRANSACTION_COST_BPS / 10_000.0)

        portfolio_return = clean_float(row["portfolio"])
        futures_return = clean_float(row["futures"])

        hedged_return_gross = portfolio_return - hedge_ratio * futures_return
        hedged_return_net = hedged_return_gross - transaction_cost

        daily_rows.append({
            "date": date.date().isoformat(),
            "strategy": "planning_based_numerical",
            "period": period,
            "cost_bps": COST_BPS,
            "portfolio": portfolio_return,
            "futures": futures_return,
            "target_hedge_ratio": target_ratio,
            "hedge_ratio": hedge_ratio,
            "current_contracts_before": current_contracts,
            "executable_contracts": executable_contracts,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "hedged_return_gross": hedged_return_gross,
            "hedged_return_net": hedged_return_net,
        })

        decision_rows.append({
            "date": date.date().isoformat(),
            "target_source": target_column,
            "target_hedge_ratio": target_ratio,
            "risk_regime": risk_regime,
            "futures_price": futures_price,
            "days_to_expiry": dte,
            "current_contracts_before": current_contracts,
            "target_contracts": int(state["target_contracts"]),
            "executable_contracts": executable_contracts,
            "hedge_gap_contracts": int(state["hedge_gap_contracts"]),
            "primary_action": result.primary_action,
            "action_sequence": " -> ".join(result.action_sequence),
            "planner_status": result.planner_status,
            "used_fallback": result.used_fallback,
            "planning_time_ms": result.planning_time_ms,
            "margin_required_amount": int(state["margin_required_amount"]),
            "margin_available_amount": int(state["margin_available_amount"]),
            "transaction_cost_micros": int(state["transaction_cost_micros"]),
            "max_transaction_cost_micros": int(state["max_transaction_cost_micros"]),
            "pddl_domain_file": result.pddl_domain_file,
            "pddl_problem_file": result.pddl_problem_file,
        })

        current_contracts = executable_contracts
        previous_hedge_ratio = hedge_ratio

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total} planning days")

    return pd.DataFrame(daily_rows), pd.DataFrame(decision_rows)


def run_for_period(period: str) -> pd.DataFrame:
    if not SOURCE_DAILY.exists():
        raise FileNotFoundError(SOURCE_DAILY)
    if not SOURCE_RATIOS.exists():
        raise FileNotFoundError(SOURCE_RATIOS)

    source_daily = pd.read_csv(SOURCE_DAILY)
    ratio_series = pd.read_csv(SOURCE_RATIOS)
    futures_prices = load_futures_prices()

    no_hedge_daily = build_existing_strategy_daily(source_daily, "no_hedge", period)
    static_daily = build_existing_strategy_daily(source_daily, "static_50", period)
    rule_daily = build_existing_strategy_daily(source_daily, "gjr_regime_dynamic", period)

    planning_daily, planner_log = build_planning_strategy_daily(
        base_daily=no_hedge_daily,
        ratio_series=ratio_series,
        futures_prices=futures_prices,
        period=period,
        target_column="gjr_scaled_minvar",
    )

    final_daily = pd.concat(
        [
            no_hedge_daily.assign(proposal_strategy=PROPOSAL_STRATEGY_MAP["no_hedge"]),
            static_daily.assign(proposal_strategy=PROPOSAL_STRATEGY_MAP["static_50"]),
            rule_daily.assign(proposal_strategy=PROPOSAL_STRATEGY_MAP["gjr_regime_dynamic"]),
            planning_daily.assign(proposal_strategy=PROPOSAL_STRATEGY_MAP["planning_based_numerical"]),
        ],
        ignore_index=True,
    )

    unhedged_returns = no_hedge_daily["hedged_return_net"].astype(float)

    summaries = [
        compute_strategy_summary(
            label="no_hedge",
            proposal_strategy=PROPOSAL_STRATEGY_MAP["no_hedge"],
            period=period,
            cost_bps=COST_BPS,
            daily=no_hedge_daily,
            unhedged_returns=unhedged_returns,
        ),
        compute_strategy_summary(
            label="static_50",
            proposal_strategy=PROPOSAL_STRATEGY_MAP["static_50"],
            period=period,
            cost_bps=COST_BPS,
            daily=static_daily,
            unhedged_returns=unhedged_returns,
        ),
        compute_strategy_summary(
            label="ml_rule_based_gjr",
            proposal_strategy=PROPOSAL_STRATEGY_MAP["gjr_regime_dynamic"],
            period=period,
            cost_bps=COST_BPS,
            daily=rule_daily,
            unhedged_returns=unhedged_returns,
        ),
        compute_strategy_summary(
            label="planning_based_numerical",
            proposal_strategy=PROPOSAL_STRATEGY_MAP["planning_based_numerical"],
            period=period,
            cost_bps=COST_BPS,
            daily=planning_daily,
            unhedged_returns=unhedged_returns,
        ),
    ]

    summary_df = pd.DataFrame(summaries)

    daily_path = REPORT_DIR / f"proposal_four_strategy_daily_details_{period}.csv"
    planner_log_path = REPORT_DIR / f"proposal_four_strategy_planner_log_{period}.csv"
    summary_path = REPORT_DIR / f"proposal_four_strategy_summary_{period}.csv"
    md_path = REPORT_DIR / f"proposal_four_strategy_summary_{period}.md"
    metadata_path = REPORT_DIR / f"proposal_four_strategy_metadata_{period}.json"

    final_daily.to_csv(daily_path, index=False)
    planner_log.to_csv(planner_log_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "purpose": "Proposal-aligned four-strategy financial evaluation",
        "period": period,
        "cost_bps": COST_BPS,
        "strategies": {
            "1_no_hedge": "Existing no_hedge baseline",
            "2_static_50": "Existing static 50 percent hedge baseline",
            "3_ml_rule_based_gjr": "Existing GJR-GARCH volatility regime mapped to rule-based hedge ratios",
            "4_planning_based_numerical": "Numerical planner applied to gjr_scaled_minvar target hedge ratio with futures contract rounding, margin, cost, expiry and execution constraints",
        },
        "portfolio_value": PORTFOLIO_VALUE,
        "available_margin": AVAILABLE_MARGIN,
        "lot_size": LOT_SIZE,
        "margin_rate": MARGIN_RATE,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "timing_note": "Uses existing leakage-safe hedge-ratio series. Existing benchmark applies forecast produced at end of day t to day t+1 hedge position.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "# Proposal-Aligned Four Strategy Financial Evaluation",
        "",
        f"Period: `{period}`",
        "",
        "This report compares the four strategy structure required by the proposal: no hedge, static hedge, ML/rule-based hedge, and planning-based hedge.",
        "",
        "| Strategy | Proposal Role | Total Return % | Annual Return % | Volatility % | Sharpe | Sortino | Downside Dev % | Max Drawdown % | CVaR 95% % | Hedge Ratio Std | Avg Hedge Ratio | Turnover | Txn Cost % | Hedging Effectiveness |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summaries:
        lines.append(
            f"| {row['label']} | {row['proposal_strategy']} | {row['total_return_pct']} | "
            f"{row['annual_return_pct']} | {row['volatility_pct']} | {row['sharpe']} | "
            f"{row['sortino']} | {row['downside_deviation_pct']} | {row['max_drawdown_pct']} | "
            f"{row['cvar_95_pct']} | {row['hedge_ratio_std']} | {row['average_hedge_ratio']} | "
            f"{row['total_turnover']} | {row['total_transaction_cost_pct']} | {row['hedging_effectiveness']} |"
        )

    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "The no-hedge and static-50 strategies are direct baselines. The ML/rule-based strategy uses the GJR-GARCH volatility regime mapped into fixed hedge levels. The planning-based strategy applies the numerical planner to the dynamic scaled minimum-variance hedge target and converts it into executable futures contracts under margin, transaction-cost, expiry and re-hedging constraints.",
        "",
        "The strongest planning claim should focus on risk reduction and operational feasibility, not only return. The key metrics are volatility, maximum drawdown, downside deviation, CVaR, hedge-ratio stability, turnover, transaction cost, and hedging effectiveness.",
    ])

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Created", daily_path)
    print("Created", planner_log_path)
    print("Created", summary_path)
    print("Created", md_path)
    print("Created", metadata_path)
    print()
    print(summary_df.to_string(index=False))
    return summary_df


def main() -> None:
    combined = []
    for period in DEFAULT_PERIODS:
        print()
        print("=" * 100)
        print("RUNNING PERIOD:", period)
        print("=" * 100)
        result = run_for_period(period)
        combined.append(result)

    combined_df = pd.concat(combined, ignore_index=True)
    combined_path = REPORT_DIR / "proposal_four_strategy_summary_all_periods.csv"
    combined_md = REPORT_DIR / "proposal_four_strategy_summary_all_periods.md"
    combined_df.to_csv(combined_path, index=False)

    lines = [
        "# Proposal-Aligned Four Strategy Evaluation Across Periods",
        "",
        "This report summarises the four proposal strategies across the selection period and confirmatory period. The confirmatory period should be treated as the main out-of-sample result.",
        "",
        "| Period | Strategy | Proposal Role | Total Return % | Volatility % | Sharpe | Downside Dev % | Max Drawdown % | CVaR 95% % | Hedge Ratio Std | Turnover | Txn Cost % | Hedging Effectiveness |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in combined_df.iterrows():
        lines.append(
            f"| {row['period']} | {row['label']} | {row['proposal_strategy']} | {row['total_return_pct']} | "
            f"{row['volatility_pct']} | {row['sharpe']} | {row['downside_deviation_pct']} | "
            f"{row['max_drawdown_pct']} | {row['cvar_95_pct']} | {row['hedge_ratio_std']} | "
            f"{row['total_turnover']} | {row['total_transaction_cost_pct']} | {row['hedging_effectiveness']} |"
        )

    lines.extend([
        "",
        "## Note",
        "",
        "The confirmatory_2025_2026 period is the main final evaluation period. The selection_2020_2024 result is included as supplementary development-period evidence.",
    ])

    combined_md.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("Created", combined_path)
    print("Created", combined_md)


if __name__ == "__main__":
    main()
