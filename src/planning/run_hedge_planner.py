"""
Constraint-aware hedging planner using Unified Planning + ENHSP.

The broader proposal action "adjust hedge ratio" is refined into two explicit
actions for clearer explanations and audit logs:
    OPEN_HEDGE
    INCREASE_HEDGE
    REDUCE_HEDGE
    CLOSE_HEDGE
    HOLD_POSITION
    ROLL_FUTURES_FORWARD

"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from unified_planning.environment import get_environment
from unified_planning.io import PDDLWriter
from unified_planning.shortcuts import (
    And,
    BoolType,
    Equals,
    Fluent,
    GT,
    InstantaneousAction,
    IntType,
    LE,
    LT,
    Not,
    OneshotPlanner,
    Or,
    Problem,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "reports" / "planning"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PLANNER_NAME = "enhsp"
PLANNER_TIMEOUT_SECONDS = 5.0

# 100 hedge bps = hedge ratio 1.00
MAX_HEDGE_BPS = 150
NO_TRADE_BAND_BPS = 5
RISK_ADJUSTMENT_BPS = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}  # Regime cap is applied before planning.
DEFAULT_TRANSACTION_COST_BPS = 2.0
DEFAULT_MAX_TRADE_COST_MICROS = 50
DEMO_MAX_TRADE_COST_MICROS = 250
DEFAULT_ROLL_THRESHOLD_DAYS = 3


@dataclass(frozen=True)
class HedgePlanningInput:
    scenario_name: str
    current_hedge_ratio: float
    core_target_ratio: float
    predicted_volatility: float
    risk_regime: str
    margin_available: bool
    days_to_expiry: int
    position_open: bool | None = None
    rehedge_allowed: bool = True
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    max_trade_cost_micros: int = DEFAULT_MAX_TRADE_COST_MICROS
    max_hedge_ratio: float = MAX_HEDGE_BPS / 100
    roll_threshold_days: int = DEFAULT_ROLL_THRESHOLD_DAYS


@dataclass(frozen=True)
class PreparedPlanningState:
    scenario_name: str
    predicted_volatility: float
    risk_regime: str
    current_hedge_bps: int
    core_target_bps: int
    risk_adjustment_bps: int
    unconstrained_target_bps: int
    planner_target_bps: int
    position_open: bool
    margin_available: bool
    rehedge_allowed: bool
    adjustment_bps: int
    no_trade_band_bps: int
    estimated_trade_cost_micros: int
    max_trade_cost_micros: int
    days_to_expiry: int
    roll_threshold_days: int
    roll_required: bool
    max_hedge_bps: int


@dataclass
class PlanningResult:
    scenario_name: str
    planner_status: str
    used_fallback: bool
    action_sequence: list[str]
    primary_action: str
    explanation: list[str]
    prepared_state: dict[str, Any]
    pddl_domain_file: str
    pddl_problem_file: str


def ratio_to_bps(value: float) -> int:
    return int(round(float(value) * 100))


def bps_to_ratio(value: int) -> float:
    return round(int(value) / 100.0, 4)


def estimate_trade_cost_micros(adjustment_bps: int, transaction_cost_bps: float) -> int:
    adjustment_ratio = abs(int(adjustment_bps)) / 100.0
    cost_return = adjustment_ratio * (float(transaction_cost_bps) / 10_000.0)
    return int(round(cost_return * 1_000_000))


def prepare_state(planning_input: HedgePlanningInput) -> PreparedPlanningState:
    regime = planning_input.risk_regime.strip().upper()
    if regime not in RISK_ADJUSTMENT_BPS:
        raise ValueError(f"Unknown risk regime: {planning_input.risk_regime}")
    if planning_input.days_to_expiry < 0:
        raise ValueError("days_to_expiry cannot be negative")

    max_bps = max(0, ratio_to_bps(planning_input.max_hedge_ratio))
    current_bps = max(0, min(ratio_to_bps(planning_input.current_hedge_ratio), max_bps))
    core_bps = max(0, ratio_to_bps(planning_input.core_target_ratio))
    risk_bps = RISK_ADJUSTMENT_BPS[regime]
    unconstrained_target = core_bps + risk_bps
    limited_target = max(0, min(unconstrained_target, max_bps))

    raw_adjustment = limited_target - current_bps
    planner_target = current_bps if abs(raw_adjustment) < NO_TRADE_BAND_BPS else limited_target
    adjustment_bps = planner_target - current_bps

    position_open = current_bps > 0 if planning_input.position_open is None else bool(planning_input.position_open)
    cost_micros = estimate_trade_cost_micros(adjustment_bps, planning_input.transaction_cost_bps)
    roll_required = position_open and planning_input.days_to_expiry <= planning_input.roll_threshold_days

    return PreparedPlanningState(
        scenario_name=planning_input.scenario_name,
        predicted_volatility=float(planning_input.predicted_volatility),
        risk_regime=regime,
        current_hedge_bps=current_bps,
        core_target_bps=core_bps,
        risk_adjustment_bps=risk_bps,
        unconstrained_target_bps=unconstrained_target,
        planner_target_bps=planner_target,
        position_open=position_open,
        margin_available=bool(planning_input.margin_available),
        rehedge_allowed=bool(planning_input.rehedge_allowed),
        adjustment_bps=adjustment_bps,
        no_trade_band_bps=NO_TRADE_BAND_BPS,
        estimated_trade_cost_micros=cost_micros,
        max_trade_cost_micros=int(planning_input.max_trade_cost_micros),
        days_to_expiry=int(planning_input.days_to_expiry),
        roll_threshold_days=int(planning_input.roll_threshold_days),
        roll_required=roll_required,
        max_hedge_bps=max_bps,
    )


def build_problem(state: PreparedPlanningState) -> Problem:
    problem = Problem(f"hedge_plan_{state.scenario_name}")

    position_open = Fluent("position_open", BoolType())
    margin_available = Fluent("margin_available", BoolType())
    rehedge_allowed = Fluent("rehedge_allowed", BoolType())
    roll_required = Fluent("roll_required", BoolType())
    roll_completed = Fluent("roll_completed", BoolType())
    decision_completed = Fluent("decision_completed", BoolType())
    hold_allowed = Fluent("hold_allowed", BoolType())

    current_bps = Fluent("current_hedge_bps", IntType())
    target_bps = Fluent("target_hedge_bps", IntType())
    max_bps = Fluent("max_hedge_bps", IntType())
    trade_cost = Fluent("estimated_trade_cost_micros", IntType())
    max_trade_cost = Fluent("max_trade_cost_micros", IntType())

    for fluent, default in [
        (position_open, False), (margin_available, False), (rehedge_allowed, False),
        (roll_required, False), (roll_completed, False), (decision_completed, False),
        (hold_allowed, False),
        (current_bps, 0), (target_bps, 0), (max_bps, MAX_HEDGE_BPS),
        (trade_cost, 0), (max_trade_cost, DEFAULT_MAX_TRADE_COST_MICROS),
    ]:
        problem.add_fluent(fluent, default_initial_value=default)

    values = {
        position_open: state.position_open,
        margin_available: state.margin_available,
        rehedge_allowed: state.rehedge_allowed,
        roll_required: state.roll_required,
        roll_completed: False,
        decision_completed: False,
        hold_allowed: (
            state.adjustment_bps == 0
            or not state.rehedge_allowed
            or state.estimated_trade_cost_micros > state.max_trade_cost_micros
            or (state.adjustment_bps > 0 and not state.margin_available)
        ),
        current_bps: state.current_hedge_bps,
        target_bps: state.planner_target_bps,
        max_bps: state.max_hedge_bps,
        trade_cost: state.estimated_trade_cost_micros,
        max_trade_cost: state.max_trade_cost_micros,
    }
    for fluent, value in values.items():
        problem.set_initial_value(fluent, value)

    cost_allowed = LE(trade_cost, max_trade_cost)
    target_allowed = LE(target_bps, max_bps)
    roll_gate = Or(Not(roll_required), roll_completed)

    open_hedge = InstantaneousAction("open_hedge")
    for condition in [Not(position_open), GT(target_bps, 0), margin_available, rehedge_allowed, cost_allowed, target_allowed]:
        open_hedge.add_precondition(condition)
    open_hedge.add_effect(position_open, True)
    open_hedge.add_effect(current_bps, target_bps)
    open_hedge.add_effect(decision_completed, True)
    problem.add_action(open_hedge)

    increase_hedge = InstantaneousAction("increase_hedge")
    for condition in [position_open, roll_gate, GT(target_bps, current_bps), margin_available, rehedge_allowed, cost_allowed, target_allowed]:
        increase_hedge.add_precondition(condition)
    increase_hedge.add_effect(current_bps, target_bps)
    increase_hedge.add_effect(decision_completed, True)
    problem.add_action(increase_hedge)

    reduce_hedge = InstantaneousAction("reduce_hedge")
    for condition in [position_open, roll_gate, GT(target_bps, 0), LT(target_bps, current_bps), rehedge_allowed, cost_allowed]:
        reduce_hedge.add_precondition(condition)
    reduce_hedge.add_effect(current_bps, target_bps)
    reduce_hedge.add_effect(decision_completed, True)
    problem.add_action(reduce_hedge)

    close_hedge = InstantaneousAction("close_hedge")
    for condition in [position_open, Equals(target_bps, 0), rehedge_allowed, cost_allowed]:
        close_hedge.add_precondition(condition)
    close_hedge.add_effect(position_open, False)
    close_hedge.add_effect(current_bps, 0)
    close_hedge.add_effect(decision_completed, True)
    problem.add_action(close_hedge)

    hold_position = InstantaneousAction("hold_position")
    hold_position.add_precondition(hold_allowed)
    hold_position.add_effect(decision_completed, True)
    problem.add_action(hold_position)

    roll_forward = InstantaneousAction("roll_futures_forward")
    for condition in [position_open, roll_required, Not(roll_completed)]:
        roll_forward.add_precondition(condition)
    roll_forward.add_effect(roll_completed, True)
    problem.add_action(roll_forward)

    problem.add_goal(decision_completed)
    problem.add_goal(Or(Not(position_open), Not(roll_required), roll_completed))
    return problem


def export_pddl(problem: Problem, scenario_name: str) -> tuple[Path, Path]:
    scenario_dir = REPORT_DIR / "pddl" / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    domain_file = scenario_dir / "domain.pddl"
    problem_file = scenario_dir / "problem.pddl"
    writer = PDDLWriter(problem)
    writer.write_domain(str(domain_file))
    writer.write_problem(str(problem_file))
    return domain_file, problem_file


def fallback_actions(state: PreparedPlanningState) -> list[str]:
    actions: list[str] = []
    if state.roll_required:
        actions.append("ROLL_FUTURES_FORWARD")

    blocked = (
        not state.rehedge_allowed
        or state.estimated_trade_cost_micros > state.max_trade_cost_micros
        or (state.adjustment_bps > 0 and not state.margin_available)
        or state.adjustment_bps == 0
    )
    if blocked:
        actions.append("HOLD_POSITION")
    elif not state.position_open and state.planner_target_bps > 0:
        actions.append("OPEN_HEDGE")
    elif state.position_open and state.planner_target_bps == 0:
        actions.append("CLOSE_HEDGE")
    elif state.adjustment_bps > 0:
        actions.append("INCREASE_HEDGE")
    elif state.adjustment_bps < 0:
        actions.append("REDUCE_HEDGE")
    else:
        actions.append("HOLD_POSITION")
    return actions


def explain(state: PreparedPlanningState, actions: list[str], used_fallback: bool) -> list[str]:
    reasons = [
        f"Risk regime={state.risk_regime}; predicted next-day volatility={state.predicted_volatility:.6f}.",
        f"Planner received risk-gated target ratio={bps_to_ratio(state.core_target_bps):.2f}; "
        f"extra planner adjustment={state.risk_adjustment_bps:+d} hedge bps.",
        f"Current ratio={bps_to_ratio(state.current_hedge_bps):.2f}; planner target={bps_to_ratio(state.planner_target_bps):.2f}; requested change={state.adjustment_bps:+d} hedge bps.",
        f"Estimated turnover cost={state.estimated_trade_cost_micros} micro-units; configured limit={state.max_trade_cost_micros} micro-units.",
    ]
    if state.roll_required:
        reasons.append(f"Contract expires in {state.days_to_expiry} day(s), so the open futures position must be rolled.")
    if not state.rehedge_allowed:
        reasons.append("Rehedging is blocked by the configured frequency rule.")
    if state.adjustment_bps > 0 and not state.margin_available:
        reasons.append("The requested increase is held because additional margin is unavailable.")
    if state.estimated_trade_cost_micros > state.max_trade_cost_micros:
        reasons.append("The requested trade is held because estimated transaction cost exceeds the configured limit.")
    if used_fallback:
        reasons.append("The deterministic fallback was used because ENHSP did not return a usable plan.")
    reasons.append("Planner action sequence: " + " -> ".join(actions) + ".")
    return reasons


def solve_planning_input(planning_input: HedgePlanningInput) -> PlanningResult:
    state = prepare_state(planning_input)
    problem = build_problem(state)
    domain_file, problem_file = export_pddl(problem, state.scenario_name)
    get_environment().credits_stream = None

    status = "UNKNOWN"
    actions: list[str] = []
    used_fallback = False
    try:
        with OneshotPlanner(name=PLANNER_NAME) as planner:
            result = planner.solve(problem, timeout=PLANNER_TIMEOUT_SECONDS)
        status = str(result.status)
        if result.plan is not None:
            actions = [str(instance.action.name).upper() for instance in result.plan.actions]
    except Exception as exc:
        status = f"ERROR: {type(exc).__name__}: {exc}"

    if not actions:
        used_fallback = True
        actions = fallback_actions(state)

    return PlanningResult(
        scenario_name=state.scenario_name,
        planner_status=status,
        used_fallback=used_fallback,
        action_sequence=actions,
        primary_action=actions[-1],
        explanation=explain(state, actions, used_fallback),
        prepared_state=asdict(state),
        pddl_domain_file=str(domain_file.relative_to(ROOT)),
        pddl_problem_file=str(problem_file.relative_to(ROOT)),
    )


def demo_scenarios() -> list[HedgePlanningInput]:
    return [
        HedgePlanningInput(
            scenario_name="demo_open_hedge",
            current_hedge_ratio=0.00,
            core_target_ratio=0.80,
            predicted_volatility=0.22,
            risk_regime="HIGH",
            margin_available=True,
            days_to_expiry=12,
            max_trade_cost_micros=DEMO_MAX_TRADE_COST_MICROS,
        ),
        HedgePlanningInput(
            scenario_name="demo_increase_hedge",
            current_hedge_ratio=0.70,
            core_target_ratio=0.90,
            predicted_volatility=0.19,
            risk_regime="HIGH",
            margin_available=True,
            days_to_expiry=10,
            max_trade_cost_micros=DEMO_MAX_TRADE_COST_MICROS,
        ),
        HedgePlanningInput(
            scenario_name="demo_reduce_hedge",
            current_hedge_ratio=1.00,
            core_target_ratio=0.75,
            predicted_volatility=0.08,
            risk_regime="LOW",
            margin_available=True,
            days_to_expiry=9,
            max_trade_cost_micros=DEMO_MAX_TRADE_COST_MICROS,
        ),
        HedgePlanningInput(
            scenario_name="demo_close_hedge",
            current_hedge_ratio=0.50,
            core_target_ratio=0.05,
            predicted_volatility=0.05,
            risk_regime="LOW",
            margin_available=True,
            days_to_expiry=11,
            max_trade_cost_micros=DEMO_MAX_TRADE_COST_MICROS,
        ),
        HedgePlanningInput(
            scenario_name="demo_hold_position",
            current_hedge_ratio=0.85,
            core_target_ratio=0.83,
            predicted_volatility=0.12,
            risk_regime="MEDIUM",
            margin_available=True,
            days_to_expiry=15,
            max_trade_cost_micros=DEMO_MAX_TRADE_COST_MICROS,
        ),
        HedgePlanningInput(
            scenario_name="demo_roll_and_hold",
            current_hedge_ratio=0.90,
            core_target_ratio=0.90,
            predicted_volatility=0.13,
            risk_regime="MEDIUM",
            margin_available=True,
            days_to_expiry=2,
            max_trade_cost_micros=DEMO_MAX_TRADE_COST_MICROS,
        ),
    ]

def print_result(result: PlanningResult) -> None:
    state = result.prepared_state
    print("\n" + "=" * 78)
    print(f"Scenario       : {result.scenario_name}")
    print(f"Planner status : {result.planner_status}")
    print(f"Used fallback  : {result.used_fallback}")
    print(f"Risk regime    : {state['risk_regime']}")
    print(
        "Hedge ratio    : "
        f"current={bps_to_ratio(state['current_hedge_bps']):.2f} | "
        f"core={bps_to_ratio(state['core_target_bps']):.2f} | "
        f"target={bps_to_ratio(state['planner_target_bps']):.2f}"
    )
    print("Action plan    : " + " -> ".join(result.action_sequence))
    print("Primary action : " + result.primary_action)
    print("Explanation:")
    for reason in result.explanation:
        print(f"  - {reason}")


def write_results(results: list[PlanningResult], filename: str) -> Path:
    output_file = REPORT_DIR / filename
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump([asdict(result) for result in results], file, indent=2)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hedra hedge planner.")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--scenario-name", default="custom_scenario")
    parser.add_argument("--current-hedge-ratio", type=float, default=0.70)
    parser.add_argument("--core-target-ratio", type=float, default=0.90)
    parser.add_argument("--predicted-volatility", type=float, default=0.18)
    parser.add_argument("--risk-regime", choices=["LOW", "MEDIUM", "HIGH"], default="HIGH")
    parser.add_argument("--margin-available", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rehedge-allowed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--days-to-expiry", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        inputs = demo_scenarios()
        filename = "planning_demo_results.json"
    else:
        inputs = [
            HedgePlanningInput(
                scenario_name=args.scenario_name,
                current_hedge_ratio=args.current_hedge_ratio,
                core_target_ratio=args.core_target_ratio,
                predicted_volatility=args.predicted_volatility,
                risk_regime=args.risk_regime,
                margin_available=args.margin_available,
                rehedge_allowed=args.rehedge_allowed,
                days_to_expiry=args.days_to_expiry,
            )
        ]
        filename = "planning_latest_result.json"

    results = [solve_planning_input(item) for item in inputs]
    for result in results:
        print_result(result)
    output_file = write_results(results, filename)
    print("\nSaved planner audit output:")
    print(f"  {output_file}")


if __name__ == "__main__":
    main()
