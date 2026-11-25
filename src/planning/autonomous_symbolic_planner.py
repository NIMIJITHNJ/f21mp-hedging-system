from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from unified_planning.environment import get_environment
from unified_planning.io import PDDLWriter
from unified_planning.shortcuts import (
    BoolType,
    Fluent,
    InstantaneousAction,
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


@dataclass
class AutonomousSymbolicPlannerInput:
    scenario_name: str
    portfolio_value: float
    candidate_hedge_ratio: float
    predicted_volatility: float
    risk_regime: str
    futures_price: float
    lot_size: int
    current_short_contracts: int
    margin_per_contract: float
    available_margin: float
    days_to_expiry: int
    rehedge_allowed: bool = True
    transaction_cost_bps: float = 2.0
    max_trade_cost_micros: int = 5000
    roll_threshold_days: int = 3
    target_hedge_ratio: float | None = None


@dataclass
class AutonomousSymbolicPreparedState:
    scenario_name: str
    predicted_volatility: float
    risk_regime: str
    candidate_hedge_ratio: float
    target_hedge_ratio: float
    current_contracts: int
    target_contracts: int
    contract_notional: float
    additional_margin_required: float
    estimated_trade_cost_micros: int
    position_open: bool
    hedge_needed: bool
    target_zero: bool
    target_above_current: bool
    target_below_current: bool
    margin_available: bool
    rehedge_allowed: bool
    cost_allowed: bool
    roll_required: bool
    hold_allowed: bool


@dataclass
class AutonomousSymbolicPlannerResult:
    scenario_name: str
    planner_status: str
    used_fallback: bool
    action_sequence: list[str]
    primary_action: str
    planning_time_ms: float
    prepared_state: dict[str, Any]
    pddl_domain_file: str
    pddl_problem_file: str
    explanation: list[str]


def risk_gated_target_ratio(candidate_ratio: float, risk_regime: str) -> float:
    """
    Apply the same hedge cap used in the dashboard.

    """
    candidate_ratio = max(0.0, float(candidate_ratio))
    regime = str(risk_regime).strip().lower()

    if regime == "low":
        return 0.0
    if regime == "medium":
        return min(candidate_ratio, 0.50)
    if regime == "high":
        return min(candidate_ratio, 0.80)

    return min(candidate_ratio, 0.50)


def round_contracts(raw_contracts: float) -> int:
    if raw_contracts <= 0:
        return 0
    rounded = int(round(raw_contracts))
    return max(1, rounded)


def prepare_state(item: AutonomousSymbolicPlannerInput) -> AutonomousSymbolicPreparedState:
    contract_notional = item.futures_price * item.lot_size

    target_ratio = (
        item.target_hedge_ratio
        if item.target_hedge_ratio is not None
        else risk_gated_target_ratio(item.candidate_hedge_ratio, item.risk_regime)
    )

    target_ratio = max(0.0, float(target_ratio))
    target_notional = item.portfolio_value * target_ratio
    target_contracts = round_contracts(target_notional / contract_notional)

    current_contracts = int(item.current_short_contracts)
    contract_delta = target_contracts - current_contracts

    additional_margin_required = max(contract_delta, 0) * item.margin_per_contract
    margin_ok = item.available_margin >= additional_margin_required

    estimated_trade_cost_micros = int(
        round(abs(contract_delta) * item.transaction_cost_bps * 100)
    )
    cost_ok = estimated_trade_cost_micros <= item.max_trade_cost_micros

    position_open = current_contracts > 0
    hedge_needed = target_contracts > 0
    target_zero = target_contracts == 0
    target_above_current = target_contracts > current_contracts
    target_below_current = target_contracts < current_contracts
    roll_required = position_open and item.days_to_expiry <= item.roll_threshold_days

    hold_allowed = (
        target_contracts == current_contracts
        or not item.rehedge_allowed
        or not cost_ok
        or (target_above_current and not margin_ok)
    )

    return AutonomousSymbolicPreparedState(
        scenario_name=item.scenario_name,
        predicted_volatility=item.predicted_volatility,
        risk_regime=item.risk_regime,
        candidate_hedge_ratio=item.candidate_hedge_ratio,
        target_hedge_ratio=target_ratio,
        current_contracts=current_contracts,
        target_contracts=target_contracts,
        contract_notional=contract_notional,
        additional_margin_required=additional_margin_required,
        estimated_trade_cost_micros=estimated_trade_cost_micros,
        position_open=position_open,
        hedge_needed=hedge_needed,
        target_zero=target_zero,
        target_above_current=target_above_current,
        target_below_current=target_below_current,
        margin_available=margin_ok,
        rehedge_allowed=item.rehedge_allowed,
        cost_allowed=cost_ok,
        roll_required=roll_required,
        hold_allowed=hold_allowed,
    )


def build_problem(state: AutonomousSymbolicPreparedState) -> Problem:
    problem = Problem(f"autonomous_symbolic_hedge_plan_{state.scenario_name}")

    position_open = Fluent("position_open", BoolType())
    hedge_needed = Fluent("hedge_needed", BoolType())
    target_zero = Fluent("target_zero", BoolType())
    target_above_current = Fluent("target_above_current", BoolType())
    target_below_current = Fluent("target_below_current", BoolType())
    margin_available = Fluent("margin_available", BoolType())
    rehedge_allowed = Fluent("rehedge_allowed", BoolType())
    cost_allowed = Fluent("cost_allowed", BoolType())
    roll_required = Fluent("roll_required", BoolType())
    roll_completed = Fluent("roll_completed", BoolType())
    hold_allowed = Fluent("hold_allowed", BoolType())
    decision_completed = Fluent("decision_completed", BoolType())

    fluents = [
        position_open,
        hedge_needed,
        target_zero,
        target_above_current,
        target_below_current,
        margin_available,
        rehedge_allowed,
        cost_allowed,
        roll_required,
        roll_completed,
        hold_allowed,
        decision_completed,
    ]

    for fluent in fluents:
        problem.add_fluent(fluent, default_initial_value=False)

    problem.set_initial_value(position_open, state.position_open)
    problem.set_initial_value(hedge_needed, state.hedge_needed)
    problem.set_initial_value(target_zero, state.target_zero)
    problem.set_initial_value(target_above_current, state.target_above_current)
    problem.set_initial_value(target_below_current, state.target_below_current)
    problem.set_initial_value(margin_available, state.margin_available)
    problem.set_initial_value(rehedge_allowed, state.rehedge_allowed)
    problem.set_initial_value(cost_allowed, state.cost_allowed)
    problem.set_initial_value(roll_required, state.roll_required)
    problem.set_initial_value(hold_allowed, state.hold_allowed)

    roll_gate = Or(Not(roll_required), roll_completed)

    open_hedge = InstantaneousAction("open_hedge")
    for condition in [
        Not(position_open),
        hedge_needed,
        margin_available,
        rehedge_allowed,
        cost_allowed,
    ]:
        open_hedge.add_precondition(condition)
    open_hedge.add_effect(position_open, True)
    open_hedge.add_effect(decision_completed, True)
    problem.add_action(open_hedge)

    increase_hedge = InstantaneousAction("increase_hedge")
    for condition in [
        position_open,
        roll_gate,
        target_above_current,
        margin_available,
        rehedge_allowed,
        cost_allowed,
    ]:
        increase_hedge.add_precondition(condition)
    increase_hedge.add_effect(decision_completed, True)
    problem.add_action(increase_hedge)

    reduce_hedge = InstantaneousAction("reduce_hedge")
    for condition in [
        position_open,
        roll_gate,
        target_below_current,
        Not(target_zero),
        rehedge_allowed,
        cost_allowed,
    ]:
        reduce_hedge.add_precondition(condition)
    reduce_hedge.add_effect(decision_completed, True)
    problem.add_action(reduce_hedge)

    close_hedge = InstantaneousAction("close_hedge")
    for condition in [
        position_open,
        roll_gate,
        target_zero,
        rehedge_allowed,
        cost_allowed,
    ]:
        close_hedge.add_precondition(condition)
    close_hedge.add_effect(position_open, False)
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
    scenario_dir = REPORT_DIR / "pddl_autonomous_symbolic" / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    domain_file = scenario_dir / "domain.pddl"
    problem_file = scenario_dir / "problem.pddl"

    writer = PDDLWriter(problem)
    writer.write_domain(str(domain_file))
    writer.write_problem(str(problem_file))

    return domain_file, problem_file


def fallback_actions(state: AutonomousSymbolicPreparedState) -> list[str]:
    actions: list[str] = []

    if state.roll_required:
        actions.append("ROLL_FUTURES_FORWARD")

    if state.hold_allowed:
        actions.append("HOLD_POSITION")
    elif not state.position_open and state.hedge_needed:
        actions.append("OPEN_HEDGE")
    elif state.position_open and state.target_zero:
        actions.append("CLOSE_HEDGE")
    elif state.position_open and state.target_above_current:
        actions.append("INCREASE_HEDGE")
    elif state.position_open and state.target_below_current:
        actions.append("REDUCE_HEDGE")
    else:
        actions.append("HOLD_POSITION")

    return actions


def explain(
    state: AutonomousSymbolicPreparedState,
    actions: list[str],
    used_fallback: bool,
) -> list[str]:
    reasons = [
        f"Risk regime was {state.risk_regime} with predicted volatility {state.predicted_volatility:.4f}.",
        f"Candidate hedge ratio was {state.candidate_hedge_ratio:.2%}; generated target hedge ratio was {state.target_hedge_ratio:.2%}.",
        f"Current contracts were {state.current_contracts}; target contracts were {state.target_contracts}.",
        f"Additional margin required was {state.additional_margin_required:,.2f}.",
    ]

    if state.roll_required:
        reasons.append("Roll condition was active because the contract was close to expiry.")
    if not state.margin_available:
        reasons.append("Margin was insufficient for increasing the hedge.")
    if not state.cost_allowed:
        reasons.append("Estimated transaction cost exceeded the configured limit.")
    if used_fallback:
        reasons.append("Fallback action logic was used because ENHSP did not return a usable plan.")

    reasons.append("Planner action sequence: " + " -> ".join(actions) + ".")
    return reasons


def solve_autonomous_symbolic_planning(
    item: AutonomousSymbolicPlannerInput,
) -> AutonomousSymbolicPlannerResult:
    start = perf_counter()
    state = prepare_state(item)
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
        actions = fallback_actions(state)
        used_fallback = True

    elapsed_ms = (perf_counter() - start) * 1000

    return AutonomousSymbolicPlannerResult(
        scenario_name=item.scenario_name,
        planner_status=status,
        used_fallback=used_fallback,
        action_sequence=actions,
        primary_action=actions[-1],
        planning_time_ms=round(elapsed_ms, 3),
        prepared_state=asdict(state),
        pddl_domain_file=str(domain_file.relative_to(ROOT)),
        pddl_problem_file=str(problem_file.relative_to(ROOT)),
        explanation=explain(state, actions, used_fallback),
    )


def demo_scenarios() -> list[AutonomousSymbolicPlannerInput]:
    portfolio_value = 8_000_000.0
    futures_price = 24_000.0
    lot_size = 65
    margin_per_contract = futures_price * lot_size * 0.12
    available_margin = 2_000_000.0

    return [
        AutonomousSymbolicPlannerInput(
            scenario_name="auto_low_risk_no_position",
            portfolio_value=portfolio_value,
            candidate_hedge_ratio=0.30,
            predicted_volatility=0.08,
            risk_regime="Low",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=0,
            margin_per_contract=margin_per_contract,
            available_margin=available_margin,
            days_to_expiry=20,
        ),
        AutonomousSymbolicPlannerInput(
            scenario_name="auto_medium_risk_open_hedge",
            portfolio_value=portfolio_value,
            candidate_hedge_ratio=0.60,
            predicted_volatility=0.12,
            risk_regime="Medium",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=0,
            margin_per_contract=margin_per_contract,
            available_margin=available_margin,
            days_to_expiry=18,
        ),
        AutonomousSymbolicPlannerInput(
            scenario_name="auto_high_risk_increase_hedge",
            portfolio_value=portfolio_value,
            candidate_hedge_ratio=0.90,
            predicted_volatility=0.18,
            risk_regime="High",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=2,
            margin_per_contract=margin_per_contract,
            available_margin=available_margin,
            days_to_expiry=15,
        ),
        AutonomousSymbolicPlannerInput(
            scenario_name="auto_close_hedge",
            portfolio_value=portfolio_value,
            candidate_hedge_ratio=0.00,
            predicted_volatility=0.07,
            risk_regime="Low",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=3,
            margin_per_contract=margin_per_contract,
            available_margin=available_margin,
            days_to_expiry=22,
            target_hedge_ratio=0.00,
        ),
        AutonomousSymbolicPlannerInput(
            scenario_name="auto_roll_required",
            portfolio_value=portfolio_value,
            candidate_hedge_ratio=0.50,
            predicted_volatility=0.13,
            risk_regime="Medium",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=3,
            margin_per_contract=margin_per_contract,
            available_margin=available_margin,
            days_to_expiry=2,
            target_hedge_ratio=0.50,
        ),
    ]


def main() -> None:
    for item in demo_scenarios():
        result = solve_autonomous_symbolic_planning(item)
        state = result.prepared_state
        print(
            result.scenario_name,
            "| action:", result.primary_action,
            "| sequence:", " -> ".join(result.action_sequence),
            "| status:", result.planner_status,
            "| fallback:", result.used_fallback,
            "| target contracts:", state["target_contracts"],
            "| time ms:", result.planning_time_ms,
        )


if __name__ == "__main__":
    main()