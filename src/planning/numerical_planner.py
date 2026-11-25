from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from unified_planning.environment import get_environment
from unified_planning.io import PDDLWriter
from unified_planning.shortcuts import (
    BoolType,
    Equals,
    Fluent,
    GT,
    IntType,
    LE,
    Not,
    OneshotPlanner,
    Or,
    Problem,
    InstantaneousAction,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "reports" / "planning"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PLANNER_NAME = "enhsp"
PLANNER_TIMEOUT_SECONDS = 5.0


@dataclass
class NumericalPlannerInput:
    scenario_name: str
    portfolio_value: float
    target_hedge_ratio: float
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


@dataclass
class NumericalPreparedState:
    scenario_name: str
    predicted_volatility: float
    risk_regime: str
    target_hedge_ratio: float
    current_contracts: int
    target_contracts: int
    executable_contracts: int
    hedge_gap_contracts: int
    contract_notional: int
    margin_available_amount: int
    margin_required_amount: int
    transaction_cost_micros: int
    max_transaction_cost_micros: int
    days_to_expiry: int
    position_open: bool
    rehedge_allowed: bool
    roll_required: bool
    hold_allowed: bool


@dataclass
class NumericalPlannerResult:
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


def round_contracts(raw_contracts: float) -> int:
    if raw_contracts <= 0:
        return 0
    return max(1, int(round(raw_contracts)))


def prepare_state(item: NumericalPlannerInput) -> NumericalPreparedState:
    contract_notional = int(round(item.futures_price * item.lot_size))
    target_notional = item.portfolio_value * max(0.0, item.target_hedge_ratio)
    target_contracts = round_contracts(target_notional / contract_notional)

    current_contracts = int(item.current_short_contracts)
    contract_delta = target_contracts - current_contracts

    margin_required = int(round(max(contract_delta, 0) * item.margin_per_contract))
    margin_available = int(round(item.available_margin))

    transaction_cost_micros = int(round(abs(contract_delta) * item.transaction_cost_bps * 100))

    margin_ok = margin_available >= margin_required
    cost_ok = transaction_cost_micros <= item.max_trade_cost_micros

    position_open = current_contracts > 0
    roll_required = position_open and item.days_to_expiry <= item.roll_threshold_days

    hold_allowed = (
        target_contracts == current_contracts
        or not item.rehedge_allowed
        or not cost_ok
        or (target_contracts > current_contracts and not margin_ok)
    )

    if hold_allowed:
        executable_contracts = current_contracts
    else:
        executable_contracts = target_contracts

    hedge_gap_contracts = target_contracts - executable_contracts

    return NumericalPreparedState(
        scenario_name=item.scenario_name,
        predicted_volatility=float(item.predicted_volatility),
        risk_regime=str(item.risk_regime),
        target_hedge_ratio=float(item.target_hedge_ratio),
        current_contracts=current_contracts,
        target_contracts=target_contracts,
        executable_contracts=executable_contracts,
        hedge_gap_contracts=hedge_gap_contracts,
        contract_notional=contract_notional,
        margin_available_amount=margin_available,
        margin_required_amount=margin_required,
        transaction_cost_micros=transaction_cost_micros,
        max_transaction_cost_micros=int(item.max_trade_cost_micros),
        days_to_expiry=int(item.days_to_expiry),
        position_open=position_open,
        rehedge_allowed=bool(item.rehedge_allowed),
        roll_required=roll_required,
        hold_allowed=hold_allowed,
    )


def build_problem(state: NumericalPreparedState) -> Problem:
    problem = Problem(f"numerical_hedge_plan_{state.scenario_name}")

    position_open = Fluent("position_open", BoolType())
    rehedge_allowed = Fluent("rehedge_allowed", BoolType())
    roll_required = Fluent("roll_required", BoolType())
    roll_completed = Fluent("roll_completed", BoolType())
    hold_allowed = Fluent("hold_allowed", BoolType())
    decision_completed = Fluent("decision_completed", BoolType())

    current_contracts = Fluent("current_contracts", IntType())
    target_contracts = Fluent("target_contracts", IntType())
    executable_contracts = Fluent("executable_contracts", IntType())
    margin_available_amount = Fluent("margin_available_amount", IntType())
    margin_required_amount = Fluent("margin_required_amount", IntType())
    transaction_cost_micros = Fluent("transaction_cost_micros", IntType())
    max_transaction_cost_micros = Fluent("max_transaction_cost_micros", IntType())
    days_to_expiry = Fluent("days_to_expiry", IntType())

    for fluent, default in [
        (position_open, False),
        (rehedge_allowed, False),
        (roll_required, False),
        (roll_completed, False),
        (hold_allowed, False),
        (decision_completed, False),
        (current_contracts, 0),
        (target_contracts, 0),
        (executable_contracts, 0),
        (margin_available_amount, 0),
        (margin_required_amount, 0),
        (transaction_cost_micros, 0),
        (max_transaction_cost_micros, 0),
        (days_to_expiry, 0),
    ]:
        problem.add_fluent(fluent, default_initial_value=default)

    problem.set_initial_value(position_open, state.position_open)
    problem.set_initial_value(rehedge_allowed, state.rehedge_allowed)
    problem.set_initial_value(roll_required, state.roll_required)
    problem.set_initial_value(hold_allowed, state.hold_allowed)
    problem.set_initial_value(current_contracts, state.current_contracts)
    problem.set_initial_value(target_contracts, state.target_contracts)
    problem.set_initial_value(executable_contracts, state.executable_contracts)
    problem.set_initial_value(margin_available_amount, state.margin_available_amount)
    problem.set_initial_value(margin_required_amount, state.margin_required_amount)
    problem.set_initial_value(transaction_cost_micros, state.transaction_cost_micros)
    problem.set_initial_value(max_transaction_cost_micros, state.max_transaction_cost_micros)
    problem.set_initial_value(days_to_expiry, state.days_to_expiry)

    margin_ok = LE(margin_required_amount, margin_available_amount)
    cost_ok = LE(transaction_cost_micros, max_transaction_cost_micros)
    roll_gate = Or(Not(roll_required), roll_completed)

    open_hedge = InstantaneousAction("open_hedge")
    for condition in [
        Not(position_open),
        GT(target_contracts, 0),
        margin_ok,
        rehedge_allowed,
        cost_ok,
    ]:
        open_hedge.add_precondition(condition)
    open_hedge.add_effect(position_open, True)
    open_hedge.add_effect(current_contracts, target_contracts)
    open_hedge.add_effect(executable_contracts, target_contracts)
    open_hedge.add_effect(decision_completed, True)
    problem.add_action(open_hedge)

    increase_hedge = InstantaneousAction("increase_hedge")
    for condition in [
        position_open,
        roll_gate,
        GT(target_contracts, current_contracts),
        margin_ok,
        rehedge_allowed,
        cost_ok,
    ]:
        increase_hedge.add_precondition(condition)
    increase_hedge.add_effect(current_contracts, target_contracts)
    increase_hedge.add_effect(executable_contracts, target_contracts)
    increase_hedge.add_effect(decision_completed, True)
    problem.add_action(increase_hedge)

    reduce_hedge = InstantaneousAction("reduce_hedge")
    for condition in [
        position_open,
        roll_gate,
        GT(current_contracts, target_contracts),
        GT(target_contracts, 0),
        rehedge_allowed,
        cost_ok,
    ]:
        reduce_hedge.add_precondition(condition)
    reduce_hedge.add_effect(current_contracts, target_contracts)
    reduce_hedge.add_effect(executable_contracts, target_contracts)
    reduce_hedge.add_effect(decision_completed, True)
    problem.add_action(reduce_hedge)

    close_hedge = InstantaneousAction("close_hedge")
    for condition in [
        position_open,
        roll_gate,
        Equals(target_contracts, 0),
        rehedge_allowed,
        cost_ok,
    ]:
        close_hedge.add_precondition(condition)
    close_hedge.add_effect(position_open, False)
    close_hedge.add_effect(current_contracts, 0)
    close_hedge.add_effect(executable_contracts, 0)
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
    scenario_dir = REPORT_DIR / "pddl_numerical" / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    domain_file = scenario_dir / "domain.pddl"
    problem_file = scenario_dir / "problem.pddl"

    writer = PDDLWriter(problem)
    writer.write_domain(str(domain_file))
    writer.write_problem(str(problem_file))

    return domain_file, problem_file


def fallback_actions(state: NumericalPreparedState) -> list[str]:
    actions: list[str] = []

    if state.roll_required:
        actions.append("ROLL_FUTURES_FORWARD")

    if state.hold_allowed:
        actions.append("HOLD_POSITION")
    elif not state.position_open and state.target_contracts > 0:
        actions.append("OPEN_HEDGE")
    elif state.position_open and state.target_contracts == 0:
        actions.append("CLOSE_HEDGE")
    elif state.position_open and state.target_contracts > state.current_contracts:
        actions.append("INCREASE_HEDGE")
    elif state.position_open and state.target_contracts < state.current_contracts:
        actions.append("REDUCE_HEDGE")
    else:
        actions.append("HOLD_POSITION")

    return actions


def explain(state: NumericalPreparedState, actions: list[str], used_fallback: bool) -> list[str]:
    reasons = [
        f"Risk regime was {state.risk_regime} with predicted volatility {state.predicted_volatility:.4f}.",
        f"Current contracts={state.current_contracts}; target contracts={state.target_contracts}; executable contracts={state.executable_contracts}.",
        f"Hedge gap after constraints={state.hedge_gap_contracts} contracts.",
        f"Margin required={state.margin_required_amount}; margin available={state.margin_available_amount}.",
        f"Transaction cost={state.transaction_cost_micros}; cost limit={state.max_transaction_cost_micros}.",
    ]

    if state.roll_required:
        reasons.append("Roll condition was active because the contract was close to expiry.")
    if state.hold_allowed:
        reasons.append("The planner was allowed to hold because the target was already met or an operational constraint blocked re-hedging.")
    if used_fallback:
        reasons.append("Fallback action logic was used because ENHSP did not return a usable plan.")

    reasons.append("Planner action sequence: " + " -> ".join(actions) + ".")
    return reasons


def solve_numerical_planning(item: NumericalPlannerInput) -> NumericalPlannerResult:
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

    return NumericalPlannerResult(
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


def demo_scenarios() -> list[NumericalPlannerInput]:
    portfolio_value = 8_000_000.0
    futures_price = 24_000.0
    lot_size = 65
    margin_per_contract = futures_price * lot_size * 0.12
    available_margin = 2_000_000.0

    return [
        NumericalPlannerInput(
            scenario_name="num_low_risk_open",
            portfolio_value=portfolio_value,
            target_hedge_ratio=0.30,
            predicted_volatility=0.08,
            risk_regime="Low",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=0,
            margin_per_contract=margin_per_contract,
            available_margin=available_margin,
            days_to_expiry=20,
        ),
        NumericalPlannerInput(
            scenario_name="num_medium_risk_open",
            portfolio_value=portfolio_value,
            target_hedge_ratio=0.60,
            predicted_volatility=0.12,
            risk_regime="Medium",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=0,
            margin_per_contract=margin_per_contract,
            available_margin=available_margin,
            days_to_expiry=18,
        ),
        NumericalPlannerInput(
            scenario_name="num_high_risk_increase",
            portfolio_value=portfolio_value,
            target_hedge_ratio=0.80,
            predicted_volatility=0.18,
            risk_regime="High",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=2,
            margin_per_contract=margin_per_contract,
            available_margin=available_margin,
            days_to_expiry=15,
        ),
        NumericalPlannerInput(
            scenario_name="num_close_existing",
            portfolio_value=portfolio_value,
            target_hedge_ratio=0.00,
            predicted_volatility=0.07,
            risk_regime="Low",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=3,
            margin_per_contract=margin_per_contract,
            available_margin=available_margin,
            days_to_expiry=22,
        ),
        NumericalPlannerInput(
            scenario_name="num_margin_insufficient",
            portfolio_value=portfolio_value,
            target_hedge_ratio=0.80,
            predicted_volatility=0.18,
            risk_regime="High",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=1,
            margin_per_contract=margin_per_contract,
            available_margin=1.0,
            days_to_expiry=20,
        ),
        NumericalPlannerInput(
            scenario_name="num_roll_required",
            portfolio_value=portfolio_value,
            target_hedge_ratio=0.50,
            predicted_volatility=0.13,
            risk_regime="Medium",
            futures_price=futures_price,
            lot_size=lot_size,
            current_short_contracts=3,
            margin_per_contract=margin_per_contract,
            available_margin=available_margin,
            days_to_expiry=2,
        ),
    ]


def main() -> None:
    for item in demo_scenarios():
        result = solve_numerical_planning(item)
        state = result.prepared_state
        print(
            result.scenario_name,
            "| action:", result.primary_action,
            "| sequence:", " -> ".join(result.action_sequence),
            "| status:", result.planner_status,
            "| fallback:", result.used_fallback,
            "| current:", state["current_contracts"],
            "| target:", state["target_contracts"],
            "| executable:", state["executable_contracts"],
            "| gap:", state["hedge_gap_contracts"],
            "| time ms:", result.planning_time_ms,
        )


if __name__ == "__main__":
    main()