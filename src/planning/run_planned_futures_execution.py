"""
Integrated planning and futures-execution workflow.

Purpose
-------
Connect the tested planning layer with the tested futures contract calculator.

The workflow:
    1. Derive the current hedge ratio from the held short NIFTY futures contracts.
    2. Pass the current state, forecast risk regime, and rolling minimum-variance
       core target into Unified Planning + ENHSP.
    3. Convert the planner-approved target ratio into a whole number of short
       NIFTY futures contracts.
    4. Validate the exact additional margin requirement.
    5. Store an explainable audit record.

The lot size remains configurable. In the later live integration it should be
loaded dynamically from broker instrument metadata.

"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.planning.futures_execution import (
    FuturesExecutionInput,
    FuturesExecutionResult,
    calculate_futures_execution,
)
from src.planning.run_hedge_planner import (
    DEFAULT_MAX_TRADE_COST_MICROS,
    DEMO_MAX_TRADE_COST_MICROS,
    HedgePlanningInput,
    PlanningResult,
    solve_planning_input,
)

REPORT_DIR = ROOT / "outputs" / "reports" / "planning"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRADE_ACTIONS = {
    "OPEN_HEDGE",
    "INCREASE_HEDGE",
    "REDUCE_HEDGE",
    "CLOSE_HEDGE",
}


@dataclass(frozen=True)
class PlannedExecutionInput:
    """Inputs for one integrated planning and execution decision."""

    scenario_name: str
    portfolio_value: float
    core_target_ratio: float
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
    max_trade_cost_micros: int = DEFAULT_MAX_TRADE_COST_MICROS
    rounding_mode: str = "nearest"


@dataclass(frozen=True)
class PlannedExecutionResult:
    """Combined planner decision, contract calculation, and audit explanation."""

    scenario_name: str
    status: str

    current_hedge_ratio: float
    planner_target_hedge_ratio: float
    execution_target_hedge_ratio: float

    planner_action_sequence: list[str]
    operational_actions: list[str]
    position_action: str

    execution_status: str
    execution_instruction: str
    current_short_contracts: int
    recommended_short_contracts: int
    contract_delta: int

    margin_sufficient: bool | None
    explanation: list[str]

    planning_result: dict[str, Any]
    futures_execution_result: dict[str, Any]


def validate_input(item: PlannedExecutionInput) -> None:
    """Validate the integrated workflow inputs."""
    if item.portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")

    if item.futures_price <= 0:
        raise ValueError("futures_price must be positive")

    if item.lot_size <= 0:
        raise ValueError("lot_size must be positive")

    if item.current_short_contracts < 0:
        raise ValueError("current_short_contracts cannot be negative")

    if item.margin_per_contract < 0:
        raise ValueError("margin_per_contract cannot be negative")

    if item.available_margin < 0:
        raise ValueError("available_margin cannot be negative")


def derive_current_hedge_ratio(
    portfolio_value: float,
    futures_price: float,
    lot_size: int,
    current_short_contracts: int,
) -> float:
    """
    Derive the current hedge ratio from the actual held short-contract count.

    current hedge ratio =
        current short contracts * futures price * lot size
        -------------------------------------------------
                         portfolio value
    """
    current_short_notional = (
        current_short_contracts
        * futures_price
        * lot_size
    )

    return float(
        current_short_notional
        / portfolio_value
    )


def approved_execution_target_ratio(
    planning_result: PlanningResult,
    current_hedge_ratio: float,
) -> float:
    """
    Pass the planner target to execution only when a trade action is approved.

    For HOLD_POSITION, including ROLL_FUTURES_FORWARD -> HOLD_POSITION, preserve
    the existing hedge ratio so the execution layer does not accidentally
    recommend an unapproved rebalance.
    """
    actions = set(planning_result.action_sequence)

    if actions.intersection(TRADE_ACTIONS):
        planner_target_bps = planning_result.prepared_state[
            "planner_target_bps"
        ]

        return float(
            planner_target_bps
            / 100.0
        )

    return float(current_hedge_ratio)


def split_actions(
    action_sequence: list[str],
) -> tuple[list[str], str]:
    """Separate operational actions such as rolling from the position decision."""
    operational_actions = [
        action
        for action in action_sequence
        if action == "ROLL_FUTURES_FORWARD"
    ]

    position_actions = [
        action
        for action in action_sequence
        if action != "ROLL_FUTURES_FORWARD"
    ]

    position_action = (
        position_actions[-1]
        if position_actions
        else "HOLD_POSITION"
    )

    return operational_actions, position_action


def determine_status(
    planning_result: PlanningResult,
    execution_result: FuturesExecutionResult,
) -> str:
    """Create a top-level integrated status for dashboard and audit use."""
    if execution_result.status == "BLOCKED_INSUFFICIENT_MARGIN":
        return "BLOCKED_INSUFFICIENT_MARGIN"

    if planning_result.used_fallback:
        return "PLANNER_FALLBACK_USED"

    if (
        planning_result.action_sequence
        == ["HOLD_POSITION"]
        and execution_result.status
        == "NO_CONTRACT_CHANGE_REQUIRED"
    ):
        return "HOLD_POSITION"

    if "ROLL_FUTURES_FORWARD" in planning_result.action_sequence:
        return "ROLL_REQUIRED"

    return "EXECUTABLE_PLAN"


def run_planned_execution(
    item: PlannedExecutionInput,
) -> PlannedExecutionResult:
    """Execute the integrated planner-to-contract-calculation workflow."""
    validate_input(item)

    current_hedge_ratio = derive_current_hedge_ratio(
        portfolio_value=item.portfolio_value,
        futures_price=item.futures_price,
        lot_size=item.lot_size,
        current_short_contracts=item.current_short_contracts,
    )

    # The PDDL layer receives an early Boolean operational signal.
    # The execution layer below performs the exact rupee-level margin check.
    margin_available_for_planner = (
        item.available_margin > 0
    )

    planning_input = HedgePlanningInput(
        scenario_name=item.scenario_name,
        current_hedge_ratio=current_hedge_ratio,
        core_target_ratio=item.core_target_ratio,
        predicted_volatility=item.predicted_volatility,
        risk_regime=item.risk_regime,
        margin_available=margin_available_for_planner,
        days_to_expiry=item.days_to_expiry,
        rehedge_allowed=item.rehedge_allowed,
        transaction_cost_bps=item.transaction_cost_bps,
        max_trade_cost_micros=item.max_trade_cost_micros,
    )

    planning_result = solve_planning_input(
        planning_input
    )

    execution_target_ratio = approved_execution_target_ratio(
        planning_result=planning_result,
        current_hedge_ratio=current_hedge_ratio,
    )

    execution_input = FuturesExecutionInput(
        scenario_name=item.scenario_name,
        portfolio_value=item.portfolio_value,
        target_hedge_ratio=execution_target_ratio,
        futures_price=item.futures_price,
        lot_size=item.lot_size,
        current_short_contracts=item.current_short_contracts,
        rounding_mode=item.rounding_mode,
        margin_per_contract=item.margin_per_contract,
        available_margin=item.available_margin,
    )

    execution_result = calculate_futures_execution(
        execution_input
    )

    operational_actions, position_action = split_actions(
        planning_result.action_sequence
    )

    planner_target_ratio = (
        planning_result.prepared_state["planner_target_bps"]
        / 100.0
    )

    integrated_status = determine_status(
        planning_result=planning_result,
        execution_result=execution_result,
    )

    explanation = [
        (
            f"Current hedge ratio={current_hedge_ratio:.4f}, derived from "
            f"{item.current_short_contracts} held short contract(s)."
        ),
        (
            f"Planner action sequence="
            f"{' -> '.join(planning_result.action_sequence)}."
        ),
        (
            f"Planner target ratio={planner_target_ratio:.4f}; "
            f"execution target ratio={execution_target_ratio:.4f}."
        ),
        (
            f"Execution instruction="
            f"{execution_result.execution_instruction}; "
            f"contract change={execution_result.contract_delta:+d}."
        ),
    ]

    if operational_actions:
        explanation.append(
            "Operational action(s)="
            + " -> ".join(operational_actions)
            + "."
        )

    if (
        planning_result.action_sequence
        == ["HOLD_POSITION"]
        and execution_target_ratio
        == current_hedge_ratio
    ):
        explanation.append(
            "The execution target was kept equal to the current hedge ratio "
            "because the planner approved HOLD_POSITION rather than a rebalance."
        )

    if execution_result.margin_sufficient is False:
        explanation.append(
            "The proposed contract change must not be executed because exact "
            "available margin is insufficient."
        )

    return PlannedExecutionResult(
        scenario_name=item.scenario_name,
        status=integrated_status,
        current_hedge_ratio=float(current_hedge_ratio),
        planner_target_hedge_ratio=float(planner_target_ratio),
        execution_target_hedge_ratio=float(execution_target_ratio),
        planner_action_sequence=list(
            planning_result.action_sequence
        ),
        operational_actions=operational_actions,
        position_action=position_action,
        execution_status=execution_result.status,
        execution_instruction=(
            execution_result.execution_instruction
        ),
        current_short_contracts=(
            execution_result.current_short_contracts
        ),
        recommended_short_contracts=(
            execution_result.recommended_short_contracts
        ),
        contract_delta=execution_result.contract_delta,
        margin_sufficient=execution_result.margin_sufficient,
        explanation=explanation,
        planning_result=asdict(planning_result),
        futures_execution_result=asdict(execution_result),
    )


def demo_inputs() -> list[PlannedExecutionInput]:
    """Demonstrate integrated open, increase, reduce, close, hold, roll, and block."""
    common = {
        "portfolio_value": 10_000_000,
        "futures_price": 25_000,
        "lot_size": 65,
        "margin_per_contract": 210_000,
        "available_margin": 1_500_000,
        "max_trade_cost_micros": DEMO_MAX_TRADE_COST_MICROS,
    }

    return [
        PlannedExecutionInput(
            scenario_name="integrated_open_hedge",
            core_target_ratio=0.80,
            predicted_volatility=0.22,
            risk_regime="HIGH",
            current_short_contracts=0,
            days_to_expiry=12,
            **common,
        ),
        PlannedExecutionInput(
            scenario_name="integrated_increase_hedge",
            core_target_ratio=0.90,
            predicted_volatility=0.19,
            risk_regime="HIGH",
            current_short_contracts=4,
            days_to_expiry=10,
            **common,
        ),
        PlannedExecutionInput(
            scenario_name="integrated_reduce_hedge",
            core_target_ratio=0.65,
            predicted_volatility=0.08,
            risk_regime="LOW",
            current_short_contracts=6,
            days_to_expiry=9,
            **common,
        ),
        PlannedExecutionInput(
            scenario_name="integrated_close_hedge",
            core_target_ratio=0.05,
            predicted_volatility=0.05,
            risk_regime="LOW",
            current_short_contracts=3,
            days_to_expiry=11,
            **common,
        ),
        PlannedExecutionInput(
            scenario_name="integrated_hold_position",
            core_target_ratio=0.63,
            predicted_volatility=0.12,
            risk_regime="MEDIUM",
            current_short_contracts=4,
            days_to_expiry=15,
            **common,
        ),
        PlannedExecutionInput(
            scenario_name="integrated_roll_and_hold",
            core_target_ratio=0.65,
            predicted_volatility=0.13,
            risk_regime="MEDIUM",
            current_short_contracts=4,
            days_to_expiry=2,
            **common,
        ),
        PlannedExecutionInput(
            scenario_name="integrated_exact_margin_block",
            core_target_ratio=0.80,
            predicted_volatility=0.22,
            risk_regime="HIGH",
            current_short_contracts=0,
            days_to_expiry=12,
            portfolio_value=10_000_000,
            futures_price=25_000,
            lot_size=65,
            margin_per_contract=210_000,
            available_margin=300_000,
            max_trade_cost_micros=DEMO_MAX_TRADE_COST_MICROS,
        ),
    ]


def print_result(
    result: PlannedExecutionResult,
) -> None:
    """Print one concise integrated recommendation."""
    print("\n" + "=" * 94)
    print(f"Scenario                 : {result.scenario_name}")
    print(f"Integrated status        : {result.status}")
    print(
        "Planner action plan      : "
        + " -> ".join(
            result.planner_action_sequence
        )
    )

    if result.operational_actions:
        print(
            "Operational action(s)    : "
            + " -> ".join(
                result.operational_actions
            )
        )

    print(f"Position action          : {result.position_action}")
    print(
        f"Current hedge ratio      : "
        f"{result.current_hedge_ratio:.4f}"
    )
    print(
        f"Planner target ratio     : "
        f"{result.planner_target_hedge_ratio:.4f}"
    )
    print(
        f"Execution target ratio   : "
        f"{result.execution_target_hedge_ratio:.4f}"
    )
    print(
        f"Current short contracts  : "
        f"{result.current_short_contracts}"
    )
    print(
        f"Recommended contracts    : "
        f"{result.recommended_short_contracts}"
    )
    print(f"Contract change          : {result.contract_delta:+d}")
    print(f"Execution status         : {result.execution_status}")
    print(f"Execution instruction    : {result.execution_instruction}")

    if result.margin_sufficient is not None:
        print(f"Margin sufficient        : {result.margin_sufficient}")

    print("Explanation:")
    for reason in result.explanation:
        print(f"  - {reason}")


def write_results(
    results: list[PlannedExecutionResult],
    filename: str,
) -> Path:
    """Save integrated audit results as JSON."""
    output_file = REPORT_DIR / filename

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [asdict(result) for result in results],
            file,
            indent=2,
        )

    return output_file


def parse_args() -> argparse.Namespace:
    """Parse CLI inputs for a demonstration set or one custom scenario."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the integrated Hedra planning and futures-execution workflow."
        )
    )

    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--scenario-name", default="custom_planned_execution")
    parser.add_argument("--portfolio-value", type=float, default=10_000_000)
    parser.add_argument("--core-target-ratio", type=float, default=0.80)
    parser.add_argument("--predicted-volatility", type=float, default=0.22)
    parser.add_argument(
        "--risk-regime",
        choices=["LOW", "MEDIUM", "HIGH"],
        default="HIGH",
    )
    parser.add_argument("--futures-price", type=float, default=25_000)
    parser.add_argument("--lot-size", type=int, default=65)
    parser.add_argument("--current-short-contracts", type=int, default=0)
    parser.add_argument("--margin-per-contract", type=float, default=210_000)
    parser.add_argument("--available-margin", type=float, default=1_500_000)
    parser.add_argument("--days-to-expiry", type=int, default=12)
    parser.add_argument(
        "--rehedge-allowed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-trade-cost-micros",
        type=int,
        default=DEFAULT_MAX_TRADE_COST_MICROS,
    )
    parser.add_argument(
        "--rounding-mode",
        choices=["nearest", "floor", "ceil"],
        default="nearest",
    )

    return parser.parse_args()


def main() -> None:
    """Run the integrated demonstrations or one custom decision."""
    args = parse_args()

    if args.demo:
        items = demo_inputs()
        filename = "planned_futures_execution_demo_results.json"
    else:
        items = [
            PlannedExecutionInput(
                scenario_name=args.scenario_name,
                portfolio_value=args.portfolio_value,
                core_target_ratio=args.core_target_ratio,
                predicted_volatility=args.predicted_volatility,
                risk_regime=args.risk_regime,
                futures_price=args.futures_price,
                lot_size=args.lot_size,
                current_short_contracts=args.current_short_contracts,
                margin_per_contract=args.margin_per_contract,
                available_margin=args.available_margin,
                days_to_expiry=args.days_to_expiry,
                rehedge_allowed=args.rehedge_allowed,
                max_trade_cost_micros=args.max_trade_cost_micros,
                rounding_mode=args.rounding_mode,
            )
        ]

        filename = "planned_futures_execution_latest_result.json"

    results = [
        run_planned_execution(item)
        for item in items
    ]

    for result in results:
        print_result(result)

    output_file = write_results(
        results=results,
        filename=filename,
    )

    print("\nSaved integrated planning-execution audit output:")
    print(f"  {output_file}")


if __name__ == "__main__":
    main()
