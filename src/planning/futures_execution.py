"""
Execution-calculation layer for Nifty futures hedging system.

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "reports" / "planning"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RoundingMode = Literal["nearest", "floor", "ceil"]


@dataclass(frozen=True)
class FuturesExecutionInput:
    """Inputs required to calculate an executable futures position."""

    scenario_name: str
    portfolio_value: float
    target_hedge_ratio: float
    futures_price: float
    lot_size: int
    current_short_contracts: int = 0
    rounding_mode: RoundingMode = "nearest"
    margin_per_contract: float | None = None
    available_margin: float | None = None


@dataclass(frozen=True)
class FuturesExecutionResult:
    """Execution recommendation after whole-contract rounding and margin checks."""

    scenario_name: str
    status: str
    execution_instruction: str

    portfolio_value: float
    target_hedge_ratio: float
    futures_price: float
    lot_size: int
    contract_notional: float

    target_futures_notional: float
    raw_required_contracts: float
    recommended_short_contracts: int
    current_short_contracts: int
    contract_delta: int

    achieved_futures_notional: float
    achieved_hedge_ratio: float
    residual_hedge_ratio_error: float

    additional_margin_required: float | None
    available_margin: float | None
    margin_sufficient: bool | None

    explanation: list[str]


def validate_input(item: FuturesExecutionInput) -> None:
    """Reject invalid execution inputs early."""
    if item.portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")

    if item.target_hedge_ratio < 0:
        raise ValueError("target_hedge_ratio cannot be negative")

    if item.futures_price <= 0:
        raise ValueError("futures_price must be positive")

    if item.lot_size <= 0:
        raise ValueError("lot_size must be a positive integer")

    if item.current_short_contracts < 0:
        raise ValueError("current_short_contracts cannot be negative")

    if item.rounding_mode not in {"nearest", "floor", "ceil"}:
        raise ValueError(
            "rounding_mode must be one of: nearest, floor, ceil"
        )

    if item.margin_per_contract is not None and item.margin_per_contract < 0:
        raise ValueError("margin_per_contract cannot be negative")

    if item.available_margin is not None and item.available_margin < 0:
        raise ValueError("available_margin cannot be negative")


def round_contract_count(
    raw_contracts: float,
    mode: RoundingMode,
) -> int:
    """
    Convert fractional futures contracts into a valid whole-contract count.

    """
    if mode == "floor":
        return max(0, int(math.floor(raw_contracts)))

    if mode == "ceil":
        return max(0, int(math.ceil(raw_contracts)))

    # Deterministic half-up rounding rather than Python's banker's rounding.
    return max(0, int(math.floor(raw_contracts + 0.5)))


def execution_instruction(
    current_short_contracts: int,
    recommended_short_contracts: int,
) -> str:
    """Translate the required contract change into an operational instruction."""
    delta = recommended_short_contracts - current_short_contracts

    if delta == 0:
        return "HOLD_EXISTING_SHORT_POSITION"

    if current_short_contracts == 0 and delta > 0:
        return "SELL_TO_OPEN_SHORT_FUTURES"

    if delta > 0:
        return "SELL_ADDITIONAL_SHORT_FUTURES"

    if recommended_short_contracts == 0:
        return "BUY_TO_CLOSE_SHORT_FUTURES"

    return "BUY_TO_REDUCE_SHORT_FUTURES"


def calculate_futures_execution(
    item: FuturesExecutionInput,
) -> FuturesExecutionResult:
    """Calculate whole-contract futures execution requirements."""
    validate_input(item)

    contract_notional = item.futures_price * item.lot_size
    target_futures_notional = (
        item.portfolio_value
        * item.target_hedge_ratio
    )

    raw_required_contracts = (
        target_futures_notional
        / contract_notional
    )

    recommended_short_contracts = round_contract_count(
        raw_contracts=raw_required_contracts,
        mode=item.rounding_mode,
    )

    contract_delta = (
        recommended_short_contracts
        - item.current_short_contracts
    )

    achieved_futures_notional = (
        recommended_short_contracts
        * contract_notional
    )

    achieved_hedge_ratio = (
        achieved_futures_notional
        / item.portfolio_value
    )

    residual_hedge_ratio_error = (
        achieved_hedge_ratio
        - item.target_hedge_ratio
    )

    additional_margin_required: float | None = None
    margin_sufficient: bool | None = None

    if item.margin_per_contract is not None:
        additional_margin_required = (
            max(contract_delta, 0)
            * item.margin_per_contract
        )

    if (
        additional_margin_required is not None
        and item.available_margin is not None
    ):
        margin_sufficient = (
            item.available_margin
            >= additional_margin_required
        )

    instruction = execution_instruction(
        current_short_contracts=item.current_short_contracts,
        recommended_short_contracts=recommended_short_contracts,
    )

    if contract_delta == 0:
        status = "NO_CONTRACT_CHANGE_REQUIRED"
    elif margin_sufficient is False:
        status = "BLOCKED_INSUFFICIENT_MARGIN"
        instruction = "DO_NOT_EXECUTE_INSUFFICIENT_MARGIN"
    else:
        status = "EXECUTABLE"

    explanation = [
        (
            f"Target short-futures notional="
            f"{target_futures_notional:,.2f}."
        ),
        (
            f"One futures contract notional="
            f"{contract_notional:,.2f} "
            f"({item.futures_price:,.2f} x {item.lot_size})."
        ),
        (
            f"Raw required contracts={raw_required_contracts:.4f}; "
            f"rounded using '{item.rounding_mode}'="
            f"{recommended_short_contracts} contract(s)."
        ),
        (
            f"Current short contracts={item.current_short_contracts}; "
            f"recommended short contracts={recommended_short_contracts}; "
            f"change={contract_delta:+d}."
        ),
        (
            f"Achieved hedge ratio={achieved_hedge_ratio:.4f}; "
            f"target hedge ratio={item.target_hedge_ratio:.4f}; "
            f"residual error={residual_hedge_ratio_error:+.4f}."
        ),
    ]

    if additional_margin_required is not None:
        explanation.append(
            f"Additional margin required="
            f"{additional_margin_required:,.2f}."
        )

    if (
        item.available_margin is not None
        and margin_sufficient is not None
    ):
        explanation.append(
            f"Available margin={item.available_margin:,.2f}; "
            f"margin sufficient={margin_sufficient}."
        )

    explanation.append(
        f"Execution instruction={instruction}."
    )

    return FuturesExecutionResult(
        scenario_name=item.scenario_name,
        status=status,
        execution_instruction=instruction,
        portfolio_value=float(item.portfolio_value),
        target_hedge_ratio=float(item.target_hedge_ratio),
        futures_price=float(item.futures_price),
        lot_size=int(item.lot_size),
        contract_notional=float(contract_notional),
        target_futures_notional=float(target_futures_notional),
        raw_required_contracts=float(raw_required_contracts),
        recommended_short_contracts=int(recommended_short_contracts),
        current_short_contracts=int(item.current_short_contracts),
        contract_delta=int(contract_delta),
        achieved_futures_notional=float(achieved_futures_notional),
        achieved_hedge_ratio=float(achieved_hedge_ratio),
        residual_hedge_ratio_error=float(residual_hedge_ratio_error),
        additional_margin_required=(
            None
            if additional_margin_required is None
            else float(additional_margin_required)
        ),
        available_margin=(
            None
            if item.available_margin is None
            else float(item.available_margin)
        ),
        margin_sufficient=margin_sufficient,
        explanation=explanation,
    )


def demo_inputs() -> list[FuturesExecutionInput]:
    """Demonstrate open, increase, reduce, close, hold, and margin-block cases."""
    common = {
        "portfolio_value": 10_000_000,
        "futures_price": 25_000,
        "lot_size": 65,
        "margin_per_contract": 210_000,
        "available_margin": 1_500_000,
    }

    return [
        FuturesExecutionInput(
            scenario_name="demo_open_short_hedge",
            target_hedge_ratio=0.90,
            current_short_contracts=0,
            **common,
        ),
        FuturesExecutionInput(
            scenario_name="demo_increase_short_hedge",
            target_hedge_ratio=1.10,
            current_short_contracts=4,
            **common,
        ),
        FuturesExecutionInput(
            scenario_name="demo_reduce_short_hedge",
            target_hedge_ratio=0.55,
            current_short_contracts=5,
            **common,
        ),
        FuturesExecutionInput(
            scenario_name="demo_close_short_hedge",
            target_hedge_ratio=0.00,
            current_short_contracts=3,
            **common,
        ),
        FuturesExecutionInput(
            scenario_name="demo_hold_short_hedge",
            target_hedge_ratio=0.65,
            current_short_contracts=4,
            **common,
        ),
        FuturesExecutionInput(
            scenario_name="demo_margin_block",
            target_hedge_ratio=1.30,
            current_short_contracts=0,
            available_margin=300_000,
            portfolio_value=10_000_000,
            futures_price=25_000,
            lot_size=65,
            margin_per_contract=210_000,
        ),
    ]


def print_result(result: FuturesExecutionResult) -> None:
    """Print one concise execution recommendation."""
    print("\n" + "=" * 86)
    print(f"Scenario                : {result.scenario_name}")
    print(f"Status                  : {result.status}")
    print(f"Target hedge ratio      : {result.target_hedge_ratio:.4f}")
    print(f"Achieved hedge ratio    : {result.achieved_hedge_ratio:.4f}")
    print(f"Current short contracts : {result.current_short_contracts}")
    print(f"Required short contracts: {result.recommended_short_contracts}")
    print(f"Contract change         : {result.contract_delta:+d}")
    print(f"Instruction             : {result.execution_instruction}")
    if result.additional_margin_required is not None:
        print(
            "Additional margin       : "
            f"{result.additional_margin_required:,.2f}"
        )
    if result.margin_sufficient is not None:
        print(f"Margin sufficient       : {result.margin_sufficient}")

    print("Explanation:")
    for reason in result.explanation:
        print(f"  - {reason}")


def write_results(
    results: list[FuturesExecutionResult],
    filename: str,
) -> Path:
    """Save execution audit results as JSON."""
    output_file = REPORT_DIR / filename

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            [asdict(result) for result in results],
            file,
            indent=2,
        )

    return output_file


def parse_args() -> argparse.Namespace:
    """Parse CLI inputs."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a planner target hedge ratio into "
            "short Nifty futures contract requirements."
        )
    )

    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--scenario-name", default="custom_execution")
    parser.add_argument("--portfolio-value", type=float, default=10_000_000)
    parser.add_argument("--target-hedge-ratio", type=float, default=0.90)
    parser.add_argument("--futures-price", type=float, default=25_000)
    parser.add_argument("--lot-size", type=int, default=65)
    parser.add_argument("--current-short-contracts", type=int, default=0)
    parser.add_argument(
        "--rounding-mode",
        choices=["nearest", "floor", "ceil"],
        default="nearest",
    )
    parser.add_argument("--margin-per-contract", type=float)
    parser.add_argument("--available-margin", type=float)

    return parser.parse_args()


def main() -> None:
    """Run demonstration cases or a custom execution calculation."""
    args = parse_args()

    if args.demo:
        items = demo_inputs()
        filename = "futures_execution_demo_results.json"
    else:
        items = [
            FuturesExecutionInput(
                scenario_name=args.scenario_name,
                portfolio_value=args.portfolio_value,
                target_hedge_ratio=args.target_hedge_ratio,
                futures_price=args.futures_price,
                lot_size=args.lot_size,
                current_short_contracts=args.current_short_contracts,
                rounding_mode=args.rounding_mode,
                margin_per_contract=args.margin_per_contract,
                available_margin=args.available_margin,
            )
        ]
        filename = "futures_execution_latest_result.json"

    results = [
        calculate_futures_execution(item)
        for item in items
    ]

    for result in results:
        print_result(result)

    output_file = write_results(
        results=results,
        filename=filename,
    )

    print("\nSaved execution audit output:")
    print(f"  {output_file}")


if __name__ == "__main__":
    main()