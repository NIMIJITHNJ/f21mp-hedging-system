"""Combine read-only Kite data with simulated exposure and margin to recommend a hedge."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.broker.kite_broker_adapter import get_live_broker_snapshot
from src.planning.run_planned_futures_execution import (
    PlannedExecutionInput,
    run_planned_execution,
)


def main() -> None:
    snapshot = get_live_broker_snapshot()

    # Simulation settings.
    # Kite is used for live futures contract data and price only.
    total_capital = 10_000_000.0
    stock_allocation = total_capital * 0.80
    simulated_margin_reserve = total_capital - stock_allocation

    # Later these values will come from the GJR-GARCH model
    # and rolling minimum-variance hedge-ratio engine.
    portfolio_value = stock_allocation
    predicted_volatility = 0.18
    risk_regime = "HIGH"
    core_target_ratio = 0.70

    # Margin estimate used for broker integration testing.
    # Later this can be replaced by an exact broker/SPAN margin calculation.
    margin_per_contract = snapshot.futures_price * snapshot.contract.lot_size * 0.12

    item = PlannedExecutionInput(
        scenario_name="live_broker_planning_test",
        portfolio_value=portfolio_value,
        core_target_ratio=core_target_ratio,
        predicted_volatility=predicted_volatility,
        risk_regime=risk_regime,
        futures_price=snapshot.futures_price,
        lot_size=snapshot.contract.lot_size,
        current_short_contracts=snapshot.current_position.current_short_contracts,
        margin_per_contract=margin_per_contract,
        available_margin=simulated_margin_reserve,
        days_to_expiry=snapshot.days_to_expiry,
        max_trade_cost_micros=250,
    )

    result = run_planned_execution(item)

    output = {
        "live_broker_snapshot": asdict(snapshot),
        "test_model_inputs": {
            "total_capital": total_capital,
            "stock_allocation": stock_allocation,
            "simulated_margin_reserve": simulated_margin_reserve,
            "kite_reported_available_margin": snapshot.available_margin,
            "portfolio_value": portfolio_value,
            "predicted_volatility": predicted_volatility,
            "risk_regime": risk_regime,
            "core_target_ratio": core_target_ratio,
            "margin_per_contract_estimate": margin_per_contract,
        },
        "planned_execution_result": asdict(result),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

