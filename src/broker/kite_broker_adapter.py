"""
kite_broker_adapter.py

Reusable Kite broker-data adapter for the Hedra hedge-planning workflow.

This module does NOT place orders. It only reads live broker/account data:
- nearest NIFTY futures contract
- futures quote / last traded price
- lot size
- expiry and days to expiry
- current NIFTY futures position
- available margin

Run from project root:
    python src/broker/kite_broker_adapter.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from kiteconnect import KiteConnect


IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class NiftyFutureContract:
    tradingsymbol: str
    instrument_token: int
    exchange: str
    name: str
    expiry: str
    lot_size: int
    instrument_type: str


@dataclass
class NiftyFuturesPosition:
    tradingsymbol: str
    net_quantity: int
    current_short_contracts: int
    current_long_contracts: int
    average_price: float


@dataclass
class BrokerSnapshot:
    contract: NiftyFutureContract
    futures_price: float
    days_to_expiry: int
    current_position: NiftyFuturesPosition
    available_margin: Optional[float]
    quote_key: str


def _today_ist() -> date:
    return datetime.now(IST).date()


def _normalise_expiry(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def get_kite_client() -> KiteConnect:
    """
    Create an authenticated Kite client using environment variables.

    Required .env keys:
        KITE_API_KEY
        KITE_ACCESS_TOKEN
    """
    load_dotenv()

    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key:
        raise RuntimeError("KITE_API_KEY is missing from environment/.env")
    if not access_token:
        raise RuntimeError("KITE_ACCESS_TOKEN is missing from environment/.env")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token.strip("'").strip('"'))
    return kite


def get_nearest_nifty_future(kite: KiteConnect) -> NiftyFutureContract:
    """
    Fetch the nearest non-expired NIFTY futures contract from the NFO instrument list.
    """
    instruments = kite.instruments("NFO")
    today = _today_ist()

    candidates = []
    for inst in instruments:
        if inst.get("name") != "NIFTY":
            continue
        if inst.get("instrument_type") != "FUT":
            continue

        expiry = _normalise_expiry(inst.get("expiry"))
        if expiry < today:
            continue

        candidates.append((expiry, inst))

    if not candidates:
        raise RuntimeError("No active NIFTY futures contract found in NFO instruments")

    expiry, inst = sorted(candidates, key=lambda item: item[0])[0]

    return NiftyFutureContract(
        tradingsymbol=str(inst["tradingsymbol"]),
        instrument_token=int(inst["instrument_token"]),
        exchange=str(inst.get("exchange", "NFO")),
        name=str(inst.get("name", "NIFTY")),
        expiry=expiry.isoformat(),
        lot_size=int(inst.get("lot_size", 0)),
        instrument_type=str(inst.get("instrument_type", "FUT")),
    )


def get_futures_price(kite: KiteConnect, contract: NiftyFutureContract) -> tuple[float, str]:
    """
    Fetch live quote and return last traded price for the selected futures contract.
    """
    quote_key = f"{contract.exchange}:{contract.tradingsymbol}"
    quotes = kite.quote([quote_key])

    if quote_key not in quotes:
        raise RuntimeError(f"Quote not returned for {quote_key}")

    price = float(quotes[quote_key]["last_price"])
    return price, quote_key


def get_current_nifty_futures_position(
    kite: KiteConnect,
    contract: NiftyFutureContract,
) -> NiftyFuturesPosition:
    """
    Read current net position for the selected NIFTY futures contract.

    In Kite, a short futures quantity is normally negative.
    Example:
        quantity = -2 means 2 short contracts.
    """
    positions = kite.positions()
    net_positions = positions.get("net", [])

    match = None
    for pos in net_positions:
        if pos.get("exchange") != contract.exchange:
            continue
        if pos.get("tradingsymbol") != contract.tradingsymbol:
            continue
        match = pos
        break

    if match is None:
        return NiftyFuturesPosition(
            tradingsymbol=contract.tradingsymbol,
            net_quantity=0,
            current_short_contracts=0,
            current_long_contracts=0,
            average_price=0.0,
        )

    quantity = int(match.get("quantity", 0))
    average_price = float(match.get("average_price", 0.0) or 0.0)

    return NiftyFuturesPosition(
        tradingsymbol=contract.tradingsymbol,
        net_quantity=quantity,
        current_short_contracts=max(-quantity, 0),
        current_long_contracts=max(quantity, 0),
        average_price=average_price,
    )


def get_available_margin(kite: KiteConnect) -> Optional[float]:
    """
    Read available equity margin.

    Kite margin response structure can vary slightly, so this function tries
    common fields safely and returns None if margin cannot be identified.
    """
    try:
        margins = kite.margins()
    except Exception:
        return None

    equity = margins.get("equity", {}) if isinstance(margins, dict) else {}

    candidates = [
        equity.get("net"),
        equity.get("available", {}).get("live_balance") if isinstance(equity.get("available"), dict) else None,
        equity.get("available", {}).get("cash") if isinstance(equity.get("available"), dict) else None,
    ]

    for value in candidates:
        if value is not None:
            try:
                return float(value)
            except Exception:
                continue

    return None


def get_live_broker_snapshot(kite: Optional[KiteConnect] = None) -> BrokerSnapshot:
    """
    Collect all live broker values needed by the planning/execution workflow.
    """
    kite = kite or get_kite_client()

    contract = get_nearest_nifty_future(kite)
    futures_price, quote_key = get_futures_price(kite, contract)
    position = get_current_nifty_futures_position(kite, contract)
    available_margin = get_available_margin(kite)

    expiry_date = _normalise_expiry(contract.expiry)
    days_to_expiry = max((expiry_date - _today_ist()).days, 0)

    return BrokerSnapshot(
        contract=contract,
        futures_price=futures_price,
        days_to_expiry=days_to_expiry,
        current_position=position,
        available_margin=available_margin,
        quote_key=quote_key,
    )


def main() -> None:
    snapshot = get_live_broker_snapshot()
    print(json.dumps(asdict(snapshot), indent=2))


if __name__ == "__main__":
    main()
