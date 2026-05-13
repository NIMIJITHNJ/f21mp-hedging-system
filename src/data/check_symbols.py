import os
from dotenv import load_dotenv
from kiteconnect import KiteConnect

# Load keys from .env file in project root
load_dotenv()

API_KEY      = os.getenv("KITE_API_KEY")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

if not API_KEY or not ACCESS_TOKEN:
    raise ValueError("KITE_API_KEY or KITE_ACCESS_TOKEN not found in .env file")

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# Check connection
profile = kite.profile()
print(f"Connected as: {profile['user_name']}\n")

# Find all NIFTY futures (excluding BANKNIFTY)
print("── NIFTY futures available on NFO ──")
nfo = kite.instruments("NFO")
nifty_futs = [
    i for i in nfo
    if i["name"] == "NIFTY"
    and i["instrument_type"] == "FUT"
]
nifty_futs.sort(key=lambda x: x["expiry"])
for i in nifty_futs[:6]:
    print(f"  {i['tradingsymbol']:<20} expiry={i['expiry']}  token={i['instrument_token']}")

# Check stock symbols
print("\n── Checking stock symbols on NSE ──")
stocks_to_check = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY",
                   "TCS", "SBIN", "LT", "BHARTIARTL", "ITC", "AXISBANK"]
nse = kite.instruments("NSE")
nse_dict = {i["tradingsymbol"]: i for i in nse}
for sym in stocks_to_check:
    found = sym in nse_dict
    token = nse_dict[sym]["instrument_token"] if found else "NOT FOUND"
    print(f"  {sym:<15} {'OK' if found else 'MISSING':<8} token={token}")

# Check NIFTY 50 spot
print("\n── Checking NIFTY 50 spot ──")
if "NIFTY 50" in nse_dict:
    print(f"  NIFTY 50  OK  token={nse_dict['NIFTY 50']['instrument_token']}")
else:
    print("  NIFTY 50 NOT FOUND — trying NIFTY...")
    if "NIFTY" in nse_dict:
        print(f"  NIFTY  OK  token={nse_dict['NIFTY']['instrument_token']}")