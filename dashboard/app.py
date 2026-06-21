"""
app.py

Entry point for Hedra â€” the dynamic risk hedging dashboard.

Flow:
    No token present  â†’ authentication page
    Token present     â†’ dashboard home (system status + portfolio + sector chart)
    Kite callback     â†’ exchange request_token, save to .env, show dashboard home

Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import subprocess
import sys
import sys
import math
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta, date as dt_date
from html import escape

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.broker.kite_broker_adapter import get_live_broker_snapshot
from src.planning.run_planned_futures_execution import (
    PlannedExecutionInput,
    run_planned_execution,
)
from src.models.live_volatility_signal import get_live_volatility_signal


load_dotenv()

API_KEY    = os.getenv("KITE_API_KEY")
API_SECRET = os.getenv("KITE_API_SECRET")
ENV_PATH   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))

IST               = timezone(timedelta(hours=5, minutes=30))
ENTRY_DATE        = dt_date(2025, 4, 1)
TOTAL_CAPITAL = 10_000_000        # ₹1,00,00,000 total simulation capital
STOCK_ALLOCATION_RATIO = 0.80     # 80% invested in stocks
MARGIN_RESERVE_RATIO = 0.20       # 20% reserved for futures margin
STOCK_ALLOCATION = TOTAL_CAPITAL * STOCK_ALLOCATION_RATIO
MARGIN_RESERVE = TOTAL_CAPITAL - STOCK_ALLOCATION

PORTFOLIO = [
    ("RELIANCE",   "Energy"),
    ("HDFCBANK",   "Financials"),
    ("ICICIBANK",  "Financials"),
    ("INFY",       "Technology"),
    ("TCS",        "Technology"),
    ("SBIN",       "Financials"),
    ("LT",         "Industrials"),
    ("BHARTIARTL", "Telecom"),
    ("ITC",        "FMCG"),
    ("AXISBANK",   "Financials"),
]

CAPITAL_PER_STOCK = STOCK_ALLOCATION / len(PORTFOLIO)

SECTOR_COLOURS = {
    "Financials":  "#6c63ff",
    "Technology":  "#4ade80",
    "Energy":      "#fbbf24",
    "Industrials": "#60a5fa",
    "Telecom":     "#f87171",
    "FMCG":        "#a78bfa",
}


DISPLAY_LABELS = {
    "OPEN_HEDGE": "Open Hedge",
    "INCREASE_HEDGE": "Increase Hedge",
    "REDUCE_HEDGE": "Reduce Hedge",
    "CLOSE_HEDGE": "Close Hedge",
    "HOLD_POSITION": "Hold Position",
    "ROLL_FUTURES_FORWARD": "Roll Futures Forward",

    "SELL_TO_OPEN_SHORT_FUTURES": "Sell to Open Short Futures",
    "SELL_ADDITIONAL_SHORT_FUTURES": "Sell Additional Short Futures",
    "BUY_TO_REDUCE_SHORT_FUTURES": "Buy to Reduce Short Futures",
    "BUY_TO_CLOSE_SHORT_FUTURES": "Buy to Close Short Futures",
    "HOLD_EXISTING_SHORT_POSITION": "Hold Existing Short Position",
    "DO_NOT_EXECUTE_INSUFFICIENT_MARGIN": "Do Not Execute - Insufficient Margin",

    "no_hedge": "No Hedge",
    "static_50": "Static 50% Hedge",
    "rolling_min_variance": "Rolling Minimum-Variance",
    "gjr_scaled_minvar": "GJR-Scaled Minimum-Variance",
    "gjr_regime_dynamic": "GJR Regime Dynamic",
    "dcc_raw": "DCC Raw",
    "dcc_ema20_band0p10": "DCC EMA(20) + 10% No-Trade Band",
    "dcc_ema20_band0p05": "DCC EMA(20) + 5% No-Trade Band",
    "dcc_ema10_band0p10": "DCC EMA(10) + 10% No-Trade Band",
    "literature_vecm_static": "Literature VECM Static",
    "portfolio_static_minvar": "Portfolio Static Minimum-Variance",
}


def ui_label(value):
    if value is None:
        return "N/A"

    raw = str(value).strip()

    if raw in DISPLAY_LABELS:
        return DISPLAY_LABELS[raw]

    dcc_match = re.match(r"^dcc_ema(\d+)_band0p(\d+)$", raw)
    if dcc_match:
        ema_window = dcc_match.group(1)
        band_pct = int(dcc_match.group(2))
        return f"DCC EMA({ema_window}) + {band_pct}% No-Trade Band"

    if raw == "dcc_raw":
        return "Raw DCC-GARCH Hedge"

    return raw.replace("_", " ").replace("-", " ").title()


def ui_text(value):
    text_value = str(value)
    for raw, clean in sorted(DISPLAY_LABELS.items(), key=lambda item: len(item[0]), reverse=True):
        text_value = text_value.replace(raw, clean)
    return text_value.replace("_", " ")


st.set_page_config(
    page_title="Hedra",
    page_icon="â¬¡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True


# â”€â”€ XIRR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def xirr(cashflows, dates):
    """
    Compute XIRR given a list of cashflows and corresponding dates.
    Negative cashflow = outflow (purchase), positive = inflow (current value).
    Uses scipy brentq to solve NPV = 0. Returns annualised % or None.
    """
    if len(cashflows) < 2:
        return None
    d0 = dates[0]

    def npv(rate):
        return sum(
            cf / (1 + rate) ** ((d - d0).days / 365.0)
            for cf, d in zip(cashflows, dates)
        )
    try:
        rate = brentq(npv, -0.999, 100.0, maxiter=1000)
        return rate * 100
    except Exception:
        return None


# â”€â”€ CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_css(auth_page=False):
    sidebar_width = "280px" if st.session_state.sidebar_open else "72px"
    main_padding  = "40px" if auth_page else (
        "340px" if st.session_state.sidebar_open else "120px"
    )

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-color: #0d0d0f !important;
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stSidebar"], [data-testid="collapsedControl"],
#MainMenu, footer {{ display: none !important; }}

[data-testid="stMainBlockContainer"] {{
    padding: 36px 40px 40px {main_padding} !important;
    max-width: 100% !important;
}}

div[data-testid="stButton"] {{
    position: fixed; top: 18px; left: 18px; z-index: 10000;
}}
div[data-testid="stButton"] button {{
    background: #1e1e28 !important; color: #f0f0f5 !important;
    border: 1px solid #2a2a30 !important; border-radius: 8px !important;
    width: 38px !important; height: 34px !important;
    padding: 0 !important; font-size: 18px !important;
}}

.custom-sidebar {{
    position: fixed; top: 0; left: 0;
    width: {sidebar_width}; height: 100vh;
    background: #16161a; border-right: 1px solid #2a2a30;
    padding: 76px 22px 34px 22px; box-sizing: border-box;
    z-index: 9999; transition: width 0.25s ease; overflow: hidden;
}}
.sidebar-logo         {{ display: flex; align-items: center; gap: 12px; margin-bottom: 28px; }}
.sidebar-logo-box     {{ width: 38px; height: 38px; min-width: 38px; border-radius: 9px; background: #6c63ff; color: white; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 16px; }}
.sidebar-title        {{ font-size: 18px; font-weight: 600; color: #f0f0f5; white-space: nowrap; }}
.sidebar-subtitle     {{ font-size: 10px; color: #6c63ff; letter-spacing: 0.8px; text-transform: uppercase; margin-top: 3px; white-space: nowrap; }}
.sidebar-line         {{ height: 1px; background: #2a2a30; margin: 22px 0; }}
.sidebar-label        {{ font-size: 11px; color: #555560; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 8px; }}
.sidebar-name         {{ font-size: 13px; color: #f0f0f5; font-weight: 500; margin-bottom: 2px; white-space: nowrap; }}
.sidebar-email        {{ font-size: 12px; color: #555560; margin-bottom: 14px; white-space: nowrap; }}
.sidebar-pill         {{ display: inline-flex; align-items: center; gap: 7px; padding: 5px 11px; border-radius: 18px; background: #1e1e28; border: 1px solid #2a2a30; font-size: 11px; color: #4ade80; white-space: nowrap; }}
.sidebar-dot          {{ width: 6px; height: 6px; background: #4ade80; border-radius: 50%; }}
.sidebar-page         {{ padding: 10px 0; border-bottom: 1px solid #1e1e28; font-size: 13px; color: #9090a0; white-space: nowrap; }}
.sidebar-page-disabled{{ color: #3a3a3a; }}
.sidebar-soon         {{ font-size: 10px; color: #3a3a3a; margin-left: 4px; }}
.sidebar-icon-item    {{ color: #9090a0; font-size: 19px; margin-bottom: 24px; text-align: center; }}
.sidebar-icon-disabled{{ color: #3a3a3a; }}

.page-wrap  {{ min-height: 100vh; background: #0d0d0f; display: flex; align-items: center; justify-content: center; padding: 40px 20px; }}
.auth-card  {{ background: #16161a; border: 1px solid #2a2a30; border-radius: 16px; padding: 48px 44px; width: 100%; max-width: 420px; }}
.logo-row   {{ display: flex; align-items: center; gap: 10px; margin-bottom: 32px; }}
.logo-hex   {{ width: 36px; height: 36px; background: #6c63ff; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 600; color: white; }}
.logo-name  {{ font-size: 20px; font-weight: 600; color: #f0f0f5; }}
.logo-tag   {{ font-size: 11px; color: #6c63ff; font-weight: 500; letter-spacing: 0.6px; text-transform: uppercase; }}
.divider    {{ height: 1px; background: #2a2a30; margin-bottom: 28px; }}
.section-label {{ font-size: 11px; font-weight: 500; letter-spacing: 0.8px; text-transform: uppercase; color: #555560; margin-bottom: 16px; }}
.body-text  {{ font-size: 14px; color: #9090a0; line-height: 1.6; margin: 0 0 24px; }}
.caption    {{ font-size: 12px; color: #555560; margin: 16px 0 0; }}
.btn-primary {{ display: block; width: 100%; background: #6c63ff; color: #ffffff !important; text-decoration: none; padding: 12px 0; border-radius: 8px; font-size: 14px; font-weight: 500; text-align: center; box-sizing: border-box; }}
.btn-primary:hover {{ background: #5a52e0; }}
.status-pill {{ display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; }}
.status-pill.success {{ background: #0d2b1a; color: #4ade80; border: 1px solid #1a4a2a; }}
.status-pill.error   {{ background: #2b0d0d; color: #f87171; border: 1px solid #4a1a1a; }}
.dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
.token-preview {{ background: #0d0d0f; border: 1px solid #2a2a30; border-radius: 8px; padding: 10px 14px; font-family: 'Courier New', monospace; font-size: 13px; color: #6c63ff; margin: 16px 0 8px; letter-spacing: 0.5px; }}
.error-box {{ background: #1a0808; border: 1px solid #4a1a1a; border-radius: 8px; padding: 12px 14px; font-family: 'Courier New', monospace; font-size: 12px; color: #f87171; margin: 16px 0; word-break: break-word; }}
.feature-row  {{ display: flex; flex-direction: column; gap: 10px; margin-top: 28px; padding-top: 24px; border-top: 1px solid #2a2a30; }}
.feature-item {{ display: flex; align-items: flex-start; gap: 10px; }}
.feature-icon {{ width: 28px; height: 28px; background: #1e1e28; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #6c63ff; }}
.feature-text {{ font-size: 12px; color: #555560; line-height: 1.5; }}
.feature-text strong {{ color: #9090a0; font-weight: 500; }}

.page-title {{ font-size: 22px; font-weight: 600; color: #f0f0f5; letter-spacing: -0.4px; margin: 0 0 4px; }}
.page-sub   {{ font-size: 12px; color: #555560; margin: 0 0 32px; }}

.stat-card     {{ background: #16161a; border: 1px solid #2a2a30; border-radius: 12px; padding: 18px 20px; height: 110px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between; }}
.stat-label    {{ font-size: 11px; font-weight: 500; letter-spacing: 0.7px; text-transform: uppercase; color: #555560; }}
.stat-value    {{ font-size: 20px; font-weight: 600; color: #f0f0f5; letter-spacing: -0.3px; }}
.stat-value-sm {{ font-size: 14px; font-weight: 500; color: #f0f0f5; }}
.stat-meta     {{ font-size: 12px; color: #555560; }}
.stat-badge    {{ display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 500; width: fit-content; }}
.badge-green   {{ background: #0d2b1a; color: #4ade80; border: 1px solid #1a4a2a; }}
.badge-amber   {{ background: #2a2010; color: #fbbf24; border: 1px solid #4a3a10; }}
.badge-gray    {{ background: #1e1e28; color: #9090a0; border: 1px solid #2a2a30; }}

.section-header {{ font-size: 11px; font-weight: 500; letter-spacing: 0.7px; text-transform: uppercase; color: #555560; margin: 32px 0 14px; padding-bottom: 10px; border-bottom: 1px solid #2a2a30; }}

.portfolio-table {{ background: #16161a; border: 1px solid #2a2a30; border-radius: 12px; padding: 0 20px; }}
.portfolio-row   {{ display: flex; align-items: center; justify-content: space-between; padding: 11px 0; border-bottom: 1px solid #1e1e28; }}
.portfolio-row:last-child {{ border-bottom: none; }}
.stock-symbol    {{ font-size: 13px; font-weight: 500; color: #f0f0f5; min-width: 90px; }}
.stock-sector    {{ font-size: 10px; font-weight: 500; padding: 2px 7px; border-radius: 10px; white-space: nowrap; }}
.stock-qty       {{ font-size: 11px; color: #555560; margin-top: 2px; }}
.stock-price     {{ font-size: 13px; font-weight: 500; color: #f0f0f5; text-align: right; }}
.stock-change-pos {{ font-size: 12px; color: #4ade80; text-align: right; }}
.stock-change-neg {{ font-size: 12px; color: #f87171; text-align: right; }}
.stock-change-neu {{ font-size: 12px; color: #9090a0; text-align: right; }}
.stock-pnl-pos   {{ font-size: 11px; color: #4ade80; text-align: right; }}
.stock-pnl-neg   {{ font-size: 11px; color: #f87171; text-align: right; }}
.stock-pnl-neu   {{ font-size: 11px; color: #9090a0; text-align: right; }}

.hedge-panel {{ background: #16161a; border: 1px solid #2a2a30; border-radius: 12px; padding: 24px; }}
.hedge-placeholder {{ display: flex; align-items: center; justify-content: center; height: 60px; color: #3a3a3a; font-size: 13px; font-family: 'Inter', sans-serif; }}


.mode-card {{
    background: #16161a;
    border: 1px solid #2a2a30;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 4px 0 24px;
    color: #9090a0;
    font-size: 13px;
}}
.mode-card strong {{
    color: #f0f0f5;
    font-weight: 600;
}}
.mode-pill {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 12px;
    border-radius: 18px;
    background: #1e1e28;
    border: 1px solid #2a2a30;
    color: #6c63ff;
    font-size: 12px;
    margin-right: 8px;
}}
.placeholder-title {{
    font-size: 15px;
    font-weight: 600;
    color: #f0f0f5;
    margin-bottom: 8px;
}}
.placeholder-text {{
    font-size: 13px;
    color: #9090a0;
    line-height: 1.6;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid #2a2a30;
    margin-bottom: 20px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: #9090a0;
    border-radius: 8px 8px 0 0;
    padding: 10px 16px;
}}
.stTabs [aria-selected="true"] {{
    color: #f0f0f5 !important;
    border-bottom: 3px solid #6c63ff !important;
}}

</style>
"""


# â”€â”€ Auth HTML â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

AUTH_CARD_TOP = """
<div class="page-wrap"><div class="auth-card">
  <div class="logo-row">
    <div class="logo-hex">H</div>
    <div><div class="logo-name">Hedra</div><div class="logo-tag">Risk Intelligence Platform</div></div>
  </div>
  <div class="divider"></div>
  <div class="section-label">Kite Connect</div>
"""

AUTH_CARD_BOTTOM = """
  <div class="feature-row">
    <div class="feature-item">
      <div class="feature-icon">&#8599;</div>
      <div class="feature-text"><strong>LSTM-GARCH prediction</strong><br>Three-class risk classification on Nifty 50 data</div>
    </div>
    <div class="feature-item">
      <div class="feature-icon">&#9638;</div>
      <div class="feature-text"><strong>PDDL planning</strong><br>Constraint-aware hedge action sequences</div>
    </div>
    <div class="feature-item">
      <div class="feature-icon">&#9678;</div>
      <div class="feature-text"><strong>SHAP explainability</strong><br>Feature attribution for every decision</div>
    </div>
  </div>
</div></div>
"""


# â”€â”€ Kite helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def exchange_and_save_token(request_token):
    kite         = KiteConnect(api_key=API_KEY)
    session      = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = session["access_token"]
    set_key(ENV_PATH, "KITE_ACCESS_TOKEN", access_token)
    return access_token


def get_kite_client():
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))
    return kite


def is_token_present():
    return bool(os.getenv("KITE_ACCESS_TOKEN"))


def verify_token(kite):
    try:
        return True, kite.profile()
    except Exception as e:
        return False, str(e)


# â”€â”€ Auth page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def render_auth_page(state="login", token=None, error=None):
    st.markdown(get_css(auth_page=True), unsafe_allow_html=True)
    kite      = KiteConnect(api_key=API_KEY)
    login_url = kite.login_url()

    if state == "success":
        preview = token[:8] + "&#8226;&#8226;&#8226;&#8226;&#8226;&#8226;" + token[-4:]
        body = (
            "<div class='status-pill success'><span class='dot'></span> Authenticated</div>"
            "<p class='body-text' style='margin-top:12px;'>Access token saved successfully.</p>"
            "<div class='token-preview'>" + preview + "</div>"
            "<p class='caption'>Valid until 6:00 AM IST &middot; Saved to .env</p>"
        )
    elif state == "error":
        safe = escape(str(error))
        body = (
            "<div class='status-pill error'><span class='dot'></span> Authentication failed</div>"
            "<div class='error-box'>" + safe + "</div>"
            "<a href='" + login_url + "' class='btn-primary' style='margin-top:16px;'>Try again</a>"
        )
    else:
        body = (
            "<p class='body-text'>Connect your Zerodha account to begin. "
            "The access token is generated automatically and stored locally in your .env file.</p>"
            "<a href='" + login_url + "' class='btn-primary'>Connect with Kite</a>"
            "<p class='caption'>Tokens expire daily at 6 AM IST and must be refreshed each session.</p>"
        )

    st.markdown(AUTH_CARD_TOP + body + AUTH_CARD_BOTTOM, unsafe_allow_html=True)


# â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def render_custom_sidebar(profile, token_valid=True):
    name  = escape(profile.get("user_name", "â€”")) if profile else "â€”"
    email = escape(profile.get("email", "â€”"))     if profile else "â€”"

    token_raw = os.getenv("KITE_ACCESS_TOKEN", "")
    token_preview = (
        escape(token_raw[:8]) + "&#8226;&#8226;&#8226;&#8226;" + escape(token_raw[-4:])
        if token_raw
        else "Unavailable"
    )
    token_state = "Token active" if token_valid and token_raw else "Token unavailable"
    capital_text = "&#8377;" + "{:,.0f}".format(TOTAL_CAPITAL)

    if st.button("\u2630", key="sidebar_toggle"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

    if st.session_state.sidebar_open:
        sidebar_html = (
            "<div class='custom-sidebar'>"
            "<div class='sidebar-logo'>"
            "<div class='sidebar-logo-box'>H</div>"
            "<div><div class='sidebar-title'>Hedra</div>"
            "<div class='sidebar-subtitle'>Risk Intelligence Platform</div></div>"
            "</div>"
            "<div class='sidebar-line'></div>"
            "<div class='sidebar-label'>Account</div>"
            "<div class='sidebar-name'>" + name + "</div>"
            "<div class='sidebar-email'>" + email + "</div>"
            "<div class='sidebar-pill'><span class='sidebar-dot'></span><span>Token active</span></div>"
            "<div class='sidebar-line'></div>"
            "<div class='sidebar-label'>Session</div>"
            "<div class='sidebar-page'>Kite connection active</div>"
            "<div class='sidebar-page'>Access token: " + token_preview + "</div>"
            "<div class='sidebar-page'>" + token_state + "</div>"
            "<div class='sidebar-page'>Refresh before 6:00 AM IST expiry</div>"
            "<div class='sidebar-line'></div>"
            "<div class='sidebar-label'>Capital Setup</div>"
            "<div class='sidebar-page'>Total capital: " + capital_text + "</div>"
            "<div class='sidebar-page'>Stocks 80% &middot; Margin reserve 20%</div>"
            "<div class='sidebar-line'></div>"
            "<div class='sidebar-label'>System Modules</div>"
            "<div class='sidebar-page'>Kite market data</div>"
            "<div class='sidebar-page'>GJR-GARCH ML signal</div>"
            "<div class='sidebar-page'>Planner backend connected</div>"
            "<div class='sidebar-page'>Futures execution layer</div>"
            "</div>"
        )
    else:
        sidebar_html = (
            "<div class='custom-sidebar'>"
            "<div class='sidebar-logo-box' style='margin-bottom:20px;'>H</div>"
            "<div class='sidebar-line'></div>"
            "<div class='sidebar-icon-item' title='Live monitoring'>&#9679;</div>"
            "<div class='sidebar-icon-item' title='Historical backtesting'>&#9638;</div>"
            "<div class='sidebar-icon-item' title='Kite market data'>&#8599;</div>"
            "<div class='sidebar-icon-item' title='Planner backend'>&#9678;</div>"
            "<div class='sidebar-icon-item' title='Futures execution'>&#9711;</div>"
            "</div>"
        )

    st.markdown(sidebar_html, unsafe_allow_html=True)


# â”€â”€ Sector pie chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def render_sector_pie(sector_totals):
    sectors = list(sector_totals.keys())
    counts  = list(sector_totals.values())
    colours = [SECTOR_COLOURS.get(s, "#555560") for s in sectors]
    total   = sum(counts)

    cx, cy, r = 110, 110, 85
    slices    = []
    start     = -math.pi / 2

    for i, (sector, count) in enumerate(zip(sectors, counts)):
        angle     = (count / total) * 2 * math.pi
        end       = start + angle
        x1        = cx + r * math.cos(start)
        y1        = cy + r * math.sin(start)
        x2        = cx + r * math.cos(end)
        y2        = cy + r * math.sin(end)
        large     = 1 if angle > math.pi else 0
        mid_angle = start + angle / 2
        lx        = cx + (r * 0.62) * math.cos(mid_angle)
        ly        = cy + (r * 0.62) * math.sin(mid_angle)
        path      = (
            f"M {cx} {cy} L {x1:.2f} {y1:.2f} "
            f"A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f} Z"
        )
        slices.append((path, colours[i], sector, count, lx, ly))
        start = end

    pie_svg = ""
    for path, col, sector, count, lx, ly in slices:
        pie_svg += (
            f"<path d='{path}' fill='{col}' opacity='0.85' "
            f"stroke='#0d0d0f' stroke-width='2'/>"
        )
        if count >= 2:
            pie_svg += (
                f"<text x='{lx:.1f}' y='{ly:.1f}' text-anchor='middle' "
                f"dominant-baseline='middle' fill='white' "
                f"font-size='11' font-family='Inter,sans-serif' font-weight='600'>"
                f"{count}</text>"
            )

    legend = ""
    for i, (sector, count) in enumerate(zip(sectors, counts)):
        col = colours[i]
        pct = count / total * 100
        legend += (
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>"
            f"<span style='width:10px;height:10px;border-radius:2px;background:{col};"
            f"display:inline-block;flex-shrink:0;'></span>"
            f"<span style='font-size:12px;color:#9090a0;font-family:Inter,sans-serif;'>{sector}</span>"
            f"<span style='font-size:12px;color:#555560;font-family:Inter,sans-serif;"
            f"margin-left:auto;'>{pct:.0f}%</span>"
            f"</div>"
        )

    st.markdown(
        "<div style='background:#16161a;border:1px solid #2a2a30;border-radius:12px;"
        "padding:20px 24px;box-sizing:border-box;margin-top:12px;'>"
        "<p style='font-size:11px;font-weight:500;letter-spacing:0.7px;text-transform:uppercase;"
        "color:#555560;font-family:Inter,sans-serif;margin:0 0 16px;'>Sector Distribution</p>"
        "<div style='display:flex;align-items:center;gap:20px;'>"
        f"<svg width='220' height='220' viewBox='0 0 220 220'>{pie_svg}</svg>"
        f"<div style='flex:1;'>{legend}</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# â”€â”€ Dashboard home â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def render_placeholder_section(title, text):
    st.markdown(
        "<div class='hedge-panel'>"
        "<div class='placeholder-title'>" + escape(title) + "</div>"
        "<div class='placeholder-text'>" + escape(text) + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )



def _parse_kite_datetime(value):
    if value is None:
        return None

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None

    if parsed.tzinfo is None:
        return parsed.to_pydatetime().replace(tzinfo=IST)

    return parsed.tz_convert(IST).to_pydatetime()


def get_kite_market_status(kite, quote_symbol="NSE:NIFTY 50", freshness_minutes=20):
    """Determine market/feed status using actual Kite quote timestamps.

    The live intraday signal is enabled only when:
    1. Current IST time is inside the regular NSE cash-market window.
    2. Kite quote timestamp is from today.
    3. The quote timestamp is recent enough.
    """

    now = datetime.now(IST)
    market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

    is_weekday = now.weekday() < 5
    is_regular_session = is_weekday and market_open_time <= now <= market_close_time

    quote_timestamp = None
    last_price = None
    feed_error = None

    try:
        quote = kite.quote([quote_symbol]).get(quote_symbol, {})
        quote_timestamp = _parse_kite_datetime(
            quote.get("last_trade_time") or quote.get("timestamp")
        )
        last_price = quote.get("last_price")
    except Exception as exc:
        feed_error = str(exc)

    quote_age_minutes = None
    quote_is_today = False
    feed_is_fresh = False

    if quote_timestamp is not None:
        quote_age_minutes = max((now - quote_timestamp).total_seconds() / 60.0, 0.0)
        quote_is_today = quote_timestamp.date() == now.date()
        feed_is_fresh = quote_is_today and quote_age_minutes <= freshness_minutes

    intraday_enabled = is_regular_session and feed_is_fresh

    if intraday_enabled:
        status = "Open"
        signal_label = "Live intraday signal active"
        guidance = "Live intraday prediction is provisional and should be confirmed after market close."
    elif not is_regular_session:
        status = "Closed"
        signal_label = "Live intraday signal disabled"
        guidance = "Market is outside regular NSE trading hours. Use the latest confirmed end-of-day hedge plan."
    elif feed_error:
        status = "Feed unavailable"
        signal_label = "Live intraday signal disabled"
        guidance = "Kite quote data could not be fetched, so the intraday signal is disabled."
    else:
        status = "Feed stale"
        signal_label = "Live intraday signal disabled"
        guidance = "Kite quote data is not fresh enough for an intraday hedge signal."

    return {
        "status": status,
        "is_open": intraday_enabled,
        "intraday_enabled": intraday_enabled,
        "signal_label": signal_label,
        "guidance": guidance,
        "quote_symbol": quote_symbol,
        "last_price": last_price,
        "quote_timestamp": quote_timestamp.strftime("%Y-%m-%d %H:%M:%S IST") if quote_timestamp else "Unavailable",
        "quote_age_minutes": quote_age_minutes,
        "current_market_date": now.strftime("%Y-%m-%d"),
        "timestamp": now.strftime("%d %b %Y, %H:%M IST"),
        "feed_error": feed_error,
    }




def render_market_status_panel(kite):
    status = get_kite_market_status(kite)

    status_colour = "#4ade80" if status["intraday_enabled"] else "#fbbf24"
    signal_colour = "#4ade80" if status["intraday_enabled"] else "#9090a0"

    quote_age = status.get("quote_age_minutes")
    quote_age_text = "Unavailable" if quote_age is None else f"{quote_age:.1f} minutes"

    last_price = status.get("last_price")
    last_price_text = "Unavailable" if last_price is None else f"₹{float(last_price):,.2f}"

    st.markdown(
        "<div class='section-header'>Market Status</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Market Status</div>"
        "<div class='stat-value-sm' style='color:" + status_colour + ";'>"
        + escape(status["status"]) +
        "</div>"
        "<div class='stat-meta'>" + escape(status["timestamp"]) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Live NIFTY Quote</div>"
        "<div class='stat-value-sm'>" + escape(last_price_text) + "</div>"
        "<div class='stat-meta'>" + escape(status["quote_symbol"]) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Quote Timestamp</div>"
        "<div class='stat-value-sm'>" + escape(status["quote_timestamp"]) + "</div>"
        "<div class='stat-meta'>Quote age: " + escape(quote_age_text) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Intraday Signal</div>"
        "<div class='stat-value-sm' style='color:" + signal_colour + ";'>"
        + escape(status["signal_label"]) +
        "</div>"
        "<div class='stat-meta'>"
        + ("Enabled" if status["intraday_enabled"] else "Disabled") +
        "</div>"
        "</div>"

        "</div>"
        "<div style='height:1px;background:#2a2a30;margin:20px 0;'></div>"
        "<div class='placeholder-text'>"
        + escape(status["guidance"]) +
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )



def render_live_overview_summary():
    st.markdown(
        "<div class='section-header'>System Summary</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Operating Mode</div>"
        "<div class='stat-value-sm'>Live Monitoring</div>"
        "<div class='stat-meta'>Kite market data active</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Capital Model</div>"
        "<div class='stat-value-sm'>&#8377;" + "{:,.0f}".format(TOTAL_CAPITAL) + "</div>"
        "<div class='stat-meta'>80% stocks &middot; 20% margin</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Planner Status</div>"
        "<div class='stat-value-sm'>Connected</div>"
        "<div class='stat-meta'>Unified Planning + ENHSP</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Hedge Engine</div>"
        "<div class='stat-value-sm'>Ready</div>"
        "<div class='stat-meta'>NIFTY futures execution layer</div>"
        "</div>"

        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )



def render_live_decision_summary(kite):
    st.markdown(
        "<div class='section-header'>Confirmed EOD Hedge Plan</div>",
        unsafe_allow_html=True,
    )

    if "live_hedge_plan_cache" not in st.session_state:
        with st.spinner("Preparing latest live hedge decision ..."):
            try:
                snapshot, item, result, inputs = build_live_hedge_plan(kite)
                st.session_state.live_hedge_plan_cache = {
                    "snapshot": snapshot,
                    "item": item,
                    "result": result,
                    "inputs": inputs,
                    "error": None,
                }
            except Exception as exc:
                st.session_state.live_hedge_plan_cache = {
                    "snapshot": None,
                    "item": None,
                    "result": None,
                    "inputs": None,
                    "error": str(exc),
                }

    cache = st.session_state.live_hedge_plan_cache

    if cache.get("error"):
        st.markdown(
            "<div style='background:#1a0808;border:1px solid #4a1a1a;border-radius:8px;"
            "padding:14px 16px;font-size:13px;color:#f87171;'>"
            "Live decision summary unavailable: " + escape(cache["error"]) + "</div>",
            unsafe_allow_html=True,
        )
        return

    snapshot = cache["snapshot"]
    result = cache["result"]
    inputs = cache["inputs"]
    execution = result.futures_execution_result

    action = ui_label(result.position_action)
    instruction = ui_label(result.execution_instruction)
    risk_regime = str(inputs.get("risk_regime", "N/A")).title()
    target_ratio = float(result.execution_target_hedge_ratio)
    achieved_ratio = float(get_result_field(execution, "achieved_hedge_ratio", 0.0))
    target_contracts = int(result.recommended_short_contracts)
    current_contracts = int(result.current_short_contracts)
    trade_now = int(result.contract_delta)
    margin_required = get_result_field(execution, "additional_margin_required", 0.0)
    margin_status = "Sufficient" if result.margin_sufficient else "Insufficient"
    margin_colour = "#4ade80" if result.margin_sufficient else "#f87171"

    trade_text = "No trade" if trade_now == 0 else f"{trade_now:+d} contracts"

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Planner Action</div>"
        "<div class='stat-value-sm'>" + escape(action) + "</div>"
        "<div class='stat-meta'>" + escape(instruction) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Risk Regime</div>"
        "<div class='stat-value-sm'>" + escape(risk_regime) + "</div>"
        "<div class='stat-meta'>Predicted volatility: " + f"{inputs.get('predicted_volatility', 0.0):.2%}" + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Hedra Target Ratio</div>"
        "<div class='stat-value-sm'>" + f"{target_ratio:.2%}" + "</div>"
        "<div class='stat-meta'>Achieved: " + f"{achieved_ratio:.2%}" + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Target Position</div>"
        "<div class='stat-value-sm'>" + str(target_contracts) + " short FUT</div>"
        "<div class='stat-meta'>Current: " + str(current_contracts) + " &middot; Trade now: " + escape(trade_text) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Margin Status</div>"
        "<div class='stat-value-sm' style='color:" + margin_colour + ";'>" + margin_status + "</div>"
        "<div class='stat-meta'>Required: " + format_money_html(margin_required, 0) + "</div>"
        "</div>"

        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )




def render_intraday_signal_panel(kite):
    status = get_kite_market_status(kite)

    if status["intraday_enabled"]:
        panel_title = "Live Intraday Signal"
        panel_value = "Active"
        panel_colour = "#4ade80"
        guidance = (
            "The live intraday signal is active because the market feed is fresh. "
            "This signal is provisional and should be confirmed after market close."
        )
    else:
        panel_title = "Live Intraday Signal"
        panel_value = "Disabled"
        panel_colour = "#9090a0"
        guidance = status["guidance"]

    st.markdown(
        "<div class='section-header'>" + escape(panel_title) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hedge-panel' style='opacity:" + ("1.0" if status["intraday_enabled"] else "0.62") + ";'>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Intraday Status</div>"
        "<div class='stat-value-sm' style='color:" + panel_colour + ";'>"
        + escape(panel_value) +
        "</div>"
        "<div class='stat-meta'>" + escape(status["signal_label"]) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Market Feed</div>"
        "<div class='stat-value-sm'>" + escape(status["status"]) + "</div>"
        "<div class='stat-meta'>" + escape(status["quote_timestamp"]) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Live Price Source</div>"
        "<div class='stat-value-sm'>" + escape(status["quote_symbol"]) + "</div>"
        "<div class='stat-meta'>Kite quote feed</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Signal Role</div>"
        "<div class='stat-value-sm'>Provisional</div>"
        "<div class='stat-meta'>Not final until EOD close</div>"
        "</div>"

        "</div>"
        "<div style='height:1px;background:#2a2a30;margin:20px 0;'></div>"
        "<div class='placeholder-text'>"
        + escape(guidance) +
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )





def _load_market_price_frame(csv_path, label):
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)

        if "date" not in df.columns or "close" not in df.columns:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")

        if df.empty:
            return pd.DataFrame()

        return df[["date", "close"]].rename(columns={"close": label}).set_index("date")

    except Exception:
        return pd.DataFrame()


def _market_dataset_summary(csv_path, dataset_name, source_note):
    if not os.path.exists(csv_path):
        return {
            "Dataset": dataset_name,
            "Source": source_note,
            "Status": "Missing",
            "Rows": 0,
            "Start Date": "N/A",
            "End Date": "N/A",
            "Latest Close": "N/A",
        }

    try:
        df = pd.read_csv(csv_path)

        if "date" not in df.columns:
            return {
                "Dataset": dataset_name,
                "Source": source_note,
                "Status": "Loaded, date column missing",
                "Rows": len(df),
                "Start Date": "N/A",
                "End Date": "N/A",
                "Latest Close": "N/A",
            }

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        latest_close = "N/A"
        if "close" in df.columns and not df.empty:
            close_series = pd.to_numeric(df["close"], errors="coerce").dropna()
            if not close_series.empty:
                latest_close = "₹" + f"{float(close_series.iloc[-1]):,.2f}"

        return {
            "Dataset": dataset_name,
            "Source": source_note,
            "Status": "Available",
            "Rows": len(df),
            "Start Date": df["date"].min().strftime("%Y-%m-%d") if not df.empty else "N/A",
            "End Date": df["date"].max().strftime("%Y-%m-%d") if not df.empty else "N/A",
            "Latest Close": latest_close,
        }

    except Exception as exc:
        return {
            "Dataset": dataset_name,
            "Source": source_note,
            "Status": "Load error: " + str(exc),
            "Rows": 0,
            "Start Date": "N/A",
            "End Date": "N/A",
            "Latest Close": "N/A",
        }



def render_eod_data_update_panel(kite):
    st.markdown(
        "<div class='section-header'>EOD Data Update</div>",
        unsafe_allow_html=True,
    )

    market_status = get_kite_market_status(kite)
    market_is_open = bool(market_status.get("intraday_enabled"))

    if market_is_open:
        status_message = (
            "The live feed appears to be active. EOD update should be run only after market close, "
            "because intraday prices are provisional."
        )
        status_colour = "#fbbf24"
    else:
        status_message = (
            "Market is closed or the feed is stale. You can update the confirmed EOD market data "
            "for the latest completed trading day."
        )
        status_colour = "#4ade80"

    # Default to the latest completed market-data date, not the next calendar/trading day.
    # This is safer for EOD updates because the current day may not yet have a confirmed close.
    default_date = datetime.now(IST).date()

    quote_timestamp_raw = str(market_status.get("quote_timestamp") or "").strip()
    quote_timestamp_clean = (
        quote_timestamp_raw
        .replace(" IST", "")
        .replace(" UTC", "")
        .replace(" GMT", "")
        .strip()
    )

    quote_dt = pd.to_datetime(quote_timestamp_clean, errors="coerce")

    if pd.notna(quote_dt):
        default_date = quote_dt.date()

    default_futures_symbol = "NIFTY26JUNFUT"

    try:
        cache = st.session_state.get("live_hedge_plan_cache", {})
        snapshot = cache.get("snapshot") if isinstance(cache, dict) else None

        if snapshot is None:
            snapshot = get_live_broker_snapshot(kite)

        contract = getattr(snapshot, "contract", None)
        detected_symbol = getattr(contract, "tradingsymbol", None)

        if detected_symbol:
            default_futures_symbol = str(detected_symbol)
    except Exception:
        pass

    st.markdown(
        "<div class='hedge-panel'>"
        "<div class='placeholder-title'>Confirmed EOD data refresh</div>"
        "<div class='placeholder-text'>"
        "Use this after the market has closed to fetch confirmed NIFTY spot and NIFTY futures daily candles "
        "from Kite and update the local CSV files used by the volatility model and dashboard."
        "</div>"
        "<div style='height:1px;background:#2a2a30;margin:18px 0;'></div>"
        "<div class='placeholder-text' style='color:" + status_colour + ";'>"
        + escape(status_message) +
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("eod_update_form"):
        col_date, col_symbol = st.columns([1, 1])

        with col_date:
            eod_date = st.date_input(
                "EOD date to update",
                value=default_date,
                key="eod_update_date",
            )

        with col_symbol:
            futures_symbol = st.text_input(
                "NIFTY futures symbol",
                value=default_futures_symbol,
                key="eod_update_futures_symbol",
                help="Auto-detected from the current live hedge snapshot. Edit only if the detected contract is incorrect.",
            )

        button_disabled = market_is_open or not str(futures_symbol).strip()

        run_update = st.form_submit_button(
            "Update confirmed EOD data",
            disabled=button_disabled,
        )

    if market_is_open:
        st.caption("Button disabled while the market feed is active. Run after market close.")

    if run_update:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script_path = os.path.join(project_root, "scripts", "update_eod_market_data.py")

        if not os.path.exists(script_path):
            st.error("EOD updater script not found: " + script_path)
            return

        command = [
            sys.executable,
            script_path,
            "--date",
            eod_date.strftime("%Y-%m-%d"),
            "--futures-symbol",
            str(futures_symbol).strip(),
        ]

        with st.spinner("Fetching confirmed EOD market data from Kite and updating local CSV files ..."):
            completed = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=180,
            )

        if completed.returncode == 0:
            st.success("EOD market data updated successfully.")

            st.session_state.pop("live_hedge_plan_cache", None)

            with st.expander("Updater output", expanded=False):
                st.code(completed.stdout or "No output returned.", language="text")

            st.info("Restart or refresh the dashboard to reload the updated CSV data into all tabs.")
        else:
            st.error("EOD update failed.")

            with st.expander("Error details", expanded=True):
                st.code((completed.stderr or completed.stdout or "No error details returned."), language="text")



def render_market_data_content(kite, mode):
    st.markdown(
        "<div class='section-header'>Market Data</div>",
        unsafe_allow_html=True,
    )

    status = get_kite_market_status(kite)

    live_futures_symbol = "N/A"
    live_futures_price = "N/A"
    live_futures_lot = "N/A"

    try:
        if "live_hedge_plan_cache" not in st.session_state:
            snapshot, item, result, inputs = build_live_hedge_plan(kite)
            st.session_state.live_hedge_plan_cache = {
                "snapshot": snapshot,
                "item": item,
                "result": result,
                "inputs": inputs,
                "error": None,
            }

        cache = st.session_state.get("live_hedge_plan_cache", {})
        if not cache.get("error"):
            snapshot = cache.get("snapshot")
            contract = getattr(snapshot, "contract", None)

            live_futures_symbol = escape(str(getattr(contract, "tradingsymbol", "N/A")))
            live_futures_lot = escape(str(getattr(contract, "lot_size", "N/A")))

            futures_price = getattr(snapshot, "futures_price", None)
            if futures_price is not None:
                live_futures_price = "₹" + f"{float(futures_price):,.2f}"

    except Exception:
        pass

    last_price = status.get("last_price")
    last_price_text = "Unavailable" if last_price is None else "₹" + f"{float(last_price):,.2f}"

    quote_age = status.get("quote_age_minutes")
    quote_age_text = "Unavailable" if quote_age is None else f"{quote_age:.1f} minutes"

    status_colour = "#4ade80" if status["intraday_enabled"] else "#fbbf24"

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Market Status</div>"
        "<div class='stat-value-sm' style='color:" + status_colour + ";'>" + escape(str(status["status"])) + "</div>"
        "<div class='stat-meta'>" + escape(str(status["timestamp"])) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>NIFTY Spot Quote</div>"
        "<div class='stat-value-sm'>" + escape(last_price_text) + "</div>"
        "<div class='stat-meta'>" + escape(str(status["quote_symbol"])) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Quote Freshness</div>"
        "<div class='stat-value-sm'>" + escape(quote_age_text) + "</div>"
        "<div class='stat-meta'>" + escape(str(status["quote_timestamp"])) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Active Futures Contract</div>"
        "<div class='stat-value-sm'>" + live_futures_symbol + "</div>"
        "<div class='stat-meta'>Price " + escape(live_futures_price) + " &middot; Lot " + live_futures_lot + "</div>"
        "</div>"

        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    spot_path = os.path.join("data", "raw", "NIFTY50_spot.csv")
    futures_path = os.path.join("data", "raw", "NIFTY_futures_continuous.csv")

    summary_df = pd.DataFrame([
        _market_dataset_summary(spot_path, "NIFTY 50 Spot", "Local historical spot CSV"),
        _market_dataset_summary(futures_path, "NIFTY Futures Continuous", "Local historical futures CSV"),
    ])

    st.markdown(
        "<div class='section-header'>Historical Data Availability</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(summary_df, hide_index=True, width="stretch")

    spot_frame = _load_market_price_frame(spot_path, "NIFTY 50 Spot Close")
    futures_frame = _load_market_price_frame(futures_path, "NIFTY Futures Close")

    chart_parts = []
    if not spot_frame.empty:
        chart_parts.append(spot_frame.tail(180))
    if not futures_frame.empty:
        chart_parts.append(futures_frame.tail(180))

    if chart_parts:
        chart_df = pd.concat(chart_parts, axis=1).dropna(how="all")

        st.markdown(
            "<div class='section-header'>Recent Index and Futures Close</div>",
            unsafe_allow_html=True,
        )

        plotted = chart_df.dropna(how="all")

        if not plotted.empty:
            values = plotted.stack().dropna()

            if not values.empty:
                y_min = float(values.min())
                y_max = float(values.max())

                # Tight dynamic range so the chart shows movement clearly
                data_range = y_max - y_min
                padding = max(data_range * 0.12, y_max * 0.003)

                fig, ax = plt.subplots(figsize=(13, 4.2))

                if "NIFTY 50 Spot Close" in plotted.columns:
                    ax.plot(
                        plotted.index,
                        plotted["NIFTY 50 Spot Close"],
                        label="NIFTY 50 Spot Close",
                        linewidth=2.2,
                        color="#60a5fa",
                    )

                if "NIFTY Futures Close" in plotted.columns:
                    ax.plot(
                        plotted.index,
                        plotted["NIFTY Futures Close"],
                        label="NIFTY Futures Close",
                        linewidth=2.2,
                        linestyle="--",
                        color="#f59e0b",
                    )

                ax.set_ylim(y_min - padding, y_max + padding)
                ax.set_xlabel("")
                ax.set_ylabel("Close Price")

                fig.patch.set_facecolor("#0f1014")
                ax.set_facecolor("#0f1014")

                ax.grid(True, alpha=0.22)
                ax.tick_params(axis="x", colors="#b8b8c2", labelsize=9)
                ax.tick_params(axis="y", colors="#b8b8c2", labelsize=9)
                ax.yaxis.label.set_color("#b8b8c2")

                for spine in ax.spines.values():
                    spine.set_color("#2a2a30")

                legend = ax.legend(loc="lower left", frameon=False)
                for text_item in legend.get_texts():
                    text_item.set_color("#d4d4dc")

                fig.autofmt_xdate(rotation=0)
                st.pyplot(fig, width="stretch")
                plt.close(fig)




    render_eod_data_update_panel(kite)


def humanise_signal_source(source):
    if not source:
        return "Volatility signal source not available"

    mapping = {
        "latest_available_nifty_spot_gjr_garch": "Latest available NIFTY spot history processed with GJR-GARCH",
        "live_intraday_nifty_spot_gjr_garch": "Live intraday NIFTY spot feed processed with GJR-GARCH",
        "latest_available_nifty_spot_garch": "Latest available NIFTY spot history processed with GARCH",
        "latest_available_nifty_spot_egarch": "Latest available NIFTY spot history processed with EGARCH",
    }

    source = str(source).strip()

    if source in mapping:
        return mapping[source]

    return source.replace("_", " ").strip().capitalize()


def build_volatility_ruler_html(predicted_volatility, low_threshold, high_threshold, regime_colour, interpretation):
    predicted_volatility = float(predicted_volatility or 0.0)
    low_threshold = float(low_threshold or 0.0)
    high_threshold = float(high_threshold or 0.0)

    scale_max = max(predicted_volatility, low_threshold, high_threshold, 0.20) * 1.15

    def to_position(value):
        if scale_max <= 0:
            return 0.0
        return max(0.0, min(100.0, (float(value) / scale_max) * 100.0))

    low_position = to_position(low_threshold)
    high_position = to_position(high_threshold)
    prediction_position = to_position(predicted_volatility)

    return (
        "<div class='hedge-panel'>"

        "<div style='margin-bottom:10px;'>"
        "<div class='placeholder-title'>Volatility Signal Ruler</div>"
        "<div class='placeholder-text'>"
        "The model forecast is positioned against the low-risk and high-risk thresholds used by the hedge planner."
        "</div>"
        "</div>"

        "<div style='position:relative;height:18px;margin:34px 8px 58px 8px;"
        "background:linear-gradient(90deg,#2563eb 0%,#f59e0b 55%,#ef4444 100%);"
        "border:1px solid #2a2a30;border-radius:999px;'>"

        "<div style='position:absolute;left:0%;top:24px;font-size:11px;color:#9090a0;'>"
        "0.00%</div>"

        "<div style='position:absolute;right:0%;top:24px;font-size:11px;color:#9090a0;text-align:right;'>"
        "Scale max<br>" + f"{scale_max:.2%}" + "</div>"

        "<div style='position:absolute;left:" + f"{low_position:.2f}" + "%;top:-9px;width:2px;height:36px;"
        "background:#4ade80;border-radius:999px;'></div>"
        "<div style='position:absolute;left:" + f"{low_position:.2f}" + "%;top:32px;transform:translateX(-50%);"
        "font-size:11px;color:#b8b8c2;text-align:center;white-space:nowrap;'>"
        "Low threshold<br>" + f"{low_threshold:.2%}" + "</div>"

        "<div style='position:absolute;left:" + f"{high_position:.2f}" + "%;top:-9px;width:2px;height:36px;"
        "background:#fbbf24;border-radius:999px;'></div>"
        "<div style='position:absolute;left:" + f"{high_position:.2f}" + "%;top:32px;transform:translateX(-50%);"
        "font-size:11px;color:#b8b8c2;text-align:center;white-space:nowrap;'>"
        "High threshold<br>" + f"{high_threshold:.2%}" + "</div>"

        "<div style='position:absolute;left:" + f"{prediction_position:.2f}" + "%;top:-14px;width:4px;height:46px;"
        "background:" + regime_colour + ";border-radius:999px;box-shadow:0 0 12px " + regime_colour + ";'></div>"
        "<div style='position:absolute;left:" + f"{prediction_position:.2f}" + "%;top:-36px;transform:translateX(-50%);"
        "font-size:11px;color:#f0f0f5;text-align:center;white-space:nowrap;'>"
        "Current prediction<br>" + f"{predicted_volatility:.2%}" + "</div>"

        "</div>"

        "<div style='height:1px;background:#2a2a30;margin:18px 0;'></div>"

        "<div class='placeholder-text'>"
        + escape(interpretation) +
        "</div>"

        "<div style='height:1px;background:#2a2a30;margin:18px 0;'></div>"

        "<div class='placeholder-text'>"
        "<strong>Important note:</strong> this live prediction view currently uses the latest available local NIFTY spot history "
        "for the selected GJR-GARCH volatility signal. Live Kite quotes are shown in the Market Data and Overview tabs, "
        "while the intraday quote-updated prediction extension can be added later if time permits."
        "</div>"

        "</div>"
    )


def render_ml_prediction_content(kite, mode):
    st.markdown(
        "<div class='section-header'>ML Prediction</div>",
        unsafe_allow_html=True,
    )

    try:
        market_status = get_kite_market_status(kite)
        signal = get_live_volatility_signal(
            live_close=market_status.get("last_price"),
            live_timestamp=market_status.get("quote_timestamp"),
            use_live_price=bool(market_status.get("intraday_enabled")),
        )
    except Exception as exc:
        st.markdown(
            "<div class='hedge-panel'>"
            "<div class='placeholder-title'>Prediction unavailable</div>"
            "<div class='placeholder-text'>"
            "The volatility prediction layer could not be loaded: "
            + escape(str(exc)) +
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    predicted_volatility = float(signal.predicted_volatility)
    low_threshold = float(signal.low_threshold)
    high_threshold = float(signal.high_threshold)
    risk_regime = str(signal.risk_regime).upper()
    target_ratio = float(signal.core_target_ratio)
    latest_close = float(signal.latest_close)

    signal_source_label = humanise_signal_source(signal.source)
    live_quote_used = bool(getattr(signal, "live_quote_used", False))
    signal_mode_label = "Intraday provisional" if live_quote_used else "Confirmed historical close"
    signal_mode_note = getattr(signal, "data_freshness_note", "Latest confirmed historical close used")

    if risk_regime == "HIGH":
        regime_colour = "#f87171"
        regime_text = "High volatility regime"
        interpretation = (
            "The predicted volatility is above the high-risk threshold. "
            "The system therefore classifies the market state as HIGH risk and allows a stronger hedge target."
        )
    elif risk_regime == "MEDIUM":
        regime_colour = "#fbbf24"
        regime_text = "Medium volatility regime"
        interpretation = (
            "The predicted volatility is between the low and high thresholds. "
            "The system classifies the market state as MEDIUM risk and uses a moderate hedge target."
        )
    else:
        regime_colour = "#4ade80"
        regime_text = "Low volatility regime"
        interpretation = (
            "The predicted volatility is below the low-risk threshold. "
            "The system classifies the market state as LOW risk and keeps the hedge target conservative."
        )

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Active Model</div>"
        "<div class='stat-value-sm'>" + escape(str(signal.model)) + "</div>"
        "<div class='stat-meta'>" + escape(signal_source_label) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Predicted Volatility</div>"
        "<div class='stat-value-sm'>" + f"{predicted_volatility:.2%}" + "</div>"
        "<div class='stat-meta'>Annualised volatility estimate</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Risk Regime</div>"
        "<div class='stat-value-sm' style='color:" + regime_colour + ";'>" + escape(risk_regime.title()) + "</div>"
        "<div class='stat-meta'>" + escape(regime_text) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Core Hedge Target</div>"
        "<div class='stat-value-sm'>" + f"{target_ratio:.2%}" + "</div>"
        "<div class='stat-meta'>Before execution constraints</div>"
        "</div>"

        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    threshold_df = pd.DataFrame([
        {
            "Signal Component": "Latest model data date",
            "Value": str(signal.data_date),
            "Interpretation": "Most recent historical observation used by the prediction layer",
        },
        {
            "Signal Component": "Signal mode",
            "Value": signal_mode_label,
            "Interpretation": signal_mode_note,
        },
        {
            "Signal Component": "Latest close in model dataset",
            "Value": "₹" + f"{latest_close:,.2f}",
            "Interpretation": "Latest close used by the volatility prediction layer",
        },
        {
            "Signal Component": "Low-risk threshold",
            "Value": f"{low_threshold:.2%}",
            "Interpretation": "Below this level, volatility is treated as LOW",
        },
        {
            "Signal Component": "High-risk threshold",
            "Value": f"{high_threshold:.2%}",
            "Interpretation": "Above this level, volatility is treated as HIGH",
        },
        {
            "Signal Component": "Predicted volatility",
            "Value": f"{predicted_volatility:.2%}",
            "Interpretation": "Current model forecast used for risk-regime classification",
        },
        {
            "Signal Component": "Selected risk regime",
            "Value": risk_regime.title(),
            "Interpretation": "Final volatility regime passed to the hedge planner",
        },
    ])

    st.markdown(
        "<div class='section-header'>Model Signal Breakdown</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(threshold_df, hide_index=True, width="stretch")

    st.markdown(
        "<div class='section-header'>Volatility Signal Position</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        build_volatility_ruler_html(
            predicted_volatility=predicted_volatility,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            regime_colour=regime_colour,
            interpretation=interpretation,
        ),
        unsafe_allow_html=True,
    )


def render_overview_command_center(kite):
    status = get_kite_market_status(kite)

    if "live_hedge_plan_cache" not in st.session_state:
        with st.spinner("Preparing confirmed hedge decision ..."):
            try:
                snapshot, item, result, inputs = build_live_hedge_plan(kite)
                st.session_state.live_hedge_plan_cache = {
                    "snapshot": snapshot,
                    "item": item,
                    "result": result,
                    "inputs": inputs,
                    "error": None,
                }
            except Exception as exc:
                st.session_state.live_hedge_plan_cache = {
                    "snapshot": None,
                    "item": None,
                    "result": None,
                    "inputs": None,
                    "error": str(exc),
                }

    cache = st.session_state.live_hedge_plan_cache

    st.markdown(
        "<div class='section-header'>Market &amp; Hedge Overview</div>",
        unsafe_allow_html=True,
    )

    status_colour = "#4ade80" if status["intraday_enabled"] else "#fbbf24"
    quote_age = status.get("quote_age_minutes")
    quote_age_text = "Unavailable" if quote_age is None else f"{quote_age:.1f} minutes"
    last_price = status.get("last_price")
    last_price_text = "Unavailable" if last_price is None else f"₹{float(last_price):,.2f}"

    if cache.get("error"):
        decision_html = (
            "<div style='font-size:13px;color:#f87171;'>"
            "Confirmed hedge decision unavailable: " + escape(cache["error"]) + "</div>"
        )
    else:
        snapshot = cache["snapshot"]
        result = cache["result"]
        inputs = cache["inputs"]
        execution = result.futures_execution_result

        action = ui_label(result.position_action)
        instruction = ui_label(result.execution_instruction)
        risk_regime = str(inputs.get("risk_regime", "N/A")).title()
        predicted_volatility = float(inputs.get("predicted_volatility", 0.0))
        target_ratio = float(result.execution_target_hedge_ratio)
        achieved_ratio = float(get_result_field(execution, "achieved_hedge_ratio", 0.0))
        target_contracts = int(result.recommended_short_contracts)
        trade_now = int(result.contract_delta)
        margin_required = float(get_result_field(execution, "additional_margin_required", 0.0))
        margin_status = "Sufficient" if result.margin_sufficient else "Insufficient"
        margin_colour = "#4ade80" if result.margin_sufficient else "#f87171"

        trade_text = "No trade" if trade_now == 0 else f"{trade_now:+d} contracts"

        decision_html = (
            "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;'>"

            "<div>"
            "<div class='stat-label'>Planner Action</div>"
            "<div class='stat-value-sm'>" + escape(action) + "</div>"
            "<div class='stat-meta'>" + escape(instruction) + "</div>"
            "</div>"

            "<div>"
            "<div class='stat-label'>Risk Regime</div>"
            "<div class='stat-value-sm'>" + escape(risk_regime) + "</div>"
            "<div class='stat-meta'>Predicted volatility: " + f"{predicted_volatility:.2%}" + "</div>"
            "</div>"

            "<div>"
            "<div class='stat-label'>Hedra Target Ratio</div>"
            "<div class='stat-value-sm'>" + f"{target_ratio:.2%}" + "</div>"
            "<div class='stat-meta'>Achieved: " + f"{achieved_ratio:.2%}" + "</div>"
            "</div>"

            "<div>"
            "<div class='stat-label'>Target Position</div>"
            "<div class='stat-value-sm'>" + str(target_contracts) + " short FUT</div>"
            "<div class='stat-meta'>Trade now: " + escape(trade_text) + "</div>"
            "</div>"

            "<div>"
            "<div class='stat-label'>Margin Status</div>"
            "<div class='stat-value-sm' style='color:" + margin_colour + ";'>" + margin_status + "</div>"
            "<div class='stat-meta'>Required: ₹" + f"{margin_required:,.0f}" + "</div>"
            "</div>"

            "<div>"
            "<div class='stat-label'>ML Signal Date</div>"
            "<div class='stat-value-sm'>" + escape(str(inputs.get("ml_signal_data_date", "N/A"))) + "</div>"
            "<div class='stat-meta'>" + escape(str(inputs.get("ml_signal_model", "GJR-GARCH"))) + "</div>"
            "</div>"

            "</div>"
        )

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:0.85fr 1.55fr;gap:18px;align-items:stretch;'>"

        "<div style='background:#111116;border:1px solid #2a2a30;border-radius:10px;padding:18px;'>"
        "<div class='placeholder-title'>Market Status</div>"
        "<div style='display:grid;grid-template-columns:1fr;gap:14px;'>"

        "<div>"
        "<div class='stat-label'>Status</div>"
        "<div class='stat-value-sm' style='color:" + status_colour + ";'>" + escape(status["status"]) + "</div>"
        "<div class='stat-meta'>" + escape(status["timestamp"]) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Live NIFTY Quote</div>"
        "<div class='stat-value-sm'>" + escape(last_price_text) + "</div>"
        "<div class='stat-meta'>" + escape(status["quote_symbol"]) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Quote Timestamp</div>"
        "<div class='stat-value-sm'>" + escape(status["quote_timestamp"]) + "</div>"
        "<div class='stat-meta'>Age: " + escape(quote_age_text) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Intraday Signal</div>"
        "<div class='stat-value-sm'>" + escape(status["signal_label"]) + "</div>"
        "<div class='stat-meta'>" + ("Enabled" if status["intraday_enabled"] else "Disabled") + "</div>"
        "</div>"

        "</div>"
        "</div>"

        "<div style='background:#111116;border:1px solid #2a2a30;border-radius:10px;padding:18px;'>"
        "<div class='placeholder-title'>Confirmed EOD Hedge Plan</div>"
        + decision_html +
        "<div style='height:1px;background:#2a2a30;margin:18px 0;'></div>"
        "<div class='placeholder-text'>"
        "This is the confirmed hedge view based on the latest completed market-data state available to the model layer. "
        "The intraday signal is treated separately and remains provisional until market close."
        "</div>"
        "</div>"

        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )



def render_live_dashboard_content(kite, profile, now, tok_prev, user_name, user_email):
    render_overview_command_center(kite)
    render_intraday_signal_panel(kite)
    return

    st.markdown(
        "<div class='section-header'>Last Hedge Decision</div>",
        unsafe_allow_html=True,
    )

    log_path = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "logs", "decision_log.csv"
    )

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        if not log_df.empty:
            last = log_df.iloc[-1]
            rc   = str(last.get("risk_class", "â€”")).lower()
            rcol = {"low": "#4ade80", "medium": "#fbbf24", "high": "#f87171"}.get(rc, "#9090a0")

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.markdown(
                    "<div class='stat-card'><div class='stat-label'>Date</div>"
                    "<div class='stat-value-sm'>" + str(last.get("date", "â€”")) + "</div></div>",
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(
                    "<div class='stat-card'><div class='stat-label'>Risk Class</div>"
                    "<div class='stat-value-sm' style='color:" + rcol + ";text-transform:capitalize;'>"
                    + rc + "</div></div>",
                    unsafe_allow_html=True,
                )
            with d3:
                st.markdown(
                    "<div class='stat-card'><div class='stat-label'>Hedge Action</div>"
                    "<div class='stat-value-sm'>" + str(last.get("hedge_action", "â€”")) + "</div></div>",
                    unsafe_allow_html=True,
                )
            with d4:
                st.markdown(
                    "<div class='stat-card'><div class='stat-label'>Hedge Ratio</div>"
                    "<div class='stat-value'>" + str(last.get("hedge_ratio", "â€”")) + "</div></div>",
                    unsafe_allow_html=True,
                )
            return

    st.markdown(
        "<div class='hedge-panel'>"
        "<div class='hedge-placeholder'>"
        "Open the Hedge Plan tab to view the latest live or historical hedge recommendation."
        "</div></div>",
        unsafe_allow_html=True,
    )


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



def render_live_portfolio_content(kite):
    st.markdown(
        "<div class='section-header'>Portfolio &mdash; Nifty 50 Constituents</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Fetching live quotes and entry prices ..."):
        try:
            symbols         = ["NSE:" + sym for sym, _ in PORTFOLIO]
            quotes          = kite.quote(symbols)
            nse_instruments = kite.instruments("NSE")
            token_map       = {i["tradingsymbol"]: i["instrument_token"]
                               for i in nse_instruments}

            entry_prices = {}
            for sym, _ in PORTFOLIO:
                try:
                    tok  = token_map.get(sym)
                    hist = kite.historical_data(tok, ENTRY_DATE, ENTRY_DATE, "day") if tok else []
                    entry_prices[sym] = hist[0]["close"] if hist else None
                except Exception:
                    entry_prices[sym] = None

            quote_data    = []
            sector_totals = {}

            for sym, sector in PORTFOLIO:
                key = "NSE:" + sym
                if key not in quotes:
                    continue
                q       = quotes[key]
                ltp     = q["last_price"]
                prev    = q["ohlc"]["close"]
                pct     = ((ltp - prev) / prev * 100) if prev else 0
                entry   = entry_prices.get(sym)
                qty     = int(CAPITAL_PER_STOCK // entry) if entry else 0
                cost    = qty * entry if entry else 0
                curr    = qty * ltp
                pnl_rs  = curr - cost if entry else None
                pnl_pct = ((ltp - entry) / entry * 100) if entry else None

                quote_data.append({
                    "symbol":  sym,   "sector":  sector,
                    "ltp":     ltp,   "pct":     pct,
                    "qty":     qty,   "cost":    cost,
                    "curr":    curr,  "pnl_rs":  pnl_rs,
                    "pnl_pct": pnl_pct,
                })
                sector_totals[sector] = sector_totals.get(sector, 0) + 1

            fetch_ok = True
        except Exception as e:
            fetch_ok    = False
            fetch_error = str(e)

    if not fetch_ok:
        st.markdown(
            "<div style='background:#1a0808;border:1px solid #4a1a1a;border-radius:8px;"
            "padding:14px 16px;font-size:13px;color:#f87171;'>"
            "Could not fetch quotes: " + escape(fetch_error) + "</div>",
            unsafe_allow_html=True,
        )
        return

    # â”€â”€ Compute portfolio totals before rendering columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    total_cost = sum(s["cost"] for s in quote_data if s["cost"])
    total_curr = sum(s["curr"] for s in quote_data if s["curr"])
    total_pnl  = total_curr - total_cost
    total_pct  = (total_pnl / total_cost * 100) if total_cost else 0
    today      = datetime.now(IST).date()
    xirr_val   = (
        xirr([-total_cost, total_curr], [ENTRY_DATE, today])
        if total_cost > 0 and total_curr > 0 else None
    )
    pnl_col   = "#4ade80" if total_pnl >= 0 else "#f87171"
    pnl_sign  = "+" if total_pnl >= 0 else ""
    xirr_html = f"{pnl_sign}{xirr_val:.1f}% p.a." if xirr_val is not None else "â€”"
    xirr_col  = "#4ade80" if xirr_val and xirr_val >= 0 else "#f87171"

    col_portfolio, col_chart = st.columns([3, 2])

    # â”€â”€ Left: portfolio table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with col_portfolio:
        def portfolio_row(item):
            pct      = item["pct"]
            pnl_pct  = item["pnl_pct"]
            pnl_rs   = item["pnl_rs"]
            sign     = "+" if pct >= 0 else ""
            pnl_sign = "+" if pnl_pct and pnl_pct >= 0 else ""
            rs_sign  = "+" if pnl_rs and pnl_rs >= 0 else ""
            chg_cls  = ("stock-change-pos" if pct > 0
                        else "stock-change-neg" if pct < 0
                        else "stock-change-neu")
            pnl_cls  = ("stock-pnl-pos" if pnl_pct and pnl_pct >= 0
                        else "stock-pnl-neg" if pnl_pct and pnl_pct < 0
                        else "stock-pnl-neu")
            col      = SECTOR_COLOURS.get(item["sector"], "#555560")
            sector_badge = (
                "<span class='stock-sector' style='background:" + col + "22;"
                "color:" + col + ";border:1px solid " + col + "44;'>"
                + item["sector"] + "</span>"
            )
            pnl_html = (
                "<div class='" + pnl_cls + "'>"
                + pnl_sign + "{:.2f}%".format(pnl_pct)
                + " &nbsp;|&nbsp; "
                + rs_sign + "&#8377;{:,.0f}".format(pnl_rs)
                + "</div>"
                if pnl_pct is not None
                else "<div class='stock-pnl-neu'>Entry unavailable</div>"
            )
            return (
                "<div class='portfolio-row'>"
                "<div>"
                "<div class='stock-symbol'>" + item["symbol"] + "</div>"
                + sector_badge +
                "<div class='stock-qty'>Qty: " + str(item["qty"])
                + " &nbsp;&middot;&nbsp; Cost: &#8377;" + "{:,.0f}".format(item["cost"]) + "</div>"
                "</div>"
                "<div style='text-align:right;'>"
                "<div class='stock-price'>&#8377;" + "{:,.2f}".format(item["ltp"]) + "</div>"
                "<div class='" + chg_cls + "'>" + sign + "{:.2f}%</div>".format(pct)
                + pnl_html +
                "</div></div>"
            )

        rows = "".join(portfolio_row(s) for s in quote_data)
        st.markdown(
            "<div class='portfolio-table'>" + rows + "</div>",
            unsafe_allow_html=True,
        )

    # â”€â”€ Right: summary at top, pie chart below â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with col_chart:
        # Portfolio summary card
        st.markdown(
            "<div style='background:#16161a;border:1px solid #2a2a30;border-radius:12px;"
            "padding:20px 24px;'>"
            "<p style='font-size:11px;font-weight:500;letter-spacing:0.7px;text-transform:uppercase;"
            "color:#555560;font-family:Inter,sans-serif;margin:0 0 16px;'>Portfolio Summary</p>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;'>"

            # Invested
            "<div>"
            "<div style='font-size:10px;color:#555560;text-transform:uppercase;"
            "letter-spacing:0.6px;font-family:Inter,sans-serif;margin-bottom:4px;'>Invested</div>"
            "<div style='font-size:16px;font-weight:600;color:#f0f0f5;font-family:Inter,sans-serif;'>"
            "&#8377;" + "{:,.0f}".format(total_cost) + "</div>"
            "</div>"

            # Current Value
            "<div>"
            "<div style='font-size:10px;color:#555560;text-transform:uppercase;"
            "letter-spacing:0.6px;font-family:Inter,sans-serif;margin-bottom:4px;'>Current Value</div>"
            "<div style='font-size:16px;font-weight:600;color:#f0f0f5;font-family:Inter,sans-serif;'>"
            "&#8377;" + "{:,.0f}".format(total_curr) + "</div>"
            "</div>"

            # P&L
            "<div>"
            "<div style='font-size:10px;color:#555560;text-transform:uppercase;"
            "letter-spacing:0.6px;font-family:Inter,sans-serif;margin-bottom:4px;'>P&amp;L</div>"
            "<div style='font-size:16px;font-weight:600;color:" + pnl_col + ";font-family:Inter,sans-serif;'>"
            + pnl_sign + "&#8377;{:,.0f}".format(total_pnl) + "</div>"
            "<div style='font-size:12px;color:" + pnl_col + ";font-family:Inter,sans-serif;'>"
            + pnl_sign + "{:.2f}%</div>".format(total_pct) +
            "</div>"

            # XIRR
            "<div>"
            "<div style='font-size:10px;color:#555560;text-transform:uppercase;"
            "letter-spacing:0.6px;font-family:Inter,sans-serif;margin-bottom:4px;'>XIRR</div>"
            "<div style='font-size:16px;font-weight:600;color:" + xirr_col + ";font-family:Inter,sans-serif;'>"
            + xirr_html + "</div>"
            "<div style='font-size:12px;color:#555560;font-family:Inter,sans-serif;'>Annualised</div>"
            "</div>"

            "</div></div>",
            unsafe_allow_html=True,
        )

        # Sector pie chart below summary
        render_sector_pie(sector_totals)

    # â”€â”€ Last hedge decision â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€




def format_money_html(value, decimals=0):
    try:
        return "&#8377;" + f"{float(value):,.{decimals}f}"
    except Exception:
        return "&#8377;0"



def _load_close_series_for_live_ratio(csv_path, label):
    if not os.path.exists(csv_path):
        return pd.Series(dtype=float, name=label)

    try:
        df = pd.read_csv(csv_path)

        if "date" not in df.columns or "close" not in df.columns:
            return pd.Series(dtype=float, name=label)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")

        if df.empty:
            return pd.Series(dtype=float, name=label)

        series = df.set_index("date")["close"].astype(float)
        series.name = label
        return series.sort_index()

    except Exception:
        return pd.Series(dtype=float, name=label)


def _append_live_price_for_live_ratio(series, live_price, live_timestamp, use_live_price):
    if not use_live_price or live_price is None:
        return series, False

    try:
        live_price = float(live_price)
    except Exception:
        return series, False

    if not np.isfinite(live_price) or live_price <= 0:
        return series, False

    timestamp_raw = str(live_timestamp or "").strip()
    timestamp_clean = (
        timestamp_raw
        .replace(" IST", "")
        .replace(" UTC", "")
        .replace(" GMT", "")
        .strip()
    )

    parsed = pd.to_datetime(timestamp_clean, errors="coerce")

    if pd.isna(parsed):
        parsed = pd.Timestamp(datetime.now(IST).date())
    else:
        parsed = pd.Timestamp(parsed).normalize()

    updated = series.copy().sort_index()
    updated.loc[parsed] = live_price
    updated = updated.sort_index()

    return updated, True



def compute_dynamic_live_hedge_ratio(
    live_spot_close=None,
    live_futures_close=None,
    live_timestamp=None,
    use_live_price=False,
    lookback_days=90,
    fallback_ratio=0.40,
):
    """Select a live hedge-ratio strategy using recent spot/futures return memory.

    The module compares multiple candidate hedge strategies on recent paired
    NIFTY spot and futures returns. The selected candidate ratio is then passed
    to the Hedra volatility risk gate in build_live_hedge_plan().
    """

    def safe_ratio(value, upper=1.25):
        try:
            value = float(value)
        except Exception:
            return 0.0

        if not np.isfinite(value):
            return 0.0

        return max(0.0, min(value, upper))

    def score_candidate(recent_returns, ratio):
        ratio = safe_ratio(ratio)

        hedged = recent_returns["spot_close"] - ratio * recent_returns["futures_close"]

        hedged_vol = float(hedged.std() * np.sqrt(252))
        downside = hedged[hedged < 0.0]
        downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) >= 5 else 0.0

        if not np.isfinite(hedged_vol):
            hedged_vol = 999.0

        if not np.isfinite(downside_vol):
            downside_vol = 999.0

        # Small ratio penalty avoids selecting unnecessarily large hedge ratios
        # when two strategies have very similar recent hedged-risk scores.
        score = hedged_vol + (0.25 * downside_vol) + (0.002 * abs(ratio))

        return {
            "hedged_volatility": hedged_vol,
            "downside_volatility": downside_vol,
            "score": float(score),
        }

    spot_path = os.path.join("data", "raw", "NIFTY50_spot.csv")
    futures_path = os.path.join("data", "raw", "NIFTY_futures_continuous.csv")

    spot = _load_close_series_for_live_ratio(spot_path, "spot_close")
    futures = _load_close_series_for_live_ratio(futures_path, "futures_close")

    fallback_ratio = safe_ratio(fallback_ratio)

    if spot.empty or futures.empty:
        return {
            "candidate_ratio": fallback_ratio,
            "raw_min_variance_ratio": fallback_ratio,
            "raw_selected_strategy_ratio": fallback_ratio,
            "selected_strategy": "fallback_volatility_regime_target",
            "selected_strategy_label": "Fallback volatility-regime target",
            "method": "Live hedge-ratio strategy selector",
            "source": "fallback_missing_price_history",
            "mode": "fallback",
            "data_date": "N/A",
            "lookback_days": 0,
            "strategy_candidates": [],
            "note": "Spot or futures price history was unavailable, so the volatility-regime fallback target was used.",
        }

    spot, live_spot_used = _append_live_price_for_live_ratio(
        series=spot,
        live_price=live_spot_close,
        live_timestamp=live_timestamp,
        use_live_price=use_live_price,
    )

    futures, live_futures_used = _append_live_price_for_live_ratio(
        series=futures,
        live_price=live_futures_close,
        live_timestamp=live_timestamp,
        use_live_price=use_live_price,
    )

    merged = pd.concat([spot, futures], axis=1).dropna().sort_index()

    if len(merged) < 40:
        return {
            "candidate_ratio": fallback_ratio,
            "raw_min_variance_ratio": fallback_ratio,
            "raw_selected_strategy_ratio": fallback_ratio,
            "selected_strategy": "fallback_insufficient_history",
            "selected_strategy_label": "Fallback volatility-regime target",
            "method": "Live hedge-ratio strategy selector",
            "source": "fallback_insufficient_price_history",
            "mode": "fallback",
            "data_date": merged.index[-1].strftime("%Y-%m-%d") if not merged.empty else "N/A",
            "lookback_days": len(merged),
            "strategy_candidates": [],
            "note": "Insufficient paired spot/futures history was available, so the volatility-regime fallback target was used.",
        }

    returns = np.log(merged).diff().dropna()
    recent_returns = returns.tail(lookback_days).dropna()

    if len(recent_returns) < 30:
        return {
            "candidate_ratio": fallback_ratio,
            "raw_min_variance_ratio": fallback_ratio,
            "raw_selected_strategy_ratio": fallback_ratio,
            "selected_strategy": "fallback_insufficient_return_memory",
            "selected_strategy_label": "Fallback volatility-regime target",
            "method": "Live hedge-ratio strategy selector",
            "source": "fallback_insufficient_return_memory",
            "mode": "fallback",
            "data_date": merged.index[-1].strftime("%Y-%m-%d"),
            "lookback_days": len(recent_returns),
            "strategy_candidates": [],
            "note": "Insufficient recent return memory was available, so the volatility-regime fallback target was used.",
        }

    futures_variance = float(recent_returns["futures_close"].var())

    if not np.isfinite(futures_variance) or futures_variance <= 0:
        return {
            "candidate_ratio": fallback_ratio,
            "raw_min_variance_ratio": fallback_ratio,
            "raw_selected_strategy_ratio": fallback_ratio,
            "selected_strategy": "fallback_invalid_futures_variance",
            "selected_strategy_label": "Fallback volatility-regime target",
            "method": "Live hedge-ratio strategy selector",
            "source": "fallback_invalid_futures_variance",
            "mode": "fallback",
            "data_date": merged.index[-1].strftime("%Y-%m-%d"),
            "lookback_days": len(recent_returns),
            "strategy_candidates": [],
            "note": "Futures variance was invalid, so the volatility-regime fallback target was used.",
        }

    covariance = float(recent_returns["spot_close"].cov(recent_returns["futures_close"]))
    raw_min_variance_ratio = covariance / futures_variance

    if not np.isfinite(raw_min_variance_ratio):
        raw_min_variance_ratio = fallback_ratio

    corr = float(recent_returns["spot_close"].corr(recent_returns["futures_close"]))
    if not np.isfinite(corr):
        corr = 0.0

    downside_returns = recent_returns[recent_returns["spot_close"] < 0.0]

    if len(downside_returns) >= 20 and float(downside_returns["futures_close"].var()) > 0:
        downside_ratio = float(
            downside_returns["spot_close"].cov(downside_returns["futures_close"])
            / downside_returns["futures_close"].var()
        )
    else:
        downside_ratio = raw_min_variance_ratio

    candidate_definitions = [
        ("no_hedge", "No hedge", 0.0),
        ("static_50", "Static 50% hedge", 0.50),
        ("rolling_min_variance", "Rolling minimum-variance hedge", raw_min_variance_ratio),
        ("downside_min_variance", "Downside minimum-variance hedge", downside_ratio),
        (
            "correlation_scaled_min_variance",
            "Correlation-scaled minimum-variance hedge",
            raw_min_variance_ratio * abs(corr),
        ),
    ]

    candidate_rows = []

    for strategy_key, strategy_label, raw_ratio in candidate_definitions:
        cleaned_ratio = safe_ratio(raw_ratio)
        score = score_candidate(recent_returns, cleaned_ratio)

        candidate_rows.append(
            {
                "strategy": strategy_key,
                "strategy_label": strategy_label,
                "raw_ratio": float(raw_ratio) if np.isfinite(raw_ratio) else 0.0,
                "candidate_ratio": cleaned_ratio,
                "hedged_volatility": score["hedged_volatility"],
                "downside_volatility": score["downside_volatility"],
                "selection_score": score["score"],
            }
        )

    ranked = sorted(candidate_rows, key=lambda row: row["selection_score"])
    selected = ranked[0]

    mode = "intraday_provisional" if live_spot_used and live_futures_used else "confirmed_eod_memory"
    source = (
        "live_strategy_selector_intraday"
        if mode == "intraday_provisional"
        else "live_strategy_selector_eod_memory"
    )

    note = (
        "Multiple live hedge-ratio candidates were compared using recent paired spot/futures return memory. "
        f"The selected strategy was {selected['strategy_label']} with the lowest recent hedged-risk score."
    )

    if mode == "intraday_provisional":
        note += " Fresh Kite spot and futures prices were appended as provisional intraday observations before strategy selection."
    else:
        note += " Latest confirmed EOD spot and futures history was used for strategy selection."

    return {
        "candidate_ratio": float(selected["candidate_ratio"]),
        "raw_min_variance_ratio": float(raw_min_variance_ratio),
        "raw_selected_strategy_ratio": float(selected["raw_ratio"]),
        "selected_strategy": selected["strategy"],
        "selected_strategy_label": selected["strategy_label"],
        "selected_strategy_score": float(selected["selection_score"]),
        "method": "Live hedge-ratio strategy selector",
        "source": source,
        "mode": mode,
        "data_date": merged.index[-1].strftime("%Y-%m-%d"),
        "lookback_days": len(recent_returns),
        "strategy_candidates": candidate_rows,
        "note": note,
    }




def build_live_hedge_plan(kite):
    snapshot = get_live_broker_snapshot(kite)

    total_capital = TOTAL_CAPITAL
    stock_allocation = STOCK_ALLOCATION
    simulated_margin_reserve = MARGIN_RESERVE

    portfolio_value = stock_allocation

    market_status = get_kite_market_status(kite)
    live_signal = get_live_volatility_signal(
        live_close=market_status.get("last_price"),
        live_timestamp=market_status.get("quote_timestamp"),
        use_live_price=bool(market_status.get("intraday_enabled")),
    )
    predicted_volatility = live_signal.predicted_volatility
    risk_regime = live_signal.risk_regime

    dynamic_hedge_signal = compute_dynamic_live_hedge_ratio(
        live_spot_close=market_status.get("last_price"),
        live_futures_close=snapshot.futures_price,
        live_timestamp=market_status.get("quote_timestamp"),
        use_live_price=bool(market_status.get("intraday_enabled")),
        fallback_ratio=live_signal.core_target_ratio,
    )

    candidate_hedge_ratio = dynamic_hedge_signal["candidate_ratio"]
    core_target_ratio, live_risk_gate_note = apply_hedra_risk_gate(
        candidate_hedge_ratio,
        risk_regime,
    )

    margin_per_contract = snapshot.futures_price * snapshot.contract.lot_size * 0.12

    item = PlannedExecutionInput(
        scenario_name="dashboard_live_hedge_plan",
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

    inputs = {
        "total_capital": total_capital,
        "stock_allocation": stock_allocation,
        "simulated_margin_reserve": simulated_margin_reserve,
        "portfolio_value": portfolio_value,
        "predicted_volatility": predicted_volatility,
        "risk_regime": risk_regime,
        "core_target_ratio": core_target_ratio,
        "candidate_hedge_ratio": candidate_hedge_ratio,
        "raw_min_variance_ratio": dynamic_hedge_signal.get("raw_min_variance_ratio"),
        "live_hedge_ratio_method": dynamic_hedge_signal.get("method"),
        "live_hedge_ratio_source": dynamic_hedge_signal.get("source"),
        "live_hedge_ratio_mode": dynamic_hedge_signal.get("mode"),
        "live_hedge_ratio_data_date": dynamic_hedge_signal.get("data_date"),
        "live_hedge_ratio_lookback_days": dynamic_hedge_signal.get("lookback_days"),
        "live_risk_gate_note": live_risk_gate_note,
        "live_hedge_ratio_note": dynamic_hedge_signal.get("note"),
        "selected_live_strategy": dynamic_hedge_signal.get("selected_strategy"),
        "selected_live_strategy_label": dynamic_hedge_signal.get("selected_strategy_label"),
        "selected_live_strategy_score": dynamic_hedge_signal.get("selected_strategy_score"),
        "raw_selected_strategy_ratio": dynamic_hedge_signal.get("raw_selected_strategy_ratio"),
        "live_strategy_candidates": dynamic_hedge_signal.get("strategy_candidates"),
        "ml_signal_source": live_signal.source,
        "ml_signal_model": live_signal.model,
        "ml_signal_data_date": live_signal.data_date,
        "ml_low_threshold": live_signal.low_threshold,
        "ml_high_threshold": live_signal.high_threshold,
        "ml_latest_close": live_signal.latest_close,
        "margin_per_contract": margin_per_contract,
    }

    return snapshot, item, result, inputs


def get_result_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)



def render_live_hedge_plan(kite):
    st.markdown(
        "<div class='section-header'>Live Hedge Recommendation</div>",
        unsafe_allow_html=True,
    )

    if "live_hedge_plan_cache" not in st.session_state:
        with st.spinner("Running live broker snapshot, planner, and futures execution engine ..."):
            try:
                snapshot, item, result, inputs = build_live_hedge_plan(kite)
                st.session_state.live_hedge_plan_cache = {
                    "snapshot": snapshot,
                    "item": item,
                    "result": result,
                    "inputs": inputs,
                    "error": None,
                }
            except Exception as exc:
                st.session_state.live_hedge_plan_cache = {
                    "snapshot": None,
                    "item": None,
                    "result": None,
                    "inputs": None,
                    "error": str(exc),
                }

    cache = st.session_state.live_hedge_plan_cache

    if cache["error"]:
        st.markdown(
            "<div style='background:#1a0808;border:1px solid #4a1a1a;border-radius:8px;"
            "padding:14px 16px;font-size:13px;color:#f87171;'>"
            "Could not generate hedge plan: " + escape(cache["error"]) + "</div>",
            unsafe_allow_html=True,
        )
        return

    snapshot = cache["snapshot"]
    result = cache["result"]
    inputs = cache["inputs"]
    execution = result.futures_execution_result
    planning = result.planning_result

    margin_colour = "#4ade80" if result.margin_sufficient else "#f87171"
    action_colour = "#4ade80" if result.status == "EXECUTABLE_PLAN" else "#fbbf24"

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Planner Action</div>"
        "<div class='stat-value-sm' style='color:" + action_colour + ";'>" + escape(ui_label(result.position_action)) + "</div>"
        "<div class='stat-meta'>" + escape(ui_label(result.execution_instruction)) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Recommended Contracts</div>"
        "<div class='stat-value-sm'>" + str(result.recommended_short_contracts) + " short NIFTY FUT</div>"
        "<div class='stat-meta'>Current: " + str(result.current_short_contracts) + " &middot; Change: +" + str(result.contract_delta) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Target Hedge Ratio</div>"
        "<div class='stat-value-sm'>" + f"{result.execution_target_hedge_ratio:.2f}" + "</div>"
        "<div class='stat-meta'>Achieved: " + f"{get_result_field(execution, 'achieved_hedge_ratio', 0.0):.4f}" + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Margin Check</div>"
        "<div class='stat-value-sm' style='color:" + margin_colour + ";'>" + str(result.margin_sufficient) + "</div>"
        "<div class='stat-meta'>Required: " + format_money_html(get_result_field(execution, 'additional_margin_required', 0.0), 0) + "</div>"
        "</div>"

        "</div>"

        "<div style='height:1px;background:#2a2a30;margin:20px 0;'></div>"

        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Futures Contract</div>"
        "<div class='stat-value-sm'>" + escape(snapshot.contract.tradingsymbol) + "</div>"
        "<div class='stat-meta'>Expiry: " + escape(str(snapshot.contract.expiry)) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Futures Price</div>"
        "<div class='stat-value-sm'>" + format_money_html(snapshot.futures_price, 2) + "</div>"
        "<div class='stat-meta'>Lot size: " + str(snapshot.contract.lot_size) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Portfolio Exposure</div>"
        "<div class='stat-value-sm'>" + format_money_html(inputs["portfolio_value"], 0) + "</div>"
        "<div class='stat-meta'>Risk regime: " + escape(inputs["risk_regime"]) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Margin Reserve</div>"
        "<div class='stat-value-sm'>" + format_money_html(inputs["simulated_margin_reserve"], 0) + "</div>"
        "<div class='stat-meta'>Kite margin shown separately in adapter</div>"
        "</div>"

        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    plan_lines = get_result_field(planning, 'explanation', []) or []
    exec_lines = get_result_field(execution, 'explanation', []) or []

    plan_html = "".join("<li>" + escape(ui_text(line)) + "</li>" for line in plan_lines)
    exec_html = "".join("<li>" + escape(ui_text(line)) + "</li>" for line in exec_lines)

    strategy_method = str(inputs.get("live_hedge_ratio_method", "Live hedge-ratio strategy selector"))
    selected_strategy_label = str(inputs.get("selected_live_strategy_label", "N/A"))
    strategy_mode = str(inputs.get("live_hedge_ratio_mode", "confirmed_eod_memory"))
    strategy_date = str(inputs.get("live_hedge_ratio_data_date", "N/A"))
    strategy_source = str(inputs.get("live_hedge_ratio_source", "N/A"))
    strategy_note = str(inputs.get("live_hedge_ratio_note", "Latest available spot/futures memory used."))
    risk_gate_note = str(inputs.get("live_risk_gate_note", "Volatility risk gate applied."))

    try:
        candidate_ratio_display = f"{float(inputs.get('candidate_hedge_ratio', result.execution_target_hedge_ratio)):.2%}"
    except Exception:
        candidate_ratio_display = "N/A"

    try:
        raw_selected_ratio_display = f"{float(inputs.get('raw_selected_strategy_ratio', inputs.get('candidate_hedge_ratio', 0.0))):.2%}"
    except Exception:
        raw_selected_ratio_display = "N/A"

    st.markdown(
        "<div class='section-header'>Applied Hedge-Ratio Strategy</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Strategy Engine</div>"
        "<div class='stat-value-sm'>" + escape(strategy_method) + "</div>"
        "<div class='stat-meta'>" + escape(strategy_mode
        .replace("confirmed_eod_memory", "Confirmed EOD memory")
        .replace("intraday_provisional", "Intraday provisional")
        .replace("fallback", "Fallback")) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Selected Strategy</div>"
        "<div class='stat-value-sm'>" + escape(selected_strategy_label) + "</div>"
        "<div class='stat-meta'>Raw selected ratio: " + escape(raw_selected_ratio_display) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Final Hedra Target</div>"
        "<div class='stat-value-sm'>" + f"{result.execution_target_hedge_ratio:.2%}" + "</div>"
        "<div class='stat-meta'>After volatility risk gate</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Strategy Data Date</div>"
        "<div class='stat-value-sm'>" + escape(strategy_date) + "</div>"
        "<div class='stat-meta'>" + escape(strategy_source
        .replace("live_strategy_selector_eod_memory", "Live strategy selector using confirmed EOD memory")
        .replace("live_strategy_selector_intraday", "Live strategy selector using intraday quote memory")
        .replace("_", " ")) + "</div>"
        "</div>"

        "</div>"
        "<div style='height:1px;background:#2a2a30;margin:18px 0;'></div>"
        "<div class='placeholder-text'>"
        + escape(strategy_note) +
        " " +
        escape(risk_gate_note) +
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-header'>Decision Rationale &amp; Execution Details</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:18px;'>"

        "<div style='background:#111116;border:1px solid #2a2a30;border-radius:10px;padding:16px;'>"
        "<div class='placeholder-title'>Planning rationale</div>"
        "<div class='placeholder-text'>"
        "The planner converts the current volatility regime, target hedge ratio, existing hedge position, "
        "and turnover-cost setting into an actionable hedge decision. The result is an interpretable hedge instruction "
        "rather than a direct black-box trade."
        "</div>"
        "</div>"

        "<div style='background:#111116;border:1px solid #2a2a30;border-radius:10px;padding:16px;'>"
        "<div class='placeholder-title'>Execution sizing</div>"
        "<div class='placeholder-text'>"
        "The execution layer translates the planner target into NIFTY futures contracts, checks the required margin, "
        "rounds the position to tradable contract units, and reports the achieved hedge ratio after rounding."
        "</div>"
        "</div>"

        "</div>"
        "<div style='height:1px;background:#2a2a30;margin:18px 0;'></div>"
        "<div class='placeholder-text'>"
        "The detailed planning and execution trace is retained below for transparency and auditability."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View technical planning and execution trace", expanded=False):
        st.markdown(
            "<div class='hedge-panel'>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:22px;'>"

            "<div>"
            "<div class='placeholder-title'>Planning trace</div>"
            "<ul class='placeholder-text'>" + plan_html + "</ul>"
            "</div>"

            "<div>"
            "<div class='placeholder-title'>Execution trace</div>"
            "<ul class='placeholder-text'>" + exec_html + "</ul>"
            "</div>"

            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )




def render_live_explanation(kite):
    st.markdown(
        "<div class='section-header'>Live Decision Explanation</div>",
        unsafe_allow_html=True,
    )

    if "live_hedge_plan_cache" not in st.session_state:
        with st.spinner("Preparing live explanation ..."):
            try:
                snapshot, item, result, inputs = build_live_hedge_plan(kite)
                st.session_state.live_hedge_plan_cache = {
                    "snapshot": snapshot,
                    "item": item,
                    "result": result,
                    "inputs": inputs,
                    "error": None,
                }
            except Exception as exc:
                st.session_state.live_hedge_plan_cache = {
                    "snapshot": None,
                    "item": None,
                    "result": None,
                    "inputs": None,
                    "error": str(exc),
                }

    cache = st.session_state.live_hedge_plan_cache

    if cache.get("error"):
        st.markdown(
            "<div style='background:#1a0808;border:1px solid #4a1a1a;border-radius:8px;"
            "padding:14px 16px;font-size:13px;color:#f87171;'>"
            "Live explanation unavailable: " + escape(cache["error"]) + "</div>",
            unsafe_allow_html=True,
        )
        return

    snapshot = cache["snapshot"]
    result = cache["result"]
    inputs = cache["inputs"]
    execution = result.futures_execution_result
    planning = result.planning_result

    plan_lines = get_result_field(planning, "explanation", []) or []
    exec_lines = get_result_field(execution, "explanation", []) or []

    action = ui_label(result.position_action)
    instruction = ui_label(result.execution_instruction)
    risk_regime = str(inputs.get("risk_regime", "N/A")).title()
    predicted_volatility = float(inputs.get("predicted_volatility", 0.0))
    target_ratio = float(result.execution_target_hedge_ratio)
    achieved_ratio = float(get_result_field(execution, "achieved_hedge_ratio", 0.0))
    target_contracts = int(result.recommended_short_contracts)
    current_contracts = int(result.current_short_contracts)
    trade_now = int(result.contract_delta)
    futures_price = float(snapshot.futures_price)
    lot_size = int(snapshot.contract.lot_size)
    contract_notional = futures_price * lot_size
    margin_required = float(get_result_field(execution, "additional_margin_required", 0.0))
    margin_status = "Sufficient" if result.margin_sufficient else "Insufficient"

    if trade_now == 0:
        trade_sentence = (
            "No additional futures trade is required because the current position already "
            "matches the target hedge position."
        )
    elif trade_now > 0:
        trade_sentence = (
            f"Hedra recommends adding {trade_now} short NIFTY futures contract(s) to reach "
            "the target hedge position."
        )
    else:
        trade_sentence = (
            f"Hedra recommends reducing the short hedge by {abs(trade_now)} contract(s) to "
            "bring the position back to the target level."
        )

    action_lower = action.lower()
    risk_lower = risk_regime.lower()

    if risk_lower == "high":
        market_text = (
            f"The current market is classified as {risk_regime} risk. The predicted volatility is "
            f"{predicted_volatility:.2%}, so Hedra treats the portfolio as needing stronger downside protection."
        )
    elif risk_lower == "medium":
        market_text = (
            f"The current market is classified as {risk_regime} risk. The predicted volatility is "
            f"{predicted_volatility:.2%}, so Hedra applies a moderate hedge rather than fully reducing exposure."
        )
    elif risk_lower == "low":
        market_text = (
            f"The current market is classified as {risk_regime} risk. The predicted volatility is "
            f"{predicted_volatility:.2%}, so Hedra avoids unnecessary hedging unless the planner finds a clear need."
        )
    else:
        market_text = (
            f"The current market risk label is {risk_regime}. The predicted volatility used by the decision layer is "
            f"{predicted_volatility:.2%}."
        )

    if "open" in action_lower:
        decision_text = "The planner recommends opening a short NIFTY futures hedge to reduce downside portfolio risk."
    elif "hold" in action_lower:
        decision_text = "The planner recommends holding the current hedge position because the existing protection is already close to the target."
    elif "reduce" in action_lower:
        decision_text = "The planner recommends reducing the hedge because the current futures position is higher than the target level."
    elif "close" in action_lower:
        decision_text = "The planner recommends closing the hedge because the system no longer requires short futures protection."
    else:
        decision_text = f"The planner selected the action: {action}."

    if target_contracts > 0:
        hedge_size_text = (
            f"The target hedge ratio is {target_ratio:.2%}. Based on the futures price and lot size, "
            f"this becomes a target position of {target_contracts} short NIFTY futures contract(s). "
            f"The achieved hedge ratio is approximately {achieved_ratio:.2%}."
        )
    else:
        hedge_size_text = (
            f"The target hedge ratio is {target_ratio:.2%}. This results in no short futures position being required."
        )

    if result.margin_sufficient:
        margin_text = (
            f"The required additional margin is ₹{margin_required:,.0f}, and the available margin reserve is sufficient. "
            "This means the recommendation can be reviewed as an executable hedge decision."
        )
    else:
        margin_text = (
            f"The required additional margin is ₹{margin_required:,.0f}, but the margin reserve is not sufficient. "
            "This means the hedge recommendation should not be executed without additional capital or position adjustment."
        )

    narrative_sections = [
        ("Market condition", market_text),
        ("Planner decision", decision_text),
        ("Hedge size", hedge_size_text),
        ("Execution check", trade_sentence + " " + margin_text),
    ]

    narrative_html = "".join(
        "<div style='padding:14px 0;border-bottom:1px solid #2a2a30;'>"
        "<div style='font-size:12px;font-weight:650;color:#f0f0f5;margin-bottom:6px;'>"
        + escape(title)
        + "</div>"
        "<div class='placeholder-text'>"
        + escape(body)
        + "</div>"
        "</div>"
        for title, body in narrative_sections
    )

    strategy_method = str(inputs.get("live_hedge_ratio_method", "Live hedge-ratio strategy selector"))
    selected_strategy_label = str(inputs.get("selected_live_strategy_label", "N/A"))
    strategy_mode = str(inputs.get("live_hedge_ratio_mode", "confirmed_eod_memory"))
    strategy_date = str(inputs.get("live_hedge_ratio_data_date", "N/A"))
    risk_gate_note = str(inputs.get("live_risk_gate_note", "Volatility risk gate applied."))

    try:
        candidate_ratio = f"{float(inputs.get('candidate_hedge_ratio', target_ratio)):.2%}"
    except Exception:
        candidate_ratio = "N/A"

    try:
        raw_selected_strategy_ratio = f"{float(inputs.get('raw_selected_strategy_ratio', inputs.get('candidate_hedge_ratio', 0.0))):.2%}"
    except Exception:
        raw_selected_strategy_ratio = "N/A"

    factor_rows = [
        ("Hedge ratio engine", strategy_method),
        ("Selected live strategy", selected_strategy_label),
        ("Strategy mode", strategy_mode.replace("_", " ").title()),
        ("Strategy data date", strategy_date),
        ("Raw selected strategy ratio", raw_selected_strategy_ratio),
        ("Candidate ratio before risk gate", candidate_ratio),
        ("Risk gate rule", risk_gate_note),
        ("Risk regime", risk_regime),
        ("Predicted volatility", f"{predicted_volatility:.2%}"),
        ("Planner action", action),
        ("Execution instruction", instruction),
        ("Target hedge ratio", f"{target_ratio:.2%}"),
        ("Achieved hedge ratio", f"{achieved_ratio:.2%}"),
        ("Current short futures position", f"{current_contracts} contracts"),
        ("Target short futures position", f"{target_contracts} contracts"),
        ("Contracts to trade now", "No trade" if trade_now == 0 else f"{trade_now:+d} contracts"),
        ("Futures contract", snapshot.contract.tradingsymbol),
        ("Futures price", f"₹{futures_price:,.2f}"),
        ("Lot size", f"{lot_size}"),
        ("Estimated contract notional", f"₹{contract_notional:,.2f}"),
        ("Required margin", f"₹{margin_required:,.0f}"),
        ("Margin status", margin_status),
    ]

    factors_df = pd.DataFrame(factor_rows, columns=["Decision Factor", "Value"])

    st.markdown(
        "<div class='hedge-panel'>"
        "<div class='placeholder-title'>Decision narrative</div>"
        + narrative_html +
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-header'>Decision Factors</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(factors_df, hide_index=True, width="stretch")

    st.markdown(
        "<div class='section-header'>XAI / SHAP Status</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='hedge-panel'>"
        "<div class='placeholder-text'>"
        "<strong>Current explanation type:</strong> planner-based and execution-based explainability. "
        "This live view explains why the planner selected the hedge action and how the futures position was sized. "
        "<br><br>"
        "<strong>SHAP status:</strong> SHAP feature attribution is not yet active in this dashboard view. "
        "It should be connected later to the volatility prediction layer to show the top model drivers behind the risk signal."
        "</div></div>",
        unsafe_allow_html=True,
    )

    trace_rows = []
    for idx, line in enumerate(plan_lines, start=1):
        trace_rows.append({
            "Trace Type": "Planning",
            "Step": idx,
            "Message": ui_text(line),
        })

    for idx, line in enumerate(exec_lines, start=1):
        trace_rows.append({
            "Trace Type": "Execution",
            "Step": idx,
            "Message": ui_text(line),
        })

    with st.expander("Technical trace: planner and execution logs", expanded=False):
        if trace_rows:
            trace_df = pd.DataFrame(trace_rows)
            st.dataframe(trace_df, hide_index=True, width="stretch")
        else:
            st.info("No technical trace lines were returned for this decision.")



def get_backtest_dataset_catalog():
    return {
        "Rolling hedge performance": {
            "summary": "outputs/reports/hedge_performance/hedge_performance_summary.csv",
            "daily": "outputs/reports/hedge_performance/hedge_performance_daily_details.csv",
            "ratio": "outputs/reports/hedge_performance/hedge_ratio_series.csv",
            "description": "Baseline hedge strategies including no hedge, static hedge, rolling minimum-variance, and GJR-scaled hedge ratios.",
        },
        "DCC-GJR-GARCH benchmark": {
            "summary": "outputs/reports/dcc_garch/dcc_garch_performance_summary.csv",
            "daily": "outputs/reports/dcc_garch/dcc_garch_daily_details.csv",
            "ratio": "outputs/reports/dcc_garch/dcc_garch_ratio_series.csv",
            "description": "Dynamic conditional correlation benchmark using DCC-GJR-GARCH hedge ratios.",
        },
        "DCC smoothing and no-trade-band benchmark": {
            "summary": "outputs/reports/dcc_smoothing/dcc_smoothing_performance_summary.csv",
            "daily": "outputs/reports/dcc_smoothing/dcc_smoothing_daily_details.csv",
            "ratio": "outputs/reports/dcc_smoothing/dcc_smoothing_ratio_series.csv",
            "description": "Smoothed DCC hedge ratios with EMA and no-trade-band variants for turnover control.",
        },
    }


@st.cache_data(show_spinner=False)
def load_backtest_csv(path_text):
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_metric_value(df, column, mode="min"):
    if df.empty or column not in df.columns:
        return None
    clean = df.dropna(subset=[column])
    if clean.empty:
        return None
    if mode == "max":
        return clean.loc[clean[column].idxmax()]
    return clean.loc[clean[column].idxmin()]


@st.cache_data(show_spinner=False)
def load_historical_replay_inputs():
    ratio_core = load_backtest_csv("outputs/reports/hedge_performance/hedge_ratio_series.csv")
    ratio_dcc = load_backtest_csv("outputs/reports/dcc_smoothing/dcc_smoothing_ratio_series.csv")
    futures = load_backtest_csv("data/raw/NIFTY_futures_continuous.csv")
    lot_schedule = load_backtest_csv("data/reference/nifty_lot_size_schedule.csv")
    predictions = load_backtest_csv("outputs/reports/regression_walkforward/regression_walkforward_predictions.csv")
    return ratio_core, ratio_dcc, futures, lot_schedule, predictions


def get_lot_size_for_date(lot_schedule, selected_date):
    if lot_schedule.empty:
        return 65, "Fallback lot size used because the lot-size schedule file is unavailable."

    schedule = lot_schedule.copy()
    schedule["effective_from"] = pd.to_datetime(schedule["effective_from"]).dt.date
    schedule["effective_to"] = pd.to_datetime(schedule["effective_to"]).dt.date

    selected = pd.to_datetime(selected_date).date()
    matched = schedule[
        (schedule["effective_from"] <= selected)
        & (schedule["effective_to"] >= selected)
    ]

    if matched.empty:
        return 65, "Fallback lot size used because no schedule row matched the selected date."

    row = matched.iloc[0]
    return int(row["lot_size"]), str(row.get("source_note", "Historical lot-size schedule."))


def infer_historical_planner_action(previous_ratio, target_ratio):
    previous_ratio = 0.0 if pd.isna(previous_ratio) else float(previous_ratio)
    target_ratio = 0.0 if pd.isna(target_ratio) else float(target_ratio)
    delta = target_ratio - previous_ratio

    if target_ratio < 0.05:
        return "CLOSE_HEDGE" if previous_ratio >= 0.05 else "HOLD_POSITION"
    if previous_ratio < 0.05:
        return "OPEN_HEDGE"
    if delta > 0.05:
        return "INCREASE_HEDGE"
    if delta < -0.05:
        return "REDUCE_HEDGE"
    return "HOLD_POSITION"


def infer_historical_trade_action(previous_contracts, target_contracts):
    """Infer planner action from the change in target futures position."""
    previous_contracts = int(previous_contracts)
    target_contracts = int(target_contracts)
    delta = target_contracts - previous_contracts

    if target_contracts == 0 and previous_contracts > 0:
        return "CLOSE_HEDGE"
    if previous_contracts == 0 and target_contracts > 0:
        return "OPEN_HEDGE"
    if delta > 0:
        return "INCREASE_HEDGE"
    if delta < 0:
        return "REDUCE_HEDGE"
    return "HOLD_POSITION"


def get_historical_prediction(predictions, selected_date):
    if predictions.empty or "date" not in predictions.columns or "prediction" not in predictions.columns:
        return None, "Not available"

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    selected_ts = pd.to_datetime(selected_date)

    pred = pred[pred["date"] <= selected_ts]
    if pred.empty:
        return None, "Not available"

    # Prefer the dissertation horizon of 10 trading days when available.
    if "horizon" in pred.columns:
        horizon_10 = pred[pred["horizon"] == 10]
        if not horizon_10.empty:
            pred = horizon_10

    latest_date = pred["date"].max()
    latest = pred[pred["date"] == latest_date]
    prediction_value = float(latest["prediction"].median())

    if prediction_value < 0.109575:
        regime = "Low"
    elif prediction_value >= 0.153290:
        regime = "High"
    else:
        regime = "Medium"

    return prediction_value, regime


def apply_hedra_risk_gate(candidate_ratio, risk_regime):
    """Convert a benchmark hedge ratio into the final Hedra target ratio.

    The benchmark ratio acts as a candidate hedge level. The final system
    gates that ratio using the predicted risk regime so that the system does
    not over-hedge in low-risk periods.
    """
    if candidate_ratio is None or pd.isna(candidate_ratio):
        candidate_ratio = 0.0

    candidate_ratio = max(float(candidate_ratio), 0.0)
    regime = str(risk_regime).strip().lower()

    if regime == "low":
        return 0.0, "Low-risk regime: close or avoid hedge exposure."

    if regime == "medium":
        return min(candidate_ratio, 0.50), "Medium-risk regime: allow a moderate hedge capped at 50%."

    if regime == "high":
        return min(candidate_ratio, 0.80), "High-risk regime: allow stronger hedge exposure capped at 80%."

    return min(candidate_ratio, 0.50), "Prediction unavailable: conservative hedge cap of 50% applied."


def render_historical_hedge_plan_replay():
    st.markdown(
        "<div class='section-header'>Historical Hedge Plan Replay</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hedge-panel'>"
        "<div class='placeholder-title'>Planner replay from historical hedge ratios</div>"
        "<div class='placeholder-text'>"
        "This replay translates saved historical hedge ratios into the same decision style used by the live system. "
        "It uses historical NIFTY futures close prices and the lot-size schedule to estimate contract value, "
        "recommended futures contracts, margin requirement, and planner action for a selected date."
        "</div></div>",
        unsafe_allow_html=True,
    )

    ratio_core, ratio_dcc, futures, lot_schedule, predictions = load_historical_replay_inputs()

    ratio_sources = {
        "Core hedge benchmark ratios": ratio_core,
        "DCC smoothing / no-trade-band ratios": ratio_dcc,
    }

    selected_source_name = st.selectbox(
        "Historical hedge-ratio source",
        list(ratio_sources.keys()),
        key="historical_replay_ratio_source",
    )

    ratio_df = ratio_sources[selected_source_name].copy()

    if ratio_df.empty:
        st.warning("The selected hedge-ratio dataset could not be loaded.")
        return

    if futures.empty:
        st.warning("Historical NIFTY futures data could not be loaded.")
        return

    ratio_df["date"] = pd.to_datetime(ratio_df["date"])
    futures = futures.copy()
    futures["date"] = pd.to_datetime(futures["date"])

    strategy_cols = [col for col in ratio_df.columns if col != "date"]

    default_strategy = (
        "gjr_scaled_minvar"
        if "gjr_scaled_minvar" in strategy_cols
        else "dcc_ema20_band0p10"
        if "dcc_ema20_band0p10" in strategy_cols
        else strategy_cols[0]
    )

    default_strategy_index = strategy_cols.index(default_strategy)

    date_options = sorted(ratio_df["date"].dt.strftime("%Y-%m-%d").unique().tolist())
    default_date = "2024-12-30" if "2024-12-30" in date_options else date_options[-1]
    default_date_index = date_options.index(default_date)

    control_left, control_right = st.columns(2)

    with control_left:
        selected_date = st.selectbox(
            "Historical date",
            date_options,
            index=default_date_index,
            key="historical_replay_date",
        )

    with control_right:
        selected_strategy = st.selectbox(
            "Candidate hedge-ratio model",
            strategy_cols,
            index=default_strategy_index,
            format_func=ui_label,
            key="historical_replay_strategy",
        )

    selected_ts = pd.to_datetime(selected_date)

    selected_row = ratio_df[ratio_df["date"] == selected_ts]
    if selected_row.empty:
        st.warning("No hedge-ratio row was found for the selected date.")
        return

    raw_target_ratio = float(selected_row.iloc[0][selected_strategy])

    previous_rows = ratio_df[ratio_df["date"] < selected_ts].sort_values("date")
    raw_previous_ratio = (
        float(previous_rows.iloc[-1][selected_strategy])
        if not previous_rows.empty
        else 0.0
    )
    previous_date_for_gate = (
        previous_rows.iloc[-1]["date"]
        if not previous_rows.empty
        else selected_ts
    )

    predicted_vol, risk_regime = get_historical_prediction(predictions, selected_date)
    previous_predicted_vol, previous_risk_regime = get_historical_prediction(
        predictions,
        previous_date_for_gate,
    )

    target_ratio, risk_gate_note = apply_hedra_risk_gate(raw_target_ratio, risk_regime)
    previous_ratio, previous_gate_note = apply_hedra_risk_gate(
        raw_previous_ratio,
        previous_risk_regime,
    )

    futures_rows = futures[futures["date"] <= selected_ts].sort_values("date")
    if futures_rows.empty:
        st.warning("No historical futures price was found on or before the selected date.")
        return

    futures_row = futures_rows.iloc[-1]
    futures_price = float(futures_row["close"])
    futures_price_date = futures_row["date"].strftime("%Y-%m-%d")

    previous_futures_rows = futures[futures["date"] <= previous_date_for_gate].sort_values("date")
    previous_futures_row = previous_futures_rows.iloc[-1] if not previous_futures_rows.empty else futures_row
    previous_futures_price = float(previous_futures_row["close"])
    previous_futures_price_date = previous_futures_row["date"].strftime("%Y-%m-%d")

    lot_size, lot_note = get_lot_size_for_date(lot_schedule, selected_date)
    previous_lot_size, _ = get_lot_size_for_date(
        lot_schedule,
        pd.to_datetime(previous_date_for_gate).strftime("%Y-%m-%d"),
    )

    portfolio_exposure = 8_000_000.0
    margin_reserve = 2_000_000.0
    margin_rate = 0.12

    contract_value = futures_price * lot_size
    previous_contract_value = previous_futures_price * previous_lot_size

    target_hedge_value = max(target_ratio, 0.0) * portfolio_exposure
    previous_target_hedge_value = max(previous_ratio, 0.0) * portfolio_exposure

    target_contracts = 0
    if target_hedge_value > 0 and contract_value > 0:
        target_contracts = int(round(target_hedge_value / contract_value))
        if target_contracts == 0:
            target_contracts = 1

    previous_target_contracts = 0
    if previous_target_hedge_value > 0 and previous_contract_value > 0:
        previous_target_contracts = int(round(previous_target_hedge_value / previous_contract_value))
        if previous_target_contracts == 0:
            previous_target_contracts = 1

    contracts_to_trade_now = target_contracts - previous_target_contracts
    recommended_contracts = target_contracts

    achieved_hedge_ratio = (
        (target_contracts * contract_value) / portfolio_exposure
        if portfolio_exposure > 0
        else 0.0
    )

    margin_required = target_contracts * contract_value * margin_rate
    margin_available_after_trade = margin_reserve - margin_required
    margin_ok = margin_required <= margin_reserve

    planner_action = infer_historical_trade_action(previous_target_contracts, target_contracts)

    metric_cols = st.columns(5)

    with metric_cols[0]:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Planner Action</div>"
            f"<div class='metric-value'>{escape(ui_label(planner_action))}</div></div>",
            unsafe_allow_html=True,
        )

    with metric_cols[1]:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Hedra Target Ratio</div>"
            f"<div class='metric-value'>{target_ratio:.2%}</div></div>",
            unsafe_allow_html=True,
        )

    with metric_cols[2]:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Target Hedge Contracts</div>"
            f"<div class='metric-value'>{recommended_contracts}</div></div>",
            unsafe_allow_html=True,
        )

    with metric_cols[3]:
        trade_label = f"{contracts_to_trade_now:+d}"
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Contracts to Trade Now</div>"
            f"<div class='metric-value'>{trade_label}</div></div>",
            unsafe_allow_html=True,
        )

    with metric_cols[4]:
        margin_status = "Sufficient" if margin_ok else "Insufficient"
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Margin Check</div>"
            f"<div class='metric-value'>{margin_status}</div></div>",
            unsafe_allow_html=True,
        )

    replay_summary = pd.DataFrame(
        [
            ["Selected date", selected_date],
            ["Hedge-ratio source", selected_source_name],
            ["Candidate hedge-ratio model", ui_label(selected_strategy)],
            ["Risk regime source", "Regression walk-forward volatility prediction"],
            ["Previous raw candidate hedge ratio", f"{raw_previous_ratio:.2%}"],
            ["Previous Hedra target hedge ratio", f"{previous_ratio:.2%}"],
            ["Raw candidate hedge ratio", f"{raw_target_ratio:.2%}"],
            ["Hedra risk-gated target ratio", f"{target_ratio:.2%}"],
            ["Risk gate rule", risk_gate_note],
            ["Previous risk gate rule", previous_gate_note],
            ["Achieved hedge ratio", f"{achieved_hedge_ratio:.2%}"],
            ["Historical futures price date", futures_price_date],
            ["Historical futures close", f"₹{futures_price:,.2f}"],
            ["Historical lot size", f"{lot_size}"],
            ["Estimated contract value", f"₹{contract_value:,.2f}"],
            ["Target hedge value", f"₹{target_hedge_value:,.2f}"],
            ["Previous target contracts", f"{previous_target_contracts}"],
            ["Target hedge contracts", f"{target_contracts}"],
            ["Contracts to trade now", f"{contracts_to_trade_now:+d}"],
            ["Previous futures price date", previous_futures_price_date],
            ["Estimated margin required", f"₹{margin_required:,.2f}"],
            ["Margin reserve", f"₹{margin_reserve:,.2f}"],
            ["Margin after trade", f"₹{margin_available_after_trade:,.2f}"],
            ["Predicted volatility", "Not available" if predicted_vol is None else f"{predicted_vol:.2%}"],
            ["Risk regime", risk_regime],
            ["Lot-size note", lot_note],
        ],
        columns=["Field", "Value"],
    )

    st.markdown(
        "<div class='section-header'>Replay Details</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(replay_summary, hide_index=True, width="stretch")

    window_start = selected_ts - pd.Timedelta(days=120)
    window_end = selected_ts + pd.Timedelta(days=30)
    ratio_window = ratio_df[
        (ratio_df["date"] >= window_start)
        & (ratio_df["date"] <= window_end)
    ][["date", selected_strategy]].copy()

    if not ratio_window.empty:
        chart_rows = []
        for _, row in ratio_window.iterrows():
            row_date = row["date"]
            raw_ratio = float(row[selected_strategy])
            _, row_regime = get_historical_prediction(predictions, row_date)
            hedra_ratio, _ = apply_hedra_risk_gate(raw_ratio, row_regime)
            chart_rows.append(
                {
                    "date": row_date,
                    "Raw Candidate Hedge Ratio": raw_ratio,
                    "Hedra Risk-Gated Target Ratio": hedra_ratio,
                }
            )

        ratio_chart = pd.DataFrame(chart_rows).set_index("date")

        st.markdown(
            "<div class='section-header'>Raw Candidate vs Hedra Risk-Gated Target</div>",
            unsafe_allow_html=True,
        )
        st.line_chart(ratio_chart, width="stretch")

    trace_lines = [
        f"Selected strategy is {ui_label(selected_strategy)} on {selected_date}.",
        f"The raw candidate hedge ratio is {raw_target_ratio:.2%}.",
        f"The predicted risk regime is {risk_regime}, so Hedra applies the rule: {risk_gate_note}",
        f"The previous target position was {previous_target_contracts} contract(s), and the current target position is {target_contracts} contract(s).",
        f"Contracts to trade now: {contracts_to_trade_now:+d}, so the inferred planner action is {ui_label(planner_action)}.",
        f"The historical NIFTY futures close used is ₹{futures_price:,.2f} from {futures_price_date}.",
        f"The lot size applied for this date is {lot_size}, giving an estimated contract value of ₹{contract_value:,.2f}.",
        f"With ₹80,00,000 stock exposure, the Hedra target hedge value is ₹{target_hedge_value:,.2f}.",
        f"The target hedge position is {target_contracts} short NIFTY futures contract(s), with estimated margin requirement of ₹{margin_required:,.2f}.",
    ]

    st.session_state["historical_replay_context"] = {
        "selected_date": str(selected_date),
        "hedge_ratio_source": selected_source_name,
        "candidate_model": ui_label(selected_strategy),
        "planner_action": ui_label(planner_action),
        "raw_candidate_ratio": f"{raw_target_ratio:.2%}",
        "previous_target_ratio": f"{previous_ratio:.2%}",
        "target_ratio": f"{target_ratio:.2%}",
        "achieved_hedge_ratio": f"{achieved_hedge_ratio:.2%}",
        "risk_regime": str(risk_regime),
        "predicted_volatility": "Not available" if predicted_vol is None else f"{predicted_vol:.2%}",
        "risk_gate_note": risk_gate_note,
        "previous_gate_note": previous_gate_note,
        "previous_contracts": str(previous_target_contracts),
        "target_contracts": str(target_contracts),
        "contracts_to_trade_now": f"{contracts_to_trade_now:+d}",
        "margin_status": "Sufficient" if margin_ok else "Insufficient",
        "margin_required": f"₹{margin_required:,.2f}",
        "margin_reserve": f"₹{margin_reserve:,.2f}",
        "margin_after_trade": f"₹{margin_available_after_trade:,.2f}",
        "futures_price_date": futures_price_date,
        "futures_close": f"₹{futures_price:,.2f}",
        "lot_size": str(lot_size),
        "contract_value": f"₹{contract_value:,.2f}",
        "target_hedge_value": f"₹{target_hedge_value:,.2f}",
        "lot_note": lot_note,
        "trace_lines": trace_lines,
    }

    st.markdown(
        "<div class='section-header'>Replay Explanation Trace</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hedge-panel'>"
        "<div class='placeholder-text'>"
        "The selected hedge-ratio model provides only the candidate hedge size. "
        "The risk regime is taken from the volatility prediction layer and is used by Hedra to gate the final hedge target."
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='hedge-panel'>"
        + "".join(
            f"<div class='trace-line'>{escape(line)}</div>"
            for line in trace_lines
        )
        + "</div>",
        unsafe_allow_html=True,
    )




def render_historical_explanation():
    st.markdown(
        "<div class='section-header'>Historical Explanation</div>",
        unsafe_allow_html=True,
    )

    ctx = st.session_state.get("historical_replay_context")

    if not ctx:
        st.markdown(
            "<div class='hedge-panel'>"
            "<div class='placeholder-title'>No replay decision selected yet</div>"
            "<div class='placeholder-text'>"
            "Open the Hedge Plan tab first and select a historical replay date. "
            "The Explanation tab will then explain that selected backtest hedge decision."
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    margin_colour = "#4ade80" if ctx.get("margin_status") == "Sufficient" else "#f87171"

    st.markdown(
        "<div class='hedge-panel'>"
        "<div class='placeholder-title'>Selected historical replay decision</div>"
        "<div class='placeholder-text'>"
        "This explanation is linked to the currently selected historical replay in the Hedge Plan tab."
        "</div>"
        "<div style='height:1px;background:#2a2a30;margin:18px 0;'></div>"

        "<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Selected Date</div>"
        "<div class='stat-value-sm'>" + escape(ctx.get("selected_date", "N/A")) + "</div>"
        "<div class='stat-meta'>" + escape(ctx.get("candidate_model", "N/A")) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Planner Action</div>"
        "<div class='stat-value-sm'>" + escape(ctx.get("planner_action", "N/A")) + "</div>"
        "<div class='stat-meta'>Contracts to trade: " + escape(ctx.get("contracts_to_trade_now", "N/A")) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Risk Regime</div>"
        "<div class='stat-value-sm'>" + escape(ctx.get("risk_regime", "N/A")) + "</div>"
        "<div class='stat-meta'>Predicted volatility: " + escape(ctx.get("predicted_volatility", "N/A")) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Hedra Target Ratio</div>"
        "<div class='stat-value-sm'>" + escape(ctx.get("target_ratio", "N/A")) + "</div>"
        "<div class='stat-meta'>Achieved: " + escape(ctx.get("achieved_hedge_ratio", "N/A")) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Margin Check</div>"
        "<div class='stat-value-sm' style='color:" + margin_colour + ";'>" + escape(ctx.get("margin_status", "N/A")) + "</div>"
        "<div class='stat-meta'>Required: " + escape(ctx.get("margin_required", "N/A")) + "</div>"
        "</div>"

        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-header'>Why this decision was generated</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:18px;'>"

        "<div style='background:#111116;border:1px solid #2a2a30;border-radius:10px;padding:16px;'>"
        "<div class='placeholder-title'>1. Risk-gated hedge target</div>"
        "<div class='placeholder-text'>"
        "The selected hedge-ratio model produced a raw candidate hedge ratio of <strong>" + escape(ctx.get("raw_candidate_ratio", "N/A")) + "</strong>. "
        "The volatility layer classified the date as <strong>" + escape(ctx.get("risk_regime", "N/A")) + "</strong>, so Hedra applied this rule: "
        + escape(ctx.get("risk_gate_note", "N/A")) +
        "</div>"
        "</div>"

        "<div style='background:#111116;border:1px solid #2a2a30;border-radius:10px;padding:16px;'>"
        "<div class='placeholder-title'>2. Planner action</div>"
        "<div class='placeholder-text'>"
        "The previous target hedge ratio was <strong>" + escape(ctx.get("previous_target_ratio", "N/A")) + "</strong>, while the selected date target is "
        "<strong>" + escape(ctx.get("target_ratio", "N/A")) + "</strong>. This changed the target position from "
        "<strong>" + escape(ctx.get("previous_contracts", "N/A")) + "</strong> to <strong>" + escape(ctx.get("target_contracts", "N/A")) + "</strong> contracts, "
        "resulting in the planner action <strong>" + escape(ctx.get("planner_action", "N/A")) + "</strong>."
        "</div>"
        "</div>"

        "<div style='background:#111116;border:1px solid #2a2a30;border-radius:10px;padding:16px;'>"
        "<div class='placeholder-title'>3. Futures execution sizing</div>"
        "<div class='placeholder-text'>"
        "The execution layer used the historical futures close of <strong>" + escape(ctx.get("futures_close", "N/A")) + "</strong> from "
        "<strong>" + escape(ctx.get("futures_price_date", "N/A")) + "</strong>, with lot size <strong>" + escape(ctx.get("lot_size", "N/A")) + "</strong>. "
        "This gives an estimated contract value of <strong>" + escape(ctx.get("contract_value", "N/A")) + "</strong> and a target hedge value of "
        "<strong>" + escape(ctx.get("target_hedge_value", "N/A")) + "</strong>."
        "</div>"
        "</div>"

        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    decision_rows = [
        ["Selected date", ctx.get("selected_date", "N/A")],
        ["Hedge-ratio source", ctx.get("hedge_ratio_source", "N/A")],
        ["Candidate model", ctx.get("candidate_model", "N/A")],
        ["Risk regime", ctx.get("risk_regime", "N/A")],
        ["Predicted volatility", ctx.get("predicted_volatility", "N/A")],
        ["Raw candidate hedge ratio", ctx.get("raw_candidate_ratio", "N/A")],
        ["Final Hedra target ratio", ctx.get("target_ratio", "N/A")],
        ["Target hedge contracts", ctx.get("target_contracts", "N/A")],
        ["Contracts to trade now", ctx.get("contracts_to_trade_now", "N/A")],
        ["Margin status", ctx.get("margin_status", "N/A")],
        ["Margin required", ctx.get("margin_required", "N/A")],
        ["Margin reserve", ctx.get("margin_reserve", "N/A")],
        ["Margin after trade", ctx.get("margin_after_trade", "N/A")],
        ["Lot-size note", ctx.get("lot_note", "N/A")],
    ]

    st.markdown(
        "<div class='section-header'>Decision Factors</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        pd.DataFrame(decision_rows, columns=["Decision Factor", "Value"]),
        hide_index=True,
        width="stretch",
    )

    trace_lines = ctx.get("trace_lines", [])

    with st.expander("View replay technical trace", expanded=False):
        if trace_lines:
            st.markdown(
                "<div class='hedge-panel'>"
                + "".join(
                    f"<div class='trace-line'>{escape(str(line))}</div>"
                    for line in trace_lines
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No replay trace is available for this selected decision.")




def render_backtest_evaluation():
    st.markdown(
        "<div class='section-header'>Historical Benchmark Evaluation</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hedge-panel'>"
        "<div class='placeholder-title'>Benchmark module</div>"
        "<div class='placeholder-text'>"
        "This section compares saved benchmark hedge strategies such as no hedge, static hedge, rolling minimum-variance, "
        "DCC-GARCH, and smoothed DCC no-trade-band variants. The planning-based historical hedge replay is handled separately "
        "through the Hedge Plan workflow."
        "</div></div>",
        unsafe_allow_html=True,
    )

    catalog = get_backtest_dataset_catalog()
    dataset_name = st.selectbox(
        "Backtest dataset",
        list(catalog.keys()),
        key="backtest_dataset_select",
    )

    config = catalog[dataset_name]
    summary_df = load_backtest_csv(config["summary"])
    daily_df = load_backtest_csv(config["daily"])
    ratio_df = load_backtest_csv(config["ratio"])

    if summary_df.empty:
        st.markdown(
            "<div class='hedge-panel'><div class='placeholder-title'>Dataset unavailable</div>"
            "<div class='placeholder-text'>The selected backtest summary file could not be loaded.</div></div>",
            unsafe_allow_html=True,
        )
        return

    best_vol = safe_metric_value(summary_df, "volatility_pct", "min")
    best_dd = safe_metric_value(summary_df, "max_drawdown_pct", "max")
    best_rank = safe_metric_value(summary_df, "balanced_rank_score", "min")
    no_hedge = summary_df[summary_df["label"].astype(str).str.lower().eq("no_hedge")]
    no_hedge_row = no_hedge.iloc[0] if not no_hedge.empty else None

    def row_label(row):
        if row is None:
            return "N/A"
        return escape(ui_label(row.get("label", "N/A")))

    def row_value(row, column, suffix=""):
        if row is None or column not in row:
            return "N/A"
        try:
            return f"{float(row[column]):,.2f}{suffix}"
        except Exception:
            return escape(str(row[column]))

    st.markdown(
        "<div class='hedge-panel'>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:18px;'>"

        "<div>"
        "<div class='stat-label'>Dataset</div>"
        "<div class='stat-value-sm'>" + escape(dataset_name) + "</div>"
        "<div class='stat-meta'>" + escape(config["description"]) + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Lowest Volatility</div>"
        "<div class='stat-value-sm'>" + row_label(best_vol) + "</div>"
        "<div class='stat-meta'>" + row_value(best_vol, "volatility_pct", "%") + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>Lowest Drawdown</div>"
        "<div class='stat-value-sm'>" + row_label(best_dd) + "</div>"
        "<div class='stat-meta'>" + row_value(best_dd, "max_drawdown_pct", "%") + "</div>"
        "</div>"

        "<div>"
        "<div class='stat-label'>No-Hedge Return</div>"
        "<div class='stat-value-sm'>" + row_value(no_hedge_row, "total_return_pct", "%") + "</div>"
        "<div class='stat-meta'>Reference baseline</div>"
        "</div>"

        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    strategies = list(summary_df["label"].astype(str).unique())
    preferred = [
        "no_hedge",
        "static_50",
        "rolling_min_variance",
        "gjr_scaled_minvar",
        "dcc_raw",
        "dcc_ema20_band0p10",
    ]
    default_strategies = [s for s in preferred if s in strategies]
    if not default_strategies:
        default_strategies = strategies[: min(5, len(strategies))]

    selected_strategies = st.multiselect(
        "Strategies to display",
        strategies,
        default=default_strategies,
        format_func=ui_label,
        key="backtest_strategy_multiselect",
    )

    display_cols = [
        "label",
        "total_return_pct",
        "annual_return_pct",
        "volatility_pct",
        "sharpe",
        "max_drawdown_pct",
        "cvar_95_pct",
        "average_hedge_ratio",
        "average_daily_turnover",
        "total_transaction_cost_pct",
        "hedging_effectiveness",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    table_df = summary_df[summary_df["label"].astype(str).isin(selected_strategies)][display_cols].copy()

    if "label" in table_df.columns:
        table_df.insert(0, "Strategy", table_df["label"].apply(ui_label))
        table_df = table_df.drop(columns=["label"])

    readable_columns = {
        "total_return_pct": "Total Return (%)",
        "annual_return_pct": "Annual Return (%)",
        "volatility_pct": "Volatility (%)",
        "sharpe": "Sharpe",
        "max_drawdown_pct": "Max Drawdown (%)",
        "cvar_95_pct": "CVaR 95% (%)",
        "average_hedge_ratio": "Average Hedge Ratio",
        "average_daily_turnover": "Average Daily Turnover",
        "total_transaction_cost_pct": "Transaction Cost (%)",
        "hedging_effectiveness": "Hedging Effectiveness",
    }
    table_df = table_df.rename(columns=readable_columns)

    st.markdown(
        "<div class='section-header'>Strategy Comparison</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(table_df, width="stretch", hide_index=True)

    if not daily_df.empty and {"date", "strategy", "hedged_return_net"}.issubset(daily_df.columns):
        chart_df = daily_df[daily_df["strategy"].astype(str).isin(selected_strategies)].copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
        chart_df = chart_df.dropna(subset=["date"])

        pivot = chart_df.pivot_table(
            index="date",
            columns="strategy",
            values="hedged_return_net",
            aggfunc="first",
        ).sort_index()

        cumulative = ((1 + pivot.fillna(0)).cumprod() - 1) * 100
        cumulative_display = cumulative.rename(
            columns={col: ui_label(col) for col in cumulative.columns}
        )

        st.markdown(
            "<div class='section-header'>Cumulative Net Return</div>",
            unsafe_allow_html=True,
        )
        st.line_chart(cumulative_display, width="stretch")

    if not ratio_df.empty and "date" in ratio_df.columns:
        ratio_plot = ratio_df.copy()
        ratio_plot["date"] = pd.to_datetime(ratio_plot["date"], errors="coerce")
        ratio_plot = ratio_plot.dropna(subset=["date"]).set_index("date")

        ratio_options = [c for c in ratio_plot.columns if pd.api.types.is_numeric_dtype(ratio_plot[c])]
        ratio_defaults = [s for s in selected_strategies if s in ratio_options]
        if not ratio_defaults:
            ratio_defaults = ratio_options[: min(5, len(ratio_options))]

        selected_ratio_cols = st.multiselect(
            "Hedge ratio series",
            ratio_options,
            default=ratio_defaults,
            format_func=ui_label,
            key="backtest_ratio_multiselect",
        )

        if selected_ratio_cols:
            ratio_display = ratio_plot[selected_ratio_cols].rename(
                columns={col: ui_label(col) for col in selected_ratio_cols}
            )

            st.markdown(
                "<div class='section-header'>Historical Hedge Ratio Series</div>",
                unsafe_allow_html=True,
            )
            st.line_chart(ratio_display, width="stretch")



def render_dashboard_home(kite, profile):
    st.markdown(get_css(auth_page=False), unsafe_allow_html=True)
    render_custom_sidebar(profile, token_valid=True)

    now        = datetime.now(IST).strftime("%d %b %Y, %H:%M IST")
    token_raw  = os.getenv("KITE_ACCESS_TOKEN", "")
    tok_prev   = (token_raw[:8] + "&#8226;&#8226;&#8226;&#8226;" + token_raw[-4:]) if token_raw else "—"
    user_name  = escape(profile.get("user_name", "—"))
    user_email = escape(profile.get("email", "—"))

    st.markdown(
        "<p class='page-title'>Dashboard</p>"
        "<p class='page-sub'>" + now + "</p>",
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Operating Mode",
        ["Live Monitoring", "Historical Backtesting"],
        horizontal=True,
        key="operating_mode",
        label_visibility="collapsed",
    )

    if mode == "Live Monitoring":
        tab_overview, tab_portfolio, tab_market, tab_ml, tab_plan, tab_explain = st.tabs([
            "Overview",
            "Portfolio",
            "Market Data",
            "ML Prediction",
            "Hedge Plan",
            "Explanation",
        ])

        with tab_overview:
            render_live_dashboard_content(kite, profile, now, tok_prev, user_name, user_email)

        with tab_portfolio:
            render_live_portfolio_content(kite)

        with tab_market:
            render_market_data_content(kite, mode)

        with tab_ml:
            render_ml_prediction_content(kite, mode)

        with tab_plan:
            render_live_hedge_plan(kite)

        with tab_explain:
            render_live_explanation(kite)

    else:
        tab_plan, tab_explain, tab_eval = st.tabs([
            "Hedge Plan",
            "Explanation",
            "Benchmark Evaluation",
        ])

        with tab_plan:
            render_historical_hedge_plan_replay()

        with tab_explain:
            render_historical_explanation()

        with tab_eval:
            render_backtest_evaluation()


def main():
    if not API_KEY or not API_SECRET:
        st.error("KITE_API_KEY or KITE_API_SECRET not found in .env.")
        st.stop()

    params        = st.query_params
    request_token = params.get("request_token")
    kite_error    = params.get("error")

    if kite_error:
        render_auth_page(state="error", error=kite_error)
        return

    if request_token:
        try:
            access_token = exchange_and_save_token(request_token)
            load_dotenv(override=True)
            st.query_params.clear()
            render_auth_page(state="success", token=access_token)
            st.rerun()
        except Exception as e:
            render_auth_page(state="error", error=str(e))
        return

    if not is_token_present():
        render_auth_page(state="login")
        return

    kite          = get_kite_client()
    valid, result = verify_token(kite)

    if not valid:
        render_auth_page(state="error", error="Session expired. Please re-authenticate.")
        return

    render_dashboard_home(kite, result)


if __name__ == "__main__":
    main()




