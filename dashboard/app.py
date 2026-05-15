"""
app.py

Entry point for Hedra — the dynamic risk hedging dashboard.

Flow:
    No token present  → authentication page
    Token present     → dashboard home (system status + portfolio + sector chart)
    Kite callback     → exchange request_token, save to .env, show dashboard home

Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import math
from datetime import datetime, timezone, timedelta, date as dt_date
from html import escape

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect
import streamlit as st

load_dotenv()

API_KEY    = os.getenv("KITE_API_KEY")
API_SECRET = os.getenv("KITE_API_SECRET")
ENV_PATH   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))

IST               = timezone(timedelta(hours=5, minutes=30))
ENTRY_DATE        = dt_date(2025, 4, 1)
CAPITAL_PER_STOCK = 100_000  # ₹1,00,000 per stock → ₹10,00,000 total

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

SECTOR_COLOURS = {
    "Financials":  "#6c63ff",
    "Technology":  "#4ade80",
    "Energy":      "#fbbf24",
    "Industrials": "#60a5fa",
    "Telecom":     "#f87171",
    "FMCG":        "#a78bfa",
}

st.set_page_config(
    page_title="Hedra",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True


# ── XIRR ─────────────────────────────────────────────────────────────────────

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


# ── CSS ───────────────────────────────────────────────────────────────────────

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
</style>
"""


# ── Auth HTML ─────────────────────────────────────────────────────────────────

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


# ── Kite helpers ──────────────────────────────────────────────────────────────

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


# ── Auth page ─────────────────────────────────────────────────────────────────

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


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_custom_sidebar(profile, token_valid=True):
    name  = escape(profile.get("user_name", "—")) if profile else "—"
    email = escape(profile.get("email", "—"))     if profile else "—"

    if st.button("☰", key="sidebar_toggle"):
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
            "<div class='sidebar-label'>Pages</div>"
            "<div class='sidebar-page'>Overview</div>"
            "<div class='sidebar-page'>Portfolio</div>"
            "<div class='sidebar-page sidebar-page-disabled'>ML Prediction <span class='sidebar-soon'>soon</span></div>"
            "<div class='sidebar-page sidebar-page-disabled'>SHAP <span class='sidebar-soon'>soon</span></div>"
            "<div class='sidebar-page sidebar-page-disabled'>Audit Log <span class='sidebar-soon'>soon</span></div>"
            "</div>"
        )
    else:
        sidebar_html = (
            "<div class='custom-sidebar'>"
            "<div class='sidebar-logo-box' style='margin-bottom:20px;'>H</div>"
            "<div class='sidebar-line'></div>"
            "<div class='sidebar-icon-item'>&#8962;</div>"
            "<div class='sidebar-icon-item'>&#9638;</div>"
            "<div class='sidebar-icon-item sidebar-icon-disabled'>&#9670;</div>"
            "<div class='sidebar-icon-item sidebar-icon-disabled'>&#9678;</div>"
            "<div class='sidebar-icon-item sidebar-icon-disabled'>&#9711;</div>"
            "</div>"
        )

    st.markdown(sidebar_html, unsafe_allow_html=True)


# ── Sector pie chart ──────────────────────────────────────────────────────────

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


# ── Dashboard home ────────────────────────────────────────────────────────────

def render_dashboard_home(kite, profile):
    st.markdown(get_css(auth_page=False), unsafe_allow_html=True)
    render_custom_sidebar(profile, token_valid=True)

    now        = datetime.now(IST).strftime("%d %b %Y, %H:%M IST")
    token_raw  = os.getenv("KITE_ACCESS_TOKEN", "")
    tok_prev   = (token_raw[:8] + "&#8226;&#8226;&#8226;&#8226;" + token_raw[-4:]) if token_raw else "—"
    user_name  = escape(profile.get("user_name", "—"))
    user_email = escape(profile.get("email", "—"))

    st.markdown(
        "<p class='page-title'>Overview</p>"
        "<p class='page-sub'>" + now + "</p>",
        unsafe_allow_html=True,
    )

    # ── Status cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            "<div class='stat-card'><div class='stat-label'>Kite Connection</div>"
            "<div class='stat-value'>Active</div>"
            "<div class='stat-badge badge-green'>"
            "<span style='width:5px;height:5px;border-radius:50%;background:#4ade80;"
            "display:inline-block;'></span>&nbsp;Connected</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='stat-card'><div class='stat-label'>Authenticated As</div>"
            "<div class='stat-value-sm'>" + user_name + "</div>"
            "<div class='stat-meta'>" + user_email + "</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='stat-card'><div class='stat-label'>Access Token</div>"
            "<div style='font-size:13px;font-weight:500;color:#6c63ff;"
            "font-family:Courier New,monospace;letter-spacing:0.5px;'>" + tok_prev + "</div>"
            "<div class='stat-meta'>Expires 6:00 AM IST</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            "<div class='stat-card'><div class='stat-label'>Portfolio Capital</div>"
            "<div class='stat-value' style='font-size:16px;'>&#8377;10,00,000</div>"
            "<div class='stat-meta'>Equal weight &middot; Entry 01 Apr 2025</div></div>",
            unsafe_allow_html=True,
        )

    # ── Portfolio + Sector chart ──────────────────────────────────────────────
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

    # ── Compute portfolio totals before rendering columns ─────────────────────
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
    xirr_html = f"{pnl_sign}{xirr_val:.1f}% p.a." if xirr_val is not None else "—"
    xirr_col  = "#4ade80" if xirr_val and xirr_val >= 0 else "#f87171"

    col_portfolio, col_chart = st.columns([3, 2])

    # ── Left: portfolio table ─────────────────────────────────────────────────
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

    # ── Right: summary at top, pie chart below ────────────────────────────────
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

    # ── Last hedge decision ───────────────────────────────────────────────────
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
            rc   = str(last.get("risk_class", "—")).lower()
            rcol = {"low": "#4ade80", "medium": "#fbbf24", "high": "#f87171"}.get(rc, "#9090a0")

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.markdown(
                    "<div class='stat-card'><div class='stat-label'>Date</div>"
                    "<div class='stat-value-sm'>" + str(last.get("date", "—")) + "</div></div>",
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
                    "<div class='stat-value-sm'>" + str(last.get("hedge_action", "—")) + "</div></div>",
                    unsafe_allow_html=True,
                )
            with d4:
                st.markdown(
                    "<div class='stat-card'><div class='stat-label'>Hedge Ratio</div>"
                    "<div class='stat-value'>" + str(last.get("hedge_ratio", "—")) + "</div></div>",
                    unsafe_allow_html=True,
                )
            return

    st.markdown(
        "<div class='hedge-panel'>"
        "<div class='hedge-placeholder'>"
        "No hedge decisions yet &mdash; model pipeline not yet active"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

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