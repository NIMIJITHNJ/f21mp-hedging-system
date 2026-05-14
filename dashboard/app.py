"""
app.py

Entry point for Hedra — dynamic risk hedging dashboard.

Run:
    streamlit run dashboard/app.py
"""

import os
from datetime import datetime
from html import escape

from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect
import streamlit as st
from datetime import timezone, timedelta


# ── Environment ──────────────────────────────────────────────────────────────

load_dotenv()

API_KEY = os.getenv("KITE_API_KEY")
API_SECRET = os.getenv("KITE_API_SECRET")

ENV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".env")
)

PORTFOLIO = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY",
    "TCS", "SBIN", "LT", "BHARTIARTL", "ITC", "AXISBANK",
]


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hedra",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Session state ────────────────────────────────────────────────────────────

if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True


# ── CSS ──────────────────────────────────────────────────────────────────────

def get_css(auth_page=False):
    sidebar_width = "280px" if st.session_state.sidebar_open else "72px"
    main_padding = "40px" if auth_page else ("340px" if st.session_state.sidebar_open else "120px")

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-color: #0d0d0f !important;
    font-family: 'Inter', sans-serif !important;
}}

/* Hide Streamlit default UI */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
#MainMenu,
footer {{
    display: none !important;
}}

/* Main content */
[data-testid="stMainBlockContainer"] {{
    padding: 36px 40px 40px {main_padding} !important;
    max-width: 100% !important;
}}

/* Toggle button */
div[data-testid="stButton"] {{
    position: fixed;
    top: 18px;
    left: 18px;
    z-index: 10000;
}}

div[data-testid="stButton"] button {{
    background: #1e1e28 !important;
    color: #f0f0f5 !important;
    border: 1px solid #2a2a30 !important;
    border-radius: 8px !important;
    width: 38px !important;
    height: 34px !important;
    padding: 0 !important;
    font-size: 18px !important;
}}

/* Custom sidebar */
.custom-sidebar {{
    position: fixed;
    top: 0;
    left: 0;
    width: {sidebar_width};
    height: 100vh;
    background: #16161a;
    border-right: 1px solid #2a2a30;
    padding: 76px 22px 34px 22px;
    box-sizing: border-box;
    z-index: 9999;
    transition: width 0.25s ease;
    overflow: hidden;
}}

.sidebar-logo {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
}}

.sidebar-logo-box {{
    width: 38px;
    height: 38px;
    min-width: 38px;
    border-radius: 9px;
    background: #6c63ff;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
}}

.sidebar-title {{
    font-size: 18px;
    font-weight: 600;
    color: #f0f0f5;
    white-space: nowrap;
}}

.sidebar-subtitle {{
    font-size: 10px;
    color: #6c63ff;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-top: 3px;
    white-space: nowrap;
}}

.sidebar-line {{
    height: 1px;
    background: #2a2a30;
    margin: 22px 0;
}}

.sidebar-label {{
    font-size: 11px;
    color: #555560;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 8px;
}}

.sidebar-name {{
    font-size: 13px;
    color: #f0f0f5;
    font-weight: 500;
    margin-bottom: 2px;
    white-space: nowrap;
}}

.sidebar-email {{
    font-size: 12px;
    color: #555560;
    margin-bottom: 14px;
    white-space: nowrap;
}}

.sidebar-pill {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 11px;
    border-radius: 18px;
    background: #1e1e28;
    border: 1px solid #2a2a30;
    font-size: 11px;
    color: #4ade80;
    white-space: nowrap;
}}

.sidebar-dot {{
    width: 6px;
    height: 6px;
    background: #4ade80;
    border-radius: 50%;
}}

.sidebar-page {{
    padding: 10px 0;
    border-bottom: 1px solid #1e1e28;
    font-size: 13px;
    color: #9090a0;
    white-space: nowrap;
}}

.sidebar-page-disabled {{
    color: #3a3a3a;
}}

.sidebar-soon {{
    font-size: 10px;
    color: #3a3a3a;
    margin-left: 4px;
}}

.sidebar-icon-item {{
    color: #9090a0;
    font-size: 19px;
    margin-bottom: 24px;
    text-align: center;
}}

.sidebar-icon-disabled {{
    color: #3a3a3a;
}}

/* Auth page */
.page-wrap {{
    min-height: 100vh;
    background: #0d0d0f;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
}}

.auth-card {{
    background: #16161a;
    border: 1px solid #2a2a30;
    border-radius: 16px;
    padding: 48px 44px;
    width: 100%;
    max-width: 420px;
}}

.logo-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 32px;
}}

.logo-hex {{
    width: 36px;
    height: 36px;
    background: #6c63ff;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 600;
    color: white;
}}

.logo-name {{
    font-size: 20px;
    font-weight: 600;
    color: #f0f0f5;
}}

.logo-tag {{
    font-size: 11px;
    color: #6c63ff;
    font-weight: 500;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}

.divider {{
    height: 1px;
    background: #2a2a30;
    margin-bottom: 28px;
}}

.section-label {{
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #555560;
    margin-bottom: 16px;
}}

.body-text {{
    font-size: 14px;
    color: #9090a0;
    line-height: 1.6;
    margin: 0 0 24px;
}}

.caption {{
    font-size: 12px;
    color: #555560;
    margin: 16px 0 0;
}}

.btn-primary {{
    display: block;
    width: 100%;
    background: #6c63ff;
    color: #ffffff !important;
    text-decoration: none;
    padding: 12px 0;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    text-align: center;
}}

.btn-primary:hover {{
    background: #5a52e0;
}}

.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}}

.status-pill.success {{
    background: #0d2b1a;
    color: #4ade80;
    border: 1px solid #1a4a2a;
}}

.status-pill.error {{
    background: #2b0d0d;
    color: #f87171;
    border: 1px solid #4a1a1a;
}}

.dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}}

.token-preview {{
    background: #0d0d0f;
    border: 1px solid #2a2a30;
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #6c63ff;
    margin: 16px 0 8px;
}}

.error-box {{
    background: #1a0808;
    border: 1px solid #4a1a1a;
    border-radius: 8px;
    padding: 12px 14px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #f87171;
    margin: 16px 0;
    word-break: break-word;
}}

.feature-row {{
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 28px;
    padding-top: 24px;
    border-top: 1px solid #2a2a30;
}}

.feature-item {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
}}

.feature-icon {{
    width: 28px;
    height: 28px;
    background: #1e1e28;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #6c63ff;
}}

.feature-text {{
    font-size: 12px;
    color: #555560;
    line-height: 1.5;
}}

.feature-text strong {{
    color: #9090a0;
    font-weight: 500;
}}

/* Dashboard */
.page-title {{
    font-size: 22px;
    font-weight: 600;
    color: #f0f0f5;
    margin: 0 0 4px;
}}

.page-sub {{
    font-size: 12px;
    color: #555560;
    margin: 0 0 32px;
}}

.stat-card {{
    background: #16161a;
    border: 1px solid #2a2a30;
    border-radius: 12px;
    padding: 18px 20px;
    height: 110px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}

.stat-label {{
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    color: #555560;
}}

.stat-value {{
    font-size: 20px;
    font-weight: 600;
    color: #f0f0f5;
}}

.stat-value-sm {{
    font-size: 14px;
    font-weight: 500;
    color: #f0f0f5;
}}

.stat-meta {{
    font-size: 12px;
    color: #555560;
}}

.stat-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    width: fit-content;
}}

.badge-green {{
    background: #0d2b1a;
    color: #4ade80;
    border: 1px solid #1a4a2a;
}}

.badge-amber {{
    background: #2a2010;
    color: #fbbf24;
    border: 1px solid #4a3a10;
}}

.badge-gray {{
    background: #1e1e28;
    color: #9090a0;
    border: 1px solid #2a2a30;
}}

.section-header {{
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    color: #555560;
    margin: 32px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #2a2a30;
}}

.stock-table {{
    background: #16161a;
    border: 1px solid #2a2a30;
    border-radius: 12px;
    padding: 4px 20px;
}}

.stock-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 0;
    border-bottom: 1px solid #1e1e28;
}}

.stock-row:last-child {{
    border-bottom: none;
}}

.stock-symbol {{
    font-size: 13px;
    font-weight: 500;
    color: #f0f0f5;
}}

.stock-price {{
    font-size: 13px;
    font-weight: 500;
    color: #f0f0f5;
    text-align: right;
}}

.stock-change-pos {{
    font-size: 12px;
    color: #4ade80;
    text-align: right;
}}

.stock-change-neg {{
    font-size: 12px;
    color: #f87171;
    text-align: right;
}}

.stock-change-neu {{
    font-size: 12px;
    color: #9090a0;
    text-align: right;
}}

.pipe-card {{
    background: #16161a;
    border: 1px solid #2a2a30;
    border-radius: 12px;
    padding: 18px 20px;
    height: 110px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}
</style>
"""


# ── Auth HTML ────────────────────────────────────────────────────────────────

AUTH_CARD_TOP = """
<div class="page-wrap">
<div class="auth-card">
<div class="logo-row">
<div class="logo-hex">H</div>
<div>
<div class="logo-name">Hedra</div>
<div class="logo-tag">Risk Intelligence Platform</div>
</div>
</div>
<div class="divider"></div>
<div class="section-label">Kite Connect</div>
"""

AUTH_CARD_BOTTOM = """
<div class="feature-row">

<div class="feature-item">
<div class="feature-icon">↗</div>
<div class="feature-text">
<strong>LSTM-GARCH prediction</strong><br>
Three-class risk classification on Nifty 50 data
</div>
</div>

<div class="feature-item">
<div class="feature-icon">▦</div>
<div class="feature-text">
<strong>PDDL planning</strong><br>
Constraint-aware hedge action sequences
</div>
</div>

<div class="feature-item">
<div class="feature-icon">◎</div>
<div class="feature-text">
<strong>SHAP explainability</strong><br>
Feature attribution for every decision
</div>
</div>

</div>
</div>
</div>
"""


# ── Kite helpers ─────────────────────────────────────────────────────────────

def exchange_and_save_token(request_token):
    kite = KiteConnect(api_key=API_KEY)
    session = kite.generate_session(request_token, api_secret=API_SECRET)
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
        profile = kite.profile()
        return True, profile
    except Exception as e:
        return False, str(e)


# ── Auth page ────────────────────────────────────────────────────────────────

def render_auth_page(state="login", token=None, error=None):
    st.markdown(get_css(auth_page=True), unsafe_allow_html=True)

    kite = KiteConnect(api_key=API_KEY)
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
        safe_error = escape(str(error))
        body = (
            "<div class='status-pill error'><span class='dot'></span> Authentication failed</div>"
            "<div class='error-box'>" + safe_error + "</div>"
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


# ── Custom sidebar ───────────────────────────────────────────────────────────

def render_custom_sidebar(profile, token_valid=True):
    name = escape(profile.get("user_name", "—")) if profile else "—"
    email = escape(profile.get("email", "—")) if profile else "—"
    token_text = "Token active" if token_valid else "Token expired"

    if st.button("☰", key="sidebar_toggle"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

    if st.session_state.sidebar_open:
        sidebar_html = f"""
<div class="custom-sidebar">

<div class="sidebar-logo">
<div class="sidebar-logo-box">H</div>
<div>
<div class="sidebar-title">Hedra</div>
<div class="sidebar-subtitle">Risk Intelligence Platform</div>
</div>
</div>

<div class="sidebar-line"></div>

<div class="sidebar-label">Account</div>
<div class="sidebar-name">{name}</div>
<div class="sidebar-email">{email}</div>

<div class="sidebar-pill">
<span class="sidebar-dot"></span>
<span>{token_text}</span>
</div>

<div class="sidebar-line"></div>

<div class="sidebar-label">Pages</div>
<div class="sidebar-page">Overview</div>
<div class="sidebar-page">Portfolio</div>
<div class="sidebar-page sidebar-page-disabled">ML Prediction <span class="sidebar-soon">soon</span></div>
<div class="sidebar-page sidebar-page-disabled">SHAP <span class="sidebar-soon">soon</span></div>
<div class="sidebar-page sidebar-page-disabled">Audit Log <span class="sidebar-soon">soon</span></div>

</div>
"""
    else:
        sidebar_html = """
<div class="custom-sidebar">
<div class="sidebar-logo-box">H</div>
<div class="sidebar-line"></div>
<div class="sidebar-icon-item">⌂</div>
<div class="sidebar-icon-item">▦</div>
<div class="sidebar-icon-item sidebar-icon-disabled">◆</div>
<div class="sidebar-icon-item sidebar-icon-disabled">◎</div>
<div class="sidebar-icon-item sidebar-icon-disabled">◷</div>
</div>
"""

    st.markdown(sidebar_html, unsafe_allow_html=True)


# ── Dashboard home ───────────────────────────────────────────────────────────

def render_dashboard_home(kite, profile):
    st.markdown(get_css(auth_page=False), unsafe_allow_html=True)
    render_custom_sidebar(profile, token_valid=True)

    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST).strftime("%d %b %Y, %H:%M IST")

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    data_files = os.listdir(raw_dir) if os.path.exists(raw_dir) else []
    csv_count = len([f for f in data_files if f.endswith(".csv")])

    token_raw = os.getenv("KITE_ACCESS_TOKEN", "")
    token_preview = (
        token_raw[:8] + "&#8226;&#8226;&#8226;&#8226;" + token_raw[-4:]
        if token_raw else "—"
    )

    user_name = escape(profile.get("user_name", "—"))
    user_email = escape(profile.get("email", "—"))

    data_badge = "badge-green" if csv_count > 0 else "badge-amber"
    data_value = f"{csv_count} files" if csv_count > 0 else "Not fetched"
    data_note = "data/raw/" if csv_count > 0 else "Run fetch_data.py"

    st.markdown(
        f"""
<p class="page-title">Overview</p>
<p class="page-sub">{now}</p>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
<div class="stat-card">
<div class="stat-label">Kite Connection</div>
<div class="stat-value">Active</div>
<div class="stat-badge badge-green">
<span style="width:5px;height:5px;border-radius:50%;background:#4ade80;display:inline-block;"></span>
Connected
</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
<div class="stat-card">
<div class="stat-label">Authenticated As</div>
<div class="stat-value-sm">{user_name}</div>
<div class="stat-meta">{user_email}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
<div class="stat-card">
<div class="stat-label">Access Token</div>
<div style="font-size:13px;font-weight:500;color:#6c63ff;font-family:Courier New,monospace;letter-spacing:0.5px;">
{token_preview}
</div>
<div class="stat-meta">Expires 6:00 AM IST</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
<div class="stat-card">
<div class="stat-label">Raw Data</div>
<div class="stat-value" style="font-size:18px;">{data_value}</div>
<div class="stat-badge {data_badge}">{data_note}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-header'>Portfolio &mdash; Nifty 50 Constituents</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Fetching live quotes ..."):
        try:
            symbols = ["NSE:" + s for s in PORTFOLIO]
            quotes = kite.quote(symbols)
            quote_data = []

            for sym in PORTFOLIO:
                key = "NSE:" + sym

                if key in quotes:
                    q = quotes[key]
                    ltp = q["last_price"]
                    close = q["ohlc"]["close"]
                    pct = ((ltp - close) / close * 100) if close else 0

                    quote_data.append(
                        {
                            "symbol": sym,
                            "ltp": ltp,
                            "pct": pct,
                        }
                    )

            fetch_ok = True
            fetch_error = None

        except Exception as e:
            fetch_ok = False
            fetch_error = str(e)
            quote_data = []

    if not fetch_ok:
        safe_error = escape(fetch_error)
        st.markdown(
            f"""
<div style="background:#1a0808;border:1px solid #4a1a1a;border-radius:8px;padding:14px 16px;font-size:13px;color:#f87171;">
Could not fetch live quotes: {safe_error}
</div>
            """,
            unsafe_allow_html=True,
        )

    else:
        def stock_row(item):
            pct = item["pct"]
            sign = "+" if pct >= 0 else ""

            if pct > 0:
                cls = "stock-change-pos"
            elif pct < 0:
                cls = "stock-change-neg"
            else:
                cls = "stock-change-neu"

            return (
                "<div class='stock-row'>"
                "<div class='stock-symbol'>" + escape(item["symbol"]) + "</div>"
                "<div style='text-align:right;'>"
                "<div class='stock-price'>&#8377;" + "{:,.2f}".format(item["ltp"]) + "</div>"
                "<div class='" + cls + "'>" + sign + "{:.2f}%".format(pct) + "</div>"
                "</div>"
                "</div>"
            )

        col_l, col_r = st.columns(2)

        with col_l:
            rows = "".join(stock_row(s) for s in quote_data[:5])
            st.markdown("<div class='stock-table'>" + rows + "</div>", unsafe_allow_html=True)

        with col_r:
            rows = "".join(stock_row(s) for s in quote_data[5:])
            st.markdown("<div class='stock-table'>" + rows + "</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-header'>Pipeline Status</div>",
        unsafe_allow_html=True,
    )

    stages = [
        ("Data Pipeline", csv_count > 0, "data/raw/" if csv_count > 0 else "Run fetch_data.py"),
        ("Feature Engineering", False, "Week 2"),
        ("LSTM-GARCH Model", False, "Week 4"),
        ("PDDL Planning", False, "Week 6"),
        ("SHAP Explainability", False, "Week 8"),
    ]

    cols = st.columns(5)

    for col, (label, done, note) in zip(cols, stages):
        badge = "badge-green" if done else "badge-gray"
        text = "Ready" if done else "Pending"

        with col:
            st.markdown(
                f"""
<div class="pipe-card">
<div class="stat-label">{label}</div>
<div class="stat-badge {badge}">{text}</div>
<div class="stat-meta">{note}</div>
</div>
                """,
                unsafe_allow_html=True,
            )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY or not API_SECRET:
        st.error("KITE_API_KEY or KITE_API_SECRET not found in .env.")
        st.stop()

    params = st.query_params
    request_token = params.get("request_token")
    kite_error = params.get("error")

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

    kite = get_kite_client()
    valid, result = verify_token(kite)

    if not valid:
        render_auth_page(
            state="error",
            error="Session expired. Please re-authenticate.",
        )
        return

    render_dashboard_home(kite, result)


if __name__ == "__main__":
    main()