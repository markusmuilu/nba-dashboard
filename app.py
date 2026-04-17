"""
NBA Prediction Analytics Dashboard
====================================
Multi-tab Streamlit application visualising NBA game predictions from R2.

Tabs:
    1. Overview          — KPIs, rolling accuracy, today's games, history
    2. Model Performance — metrics, calibration, confusion matrix, baselines
    3. Teams             — per-team stats, form guide, H2H breakdown
    4. Upset Analysis    — favourite/underdog accuracy, implied prob scatter
    5. Odds & Betting    — P&L simulation, drawdown, monthly summary

Data: Cloudflare R2 bucket "nbaprediction"
    history/prediction_history.json   — resolved predictions
    current/current_predictions.json  — today's unresolved predictions

Credentials (priority order): OS env → Streamlit secrets
    R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
"""

import os
import json

import boto3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from sklearn.metrics import f1_score, precision_score, recall_score
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="NBA Prediction Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Backgrounds ── */
.stApp { background-color: #020617; color: #e5e7eb; }
section[data-testid="stSidebar"] { background-color: #0a0f1e; border-right: 1px solid rgba(148,163,184,0.12); }

/* ── Page header gradient ── */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(59,130,246,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.app-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e5e7eb 0%, #93c5fd 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 4px 0;
}
.app-header p { color: #94a3b8; font-size: 0.88rem; margin: 0; }

/* ── Section heading ── */
.section-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
}
.section-head-bar {
    width: 4px;
    height: 22px;
    background: linear-gradient(180deg, #3b82f6, #60a5fa);
    border-radius: 2px;
    flex-shrink: 0;
}
.section-head h3 {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e5e7eb;
    margin: 0;
}
.section-head span {
    font-size: 0.78rem;
    color: #64748b;
    margin-left: 4px;
}

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 12px; flex-wrap: wrap; }
.kpi-card {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(145deg, #0f172a, #111827);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 14px;
    padding: 18px 20px 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    opacity: 0;
    transition: opacity 0.2s;
}
.kpi-card:hover { border-color: rgba(96,165,250,0.45); transform: translateY(-2px); }
.kpi-card:hover::after { opacity: 1; }
.kpi-label {
    color: #64748b;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 6px;
}
.kpi-value {
    color: #f1f5f9;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.kpi-value.blue   { color: #60a5fa; }
.kpi-value.green  { color: #34d399; }
.kpi-value.red    { color: #f87171; }
.kpi-value.yellow { color: #fbbf24; }
.kpi-delta {
    font-size: 0.72rem;
    font-weight: 500;
    margin-top: 4px;
}
.kpi-delta.pos { color: #34d399; }
.kpi-delta.neg { color: #f87171; }
.kpi-delta.neu { color: #64748b; }

/* ── Game card (today's predictions) ── */
.game-card {
    background: linear-gradient(145deg, #0f172a, #111827);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.2s;
}
.game-card:hover { border-color: rgba(96,165,250,0.4); }
.game-matchup { font-size: 1rem; font-weight: 600; color: #e5e7eb; }
.game-matchup .vs { color: #475569; font-size: 0.8rem; font-weight: 400; margin: 0 8px; }
.game-pick { font-size: 0.78rem; color: #64748b; }
.game-pick strong { color: #60a5fa; }
.game-conf {
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #93c5fd;
}
.game-odds { font-size: 0.78rem; color: #64748b; text-align: right; }

/* ── Version badge ── */
.ver-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: .04em;
}
.ver-v1   { background: rgba(148,163,184,0.12); color: #94a3b8; }
.ver-v21  { background: rgba(167,139,250,0.15); color: #a78bfa; }
.ver-nn   { background: rgba(251,191,36,0.15);  color: #fbbf24; }
.ver-v22  { background: rgba(52,211,153,0.15);  color: #34d399; }

/* ── Form dots (team form guide) ── */
.form-guide { display: flex; gap: 5px; align-items: center; }
.form-dot {
    width: 20px; height: 20px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.62rem;
    font-weight: 700;
}
.form-dot.win  { background: rgba(52,211,153,0.2);  color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.form-dot.loss { background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }

/* ── Insight banner ── */
.insight-box {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(96,165,250,0.04));
    border: 1px solid rgba(59,130,246,0.2);
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.83rem;
    color: #93c5fd;
}
.insight-box b { color: #e5e7eb; }

/* ── Divider ── */
.styled-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.15), transparent);
    margin: 24px 0;
}

/* ── Tab bar ── */
button[data-baseweb="tab"] { color: #64748b !important; font-size: 0.88rem !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom-color: #3b82f6 !important;
}
div[data-baseweb="tab-list"] { border-bottom-color: rgba(148,163,184,0.12) !important; }

/* ── Sidebar widgets ── */
.sidebar-label { font-size: 0.7rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 4px; }
.filter-count {
    display: inline-block;
    background: rgba(59,130,246,0.2);
    color: #60a5fa;
    border-radius: 12px;
    padding: 1px 7px;
    font-size: 0.68rem;
    font-weight: 600;
    margin-left: 6px;
}

/* ── Dataframe ── */
.stDataFrame thead th { background-color: #0a0f1e !important; color: #3b82f6 !important; font-weight: 600 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: .04em; }
.stDataFrame tbody tr:hover td { background-color: rgba(59,130,246,0.06) !important; }

/* ── Info/empty states ── */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: #475569;
    font-size: 0.9rem;
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT: dict = dict(
    paper_bgcolor="#020617",
    plot_bgcolor="#020617",
    font_color="#e5e7eb",
    font_family="Inter, sans-serif",
    colorway=["#3b82f6", "#22d3ee", "#a78bfa", "#f472b6", "#34d399", "#fb923c"],
    xaxis=dict(gridcolor="rgba(148,163,184,0.07)", zerolinecolor="rgba(148,163,184,0.2)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(148,163,184,0.07)", zerolinecolor="rgba(148,163,184,0.2)", tickfont=dict(size=11)),
    margin=dict(t=48, b=36, l=48, r=24),
    hoverlabel=dict(bgcolor="#0f172a", bordercolor="rgba(148,163,184,0.3)", font_size=12, font_family="Inter"),
)

VERSION_COLOURS = {
    "Logistic reg V1":   "#94a3b8",
    "Logistic reg V2.1": "#a78bfa",
    "Custom NN V1":      "#fbbf24",
    "Logistic reg V2.2": "#34d399",
}

PLAYIN_STARTS: dict[int, pd.Timestamp] = {
    2025: pd.Timestamp("2025-04-15"),
    2026: pd.Timestamp("2026-04-14"),
}
PLAYOFF_STARTS: dict[int, pd.Timestamp] = {
    2025: pd.Timestamp("2025-04-19"),
    2026: pd.Timestamp("2026-04-19"),
}

def _get_season_type(d: pd.Timestamp) -> str:
    # NBA seasons end in June; Oct-Dec belong to the season ending *next* year
    season_year = d.year + 1 if d.month >= 7 else d.year
    playoff = PLAYOFF_STARTS.get(season_year)
    if playoff and d >= playoff:
        return "Playoffs"
    playin = PLAYIN_STARTS.get(season_year)
    if playin and d >= playin:
        return "Play-In"
    return "Regular Season"
VERSION_BADGE_CLASS = {
    "Logistic reg V1":   "ver-v1",
    "Logistic reg V2.1": "ver-v21",
    "Custom NN V1":      "ver-nn",
    "Logistic reg V2.2": "ver-v22",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def model_version(date_str: str) -> str:
    """Map a prediction date string to its model version label."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return "Unknown"
    if d <= date(2025, 12, 5):   return "Logistic reg V1"
    elif d <= date(2025, 12, 14): return "Logistic reg V2.1"
    elif d <= date(2026, 1, 8):   return "Custom NN V1"
    else:                          return "Logistic reg V2.2"


def kpi(label: str, value: str, color: str = "", delta: str = "", delta_dir: str = "neu", tooltip: str = "") -> None:
    """Render a styled KPI card. color: 'blue'|'green'|'red'|'yellow'|''."""
    tip_html = (f' <span title="{tooltip}" style="cursor:help;color:#475569;font-size:0.65rem;'
                f'vertical-align:middle">ⓘ</span>') if tooltip else ""
    delta_html = f'<div class="kpi-delta {delta_dir}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi-card">'
        f'  <div class="kpi-label">{label}{tip_html}</div>'
        f'  <div class="kpi-value {color}">{value}</div>'
        f'  {delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    """Render a section heading with a left blue accent bar."""
    sub = f'<span>{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div class="section-head">'
        f'  <div class="section-head-bar"></div>'
        f'  <h3>{title}{sub}</h3>'
        f'</div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)


def insight(html: str) -> None:
    """Render a blue-tinted insight callout box."""
    st.markdown(f'<div class="insight-box">{html}</div>', unsafe_allow_html=True)


def ver_badge(version: str) -> str:
    cls = VERSION_BADGE_CLASS.get(version, "ver-v1")
    return f'<span class="ver-badge {cls}">{version}</span>'


def accuracy_color(acc: float) -> str:
    if acc >= 62: return "green"
    if acc >= 55: return "blue"
    if acc >= 50: return "yellow"
    return "red"


def profit(stake: float, odds_val: float, correct: bool) -> float:
    return stake * odds_val - stake if correct else -stake


# Vertical annotation events shown on time-series charts.
# Each tuple: (ISO date, label, hex colour)
ALL_EVENTS: list[tuple[str, str, str]] = [
    # ── Model version changes ──────────────────────────────────────────────
    ("2025-12-06", "→ V2.1",   "#a78bfa"),
    ("2025-12-15", "→ NN V1",  "#fbbf24"),
    ("2026-01-09", "→ V2.2",   "#34d399"),
    # ── NBA calendar ──────────────────────────────────────────────────────
    ("2026-02-06", "Trade deadline",   "#f472b6"),
    ("2026-02-13", "All-Star break",   "#94a3b8"),
    ("2026-04-13", "Reg. season ends", "#64748b"),
    ("2026-04-14", "Playoffs start",   "#22d3ee"),
]


def _annotate_chart(fig: go.Figure, ds_min, ds_max) -> None:
    """Add dotted vertical event lines + rotated labels to a plain go.Figure."""
    for ev_date_str, ev_label, ev_colour in ALL_EVENTS:
        ev_ts = pd.Timestamp(ev_date_str).date()
        if ds_min <= ev_ts <= ds_max:
            fig.add_shape(type="line", x0=ev_ts, x1=ev_ts, y0=0, y1=1,
                          xref="x", yref="paper",
                          line=dict(color=ev_colour, dash="dot", width=1))
            fig.add_annotation(x=ev_ts, y=0.97, xref="x", yref="paper",
                               text=ev_label, showarrow=False,
                               font=dict(size=9, color=ev_colour),
                               textangle=-90, xanchor="left", yanchor="top",
                               bgcolor="rgba(2,6,23,0.7)")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _r2_client():
    endpoint  = os.environ.get("R2_ENDPOINT")   or st.secrets.get("R2_ENDPOINT", "")
    access_key = os.environ.get("R2_ACCESS_KEY_ID") or st.secrets.get("R2_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY") or st.secrets.get("R2_SECRET_ACCESS_KEY", "")
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Common normalisation applied to both history and current DataFrames.

    Returns an empty DataFrame with expected columns if the raw data is empty
    (e.g. no games today) or only contains NO_GAMES_TODAY sentinel rows.
    """
    EXPECTED_COLS = ["team", "opponent", "date", "prediction", "confidence",
                     "gameId", "model_version", "home_odds", "away_odds",
                     "predicted_winner", "home_prob", "away_prob", "season_type"]

    # Guard: JSON returned [] or DataFrame has no columns
    if df.empty or "team" not in df.columns:
        return pd.DataFrame(columns=EXPECTED_COLS)

    df = df[df["team"] != "NO_GAMES_TODAY"].copy()

    # Guard: all rows were sentinel rows
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLS)

    df["date"]          = pd.to_datetime(df["date"])
    df["model_version"] = df["date"].dt.strftime("%Y-%m-%d").apply(model_version)
    df["season_type"]   = df["date"].apply(_get_season_type)
    df["home_odds"]     = pd.to_numeric(df.get("home_odds", np.nan), errors="coerce")
    df["away_odds"]     = pd.to_numeric(df.get("away_odds", np.nan), errors="coerce")
    df["predicted_winner"] = df.apply(lambda r: r["team"] if r["prediction"] else r["opponent"], axis=1)
    # Implied probability from odds (1/odds, normalised to remove bookmaker margin)
    mask = df["home_odds"].notna() & df["away_odds"].notna()
    if mask.any():
        total = 1/df.loc[mask, "home_odds"] + 1/df.loc[mask, "away_odds"]
        df.loc[mask, "home_prob"] = (1 / df.loc[mask, "home_odds"]) / total * 100
        df.loc[mask, "away_prob"] = 100 - df.loc[mask, "home_prob"]
    return df


@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
    bucket = os.environ.get("R2_BUCKET_NAME") or st.secrets.get("R2_BUCKET_NAME", "nbaprediction")
    obj = _r2_client().get_object(Bucket=bucket, Key="history/prediction_history.json")
    df  = pd.DataFrame(json.loads(obj["Body"].read()))
    return _normalise(df)


@st.cache_data(ttl=300)
def load_current() -> pd.DataFrame:
    bucket = os.environ.get("R2_BUCKET_NAME") or st.secrets.get("R2_BUCKET_NAME", "nbaprediction")
    obj = _r2_client().get_object(Bucket=bucket, Key="current/current_predictions.json")
    df  = pd.DataFrame(json.loads(obj["Body"].read()))
    return _normalise(df)


try:
    hist_raw = load_history()
    curr_raw = load_current()
except Exception as exc:
    st.error(f"Failed to load data from R2: {exc}")
    st.info("Set R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME.")
    st.stop()

all_versions: list[str] = sorted(hist_raw["model_version"].unique().tolist())
all_teams: list[str]    = sorted(set(hist_raw["team"]) | set(hist_raw["opponent"]))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 4px 8px">
        <div style="font-size:1.3rem;font-weight:700;color:#e5e7eb;letter-spacing:-0.02em">
            🏀 NBA Predictions
        </div>
        <div style="font-size:0.72rem;color:#475569;margin-top:2px">Analytics Dashboard</div>
    </div>
    <hr style="border:none;height:1px;background:rgba(148,163,184,0.1);margin:8px 0 16px">
    """, unsafe_allow_html=True)

    sel_versions: list[str] = st.multiselect(
        "Model version",
        options=all_versions,
        default=["Logistic reg V2.2"] if "Logistic reg V2.2" in all_versions else all_versions,
    )
    conf_range: tuple[int, int] = st.slider("Confidence range (%)", 50, 100, (50, 100))

    team_options = ["All teams"] + all_teams
    sel_team: str = st.selectbox("Team", team_options, index=0,
                                  help="Filter all tabs to games involving this team.")

    min_date: date = hist_raw["date"].min().date()
    max_date: date = hist_raw["date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date),
                                min_value=min_date, max_value=max_date)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d_start, d_end = date_range
    else:
        d_start, d_end = min_date, max_date

    all_season_types = ["Regular Season", "Play-In", "Playoffs"]
    sel_season_types: list[str] = st.multiselect(
        "Season type",
        options=all_season_types,
        default=["Regular Season"],
        help="Regular Season = Oct–mid Apr · Play-In = ~Apr 14-18 · Playoffs = Apr 19+",
    )

    st.markdown('<hr style="border:none;height:1px;background:rgba(148,163,184,0.1);margin:16px 0">', unsafe_allow_html=True)

    # Active filter summary
    active = []
    if sel_versions != all_versions: active.append(f"{len(sel_versions)} version(s)")
    if conf_range != (50, 100): active.append(f"conf {conf_range[0]}-{conf_range[1]}%")
    if sel_team != "All teams": active.append(sel_team)
    if (d_start, d_end) != (min_date, max_date): active.append("date")
    if sel_season_types != all_season_types: active.append(" · ".join(sel_season_types))
    if active:
        st.markdown(f'<div style="font-size:0.72rem;color:#475569">Active filters: <span style="color:#60a5fa">{" · ".join(active)}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.72rem;color:#475569">No active filters — showing all data</div>', unsafe_allow_html=True)

    st.caption("Refreshes every 5 min · R2 backed")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["model_version"].isin(sel_versions)
        & df["confidence"].between(conf_range[0], conf_range[1])
        & (df["date"].dt.date >= d_start)
        & (df["date"].dt.date <= d_end)
        & df["season_type"].isin(sel_season_types)
    )
    if sel_team != "All teams":
        mask &= (df["team"] == sel_team) | (df["opponent"] == sel_team)
    return df[mask].copy()


hist: pd.DataFrame = apply_filters(hist_raw)

# ── Page header ──────────────────────────────────────────────────────────────
latest_ver = hist["model_version"].iloc[-1] if len(hist) else "—"
n_today    = len(curr_raw)
st.markdown(f"""
<div class="app-header">
  <h1>NBA Prediction Analytics</h1>
  <p>Active model: {ver_badge(latest_ver)} &nbsp;·&nbsp;
     {len(hist):,} predictions tracked &nbsp;·&nbsp;
     {n_today} game{"s" if n_today != 1 else ""} today</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🧠 Model Performance", "🏀 Teams", "⚡ Upset Analysis", "💰 Odds & Betting"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    total:     int   = len(hist)
    correct:   int   = int(hist["prediction_correct"].sum()) if total else 0
    accuracy:  float = correct / total * 100 if total else 0.0
    home_win_pct: float = hist["winner"].mean() * 100 if total else 0.0
    earliest = hist["date"].min().strftime("%Y-%m-%d") if total else "—"
    latest   = hist["date"].max().strftime("%Y-%m-%d") if total else "—"

    # 30-day accuracy for delta context
    last30 = hist[hist["date"] >= (hist["date"].max() - pd.Timedelta(days=30))]
    acc30  = last30["prediction_correct"].mean() * 100 if len(last30) else accuracy
    delta_acc = acc30 - accuracy
    delta_str = f"{'▲' if delta_acc >= 0 else '▼'} {abs(delta_acc):.1f}% vs overall (30-day)"
    delta_dir = "pos" if delta_acc >= 0 else "neg"

    # Current streak
    streak_val, streak_type = 0, ""
    if total:
        recent = hist.sort_values("date")["prediction_correct"].tolist()
        streak_val = 1
        streak_type = "W" if recent[-1] else "L"
        for r in reversed(recent[:-1]):
            if bool(r) == (streak_type == "W"):
                streak_val += 1
            else:
                break

    section_header("Key Metrics", f"· {total:,} predictions · filtered view")
    c = st.columns(6)
    with c[0]: kpi("Running since", earliest)
    with c[1]: kpi("Latest prediction", latest)
    with c[2]: kpi("Overall accuracy", f"{accuracy:.1f}%",
                   color=accuracy_color(accuracy),
                   delta=delta_str, delta_dir=delta_dir)
    with c[3]: kpi("Home win %", f"{home_win_pct:.1f}%", color="blue")
    with c[4]: kpi("Total predictions", f"{total:,}")
    with c[5]: kpi("Current streak",
                   f"{streak_val}{streak_type}",
                   color="green" if streak_type == "W" else "red")

    # Auto-insight
    if total:
        gap = accuracy - home_win_pct
        if gap > 3:
            insight(f"<b>Model is beating the always-home baseline</b> by <b>{gap:.1f}pp</b>. "
                    f"It correctly identifies away wins above the naive rate.")
        elif gap < -3:
            insight(f"<b>Model is underperforming vs always-home</b> by <b>{abs(gap):.1f}pp</b>. "
                    f"Consider whether the current filter set is representative.")

    divider()

    # ── Daily predictions + 7-day rolling accuracy ────────────────────────────
    if total:
        section_header("Volume & Rolling Accuracy", "· 7-day window")
        daily = (
            hist.groupby(hist["date"].dt.date)
            .agg(total_preds=("prediction_correct","count"), correct_preds=("prediction_correct","sum"))
            .reset_index().rename(columns={"date":"ds"}).sort_values("ds")
        )
        daily["rolling_acc"] = (
            daily["correct_preds"].rolling(7, min_periods=1).sum()
            / daily["total_preds"].rolling(7, min_periods=1).sum() * 100
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=daily["ds"], y=daily["total_preds"], name="Predictions",
                             marker_color="#3b82f6", opacity=0.55), secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["ds"], y=daily["rolling_acc"], name="7-day accuracy %",
                                  line=dict(color="#22d3ee", width=2.5), mode="lines"), secondary_y=True)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(148,163,184,0.25)", secondary_y=True)
        _annotate_chart(fig, daily["ds"].min(), daily["ds"].max())
        fig.update_layout(**PLOTLY_LAYOUT, legend=dict(bgcolor="#0f172a", x=0, y=1.1, orientation="h"))
        fig.update_yaxes(title_text="Predictions", secondary_y=False, gridcolor="rgba(148,163,184,0.07)")
        fig.update_yaxes(title_text="Accuracy %",  secondary_y=True,  range=[0,100], gridcolor="rgba(148,163,184,0.07)")
        st.plotly_chart(fig, width='stretch')

    divider()

    # ── Today's games ─────────────────────────────────────────────────────────
    section_header("Today's Predictions", f"· {n_today} game{'s' if n_today != 1 else ''} · unresolved")
    if curr_raw.empty:
        st.markdown('<div class="empty-state"><div class="icon">🏖️</div>No games scheduled today.</div>', unsafe_allow_html=True)
    else:
        for _, r in curr_raw.sort_values("confidence", ascending=False).iterrows():
            ho = f"{r['home_odds']:.2f}" if pd.notna(r.get("home_odds")) else "—"
            ao = f"{r['away_odds']:.2f}" if pd.notna(r.get("away_odds")) else "—"
            hp = f"{r['home_prob']:.0f}%" if pd.notna(r.get("home_prob")) else ""
            ap = f"{r['away_prob']:.0f}%" if pd.notna(r.get("away_prob")) else ""
            winner_label = r["predicted_winner"]
            st.markdown(f"""
            <div class="game-card">
              <div>
                <div class="game-matchup">
                  {r['team']} <span class="vs">vs</span> {r['opponent']}
                </div>
                <div class="game-pick" style="margin-top:4px">
                  Pick: <strong>{winner_label}</strong>
                </div>
              </div>
              <div style="text-align:center">
                <div class="game-conf">{r['confidence']:.1f}%</div>
                <div style="font-size:0.68rem;color:#475569;margin-top:4px">confidence</div>
              </div>
              <div class="game-odds">
                <div style="color:#e5e7eb;font-size:0.78rem">{r['team']} {ho} {f'({hp})' if hp else ''}</div>
                <div style="color:#e5e7eb;font-size:0.78rem">{r['opponent']} {ao} {f'({ap})' if ap else ''}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # ── Prediction history (paginated) ────────────────────────────────────────
    section_header("Prediction History", f"· {total:,} records")
    hist_disp = hist[["date","team","opponent","predicted_winner","confidence","prediction_correct","model_version"]].copy()
    hist_disp["date"] = hist_disp["date"].dt.strftime("%Y-%m-%d")
    hist_disp["prediction_correct"] = hist_disp["prediction_correct"].map({True: "✅", False: "❌"})
    hist_disp.columns = ["Date","Home","Away","Predicted Winner","Confidence %","Correct","Model"]
    hist_disp = hist_disp.sort_values("Date", ascending=False)

    page_size = 25
    total_pages = max(1, (len(hist_disp) - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, label_visibility="collapsed")
    st.dataframe(hist_disp.iloc[(page-1)*page_size:(page)*page_size], width='stretch', hide_index=True)
    st.caption(f"Page {page} of {total_pages}  ·  {len(hist_disp):,} records total")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if hist.empty:
        st.markdown('<div class="empty-state"><div class="icon">🔍</div>No data after applying filters.</div>', unsafe_allow_html=True)
    else:
        # ── Per-version metrics ───────────────────────────────────────────────
        rows = []
        for ver, grp in hist.groupby("model_version"):
            y_true = grp["winner"].astype(int).tolist()
            y_pred = grp["prediction"].astype(int).tolist()
            acc  = grp["prediction_correct"].mean() * 100
            f1   = f1_score(y_true, y_pred, zero_division=0) * 100
            prec = precision_score(y_true, y_pred, zero_division=0) * 100
            rec  = recall_score(y_true, y_pred, zero_division=0) * 100
            rows.append(dict(Version=ver, Games=len(grp),
                             Accuracy=round(acc,1), F1=round(f1,1),
                             Precision=round(prec,1), Recall=round(rec,1)))
        metrics_df = pd.DataFrame(rows)

        section_header("Metrics by Version")
        st.dataframe(metrics_df, width='stretch', hide_index=True)

        # Version performance comparison bar chart with version-specific colours
        fig_m = go.Figure()
        for _, row in metrics_df.iterrows():
            color = VERSION_COLOURS.get(row["Version"], "#3b82f6")
            for metric, val in [("Accuracy", row["Accuracy"]), ("F1", row["F1"]),
                                  ("Precision", row["Precision"]), ("Recall", row["Recall"])]:
                fig_m.add_trace(go.Bar(name=f'{row["Version"]} — {metric}', x=[f'{row["Version"]}<br>{metric}'],
                                        y=[val], marker_color=color, opacity=0.8,
                                        hovertemplate=f'{row["Version"]}<br>{metric}: %{{y:.1f}}%<extra></extra>',
                                        showlegend=False))
        melted = metrics_df.melt(id_vars="Version", value_vars=["Accuracy","F1","Precision","Recall"])
        fig_m = px.bar(melted, x="variable", y="value", color="Version", barmode="group",
                       color_discrete_map=VERSION_COLOURS,
                       labels={"value": "%", "variable": "Metric", "Version": "Model"})
        fig_m.update_layout(**PLOTLY_LAYOUT, legend=dict(bgcolor="#0f172a", orientation="h", x=0, y=1.08),
                             title="Classification Metrics by Model Version")
        fig_m.update_traces(hovertemplate="%{x} — %{fullData.name}: %{y:.1f}%<extra></extra>")
        st.plotly_chart(fig_m, width='stretch')

        # Best version callout
        best_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
        insight(f"Best accuracy: {ver_badge(best_row['Version'])} at <b>{best_row['Accuracy']:.1f}%</b> "
                f"over {best_row['Games']} games · F1 {best_row['F1']:.1f}%")

        divider()

        # ── Calibration + Confusion Matrix ────────────────────────────────────
        col_cal, col_cm = st.columns(2)

        with col_cal:
            section_header("Calibration Chart")
            st.caption(
                "Bars show actual accuracy per confidence bucket. "
                "The dashed line is the mean model confidence within each bucket. "
                "Green = accuracy exceeds the bucket's lower bound (e.g. >61 % in the 61-70 bucket). "
                "Red = accuracy falls below the bucket's lower bound."
            )
            bins   = [49, 60, 70, 80, 90, 101]
            labels = ["50-60","61-70","71-80","81-90","90+"]
            # Lower bound of each bucket — green if actual accuracy exceeds it
            _lower_map = {"50-60": 50, "61-70": 61, "71-80": 71, "81-90": 81, "90+": 90}
            # Round-half-up: (x + 0.5).astype(int) so 60.5 → 61, 60.49 → 60
            hist["conf_bucket"] = pd.cut(
                (hist["confidence"] + 0.5).astype(int),
                bins=bins, labels=labels, right=True,
            )
            cal = (
                hist.groupby("conf_bucket", observed=True)
                .agg(acc=("prediction_correct","mean"), n=("prediction_correct","count"),
                     mean_conf=("confidence","mean"))
                .reset_index()
            )
            cal["acc_pct"]   = cal["acc"] * 100
            cal["lower"]     = cal["conf_bucket"].astype(str).map(_lower_map)
            cal["colour"]    = cal.apply(
                lambda r: "#34d399" if r["acc_pct"] >= r["lower"] else "#f87171", axis=1
            )

            fig_cal = go.Figure()
            fig_cal.add_trace(go.Bar(
                x=cal["conf_bucket"].astype(str), y=cal["acc_pct"],
                marker_color=cal["colour"].tolist(),
                text=[f"{a:.1f}%<br>n={n}" for a, n in zip(cal["acc_pct"], cal["n"])],
                texttemplate="%{text}", textposition="outside",
                hovertemplate="%{x}: accuracy %{y:.1f}%<extra></extra>",
                name="Actual accuracy",
            ))
            fig_cal.add_trace(go.Scatter(
                x=cal["conf_bucket"].astype(str), y=cal["mean_conf"],
                mode="lines+markers", name="Mean confidence",
                line=dict(color="rgba(248,113,113,0.7)", dash="dash", width=1.5),
                marker=dict(color="rgba(248,113,113,0.7)", size=7),
                hovertemplate="%{x} mean confidence: %{y:.1f}%<extra></extra>",
            ))
            fig_cal.update_layout(**PLOTLY_LAYOUT, yaxis_range=[0, 115],
                                   xaxis_title="Confidence bucket",
                                   yaxis_title="Actual accuracy (%)",
                                   legend=dict(bgcolor="#0f172a"))
            st.plotly_chart(fig_cal, width='stretch')

        with col_cm:
            section_header("Confusion Matrix")
            if metrics_df.empty:
                st.info("No data matches the current filters.")
            else:
                ver_sel = st.selectbox("Version", metrics_df["Version"].tolist(), key="cm_ver")
                grp_cm  = hist[hist["model_version"] == ver_sel]
                tp = int(((grp_cm["prediction"]==True)  & (grp_cm["winner"]==True)).sum())
                fp = int(((grp_cm["prediction"]==True)  & (grp_cm["winner"]==False)).sum())
                fn = int(((grp_cm["prediction"]==False) & (grp_cm["winner"]==True)).sum())
                tn = int(((grp_cm["prediction"]==False) & (grp_cm["winner"]==False)).sum())
                total_cm = tp+fp+fn+tn
                cm_text = [[f"TP\n{tp}\n{tp/total_cm*100:.0f}%", f"FP\n{fp}\n{fp/total_cm*100:.0f}%"],
                           [f"FN\n{fn}\n{fn/total_cm*100:.0f}%", f"TN\n{tn}\n{tn/total_cm*100:.0f}%"]]
                fig_cm = go.Figure(go.Heatmap(
                    z=[[tp,fp],[fn,tn]], text=cm_text, texttemplate="%{text}",
                    x=["Pred Home Win","Pred Away Win"], y=["Actual Home Win","Actual Away Win"],
                    colorscale=[[0,"#0a0f1e"],[0.5,"#1e3a5f"],[1,"#3b82f6"]], showscale=False,
                    hovertemplate="%{text}<extra></extra>",
                ))
                fig_cm.update_layout(**PLOTLY_LAYOUT)
                fig_cm.update_layout(margin=dict(t=20,b=36,l=48,r=24))
                ppv = tp/(tp+fp)*100 if (tp+fp) else 0
                npv = tn/(tn+fn)*100 if (tn+fn) else 0
                st.plotly_chart(fig_cm, width='stretch')
                st.caption(f"PPV (precision): {ppv:.1f}%  ·  NPV: {npv:.1f}%  ·  {total_cm} games")

        divider()

        # ── Model vs baselines ────────────────────────────────────────────────
        divider()
        section_header("Model vs Baselines", "· how does the model compare to simple strategies?")
        overall_acc   = hist["prediction_correct"].mean() * 100
        home_baseline = hist["winner"].mean() * 100
        away_baseline = (1 - hist["winner"].mean()) * 100

        base_labels  = ["Model", "Always home", "Always away"]
        base_values  = [overall_acc, home_baseline, away_baseline]
        base_colours = ["#3b82f6", "#94a3b8", "#a78bfa"]

        fig_base = go.Figure(go.Bar(
            x=base_values, y=base_labels, orientation="h",
            marker_color=base_colours,
            text=[f"{v:.1f}%" for v in base_values],
            texttemplate="%{text}", textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig_base.add_vline(x=50, line_dash="dot", line_color="rgba(148,163,184,0.25)")
        fig_base.update_layout(**PLOTLY_LAYOUT, xaxis_range=[0, 110],
                                xaxis_title="Accuracy %", yaxis_title="",
                                showlegend=False, height=260)
        st.plotly_chart(fig_base, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TEAMS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if hist.empty:
        st.markdown('<div class="empty-state"><div class="icon">🔍</div>No data after applying filters.</div>', unsafe_allow_html=True)
    else:
        all_involved_teams = sorted(set(hist["team"]) | set(hist["opponent"]))

        # Build team stats table
        team_rows = []
        for team in all_involved_teams:
            home = hist[hist["team"] == team]
            away = hist[hist["opponent"] == team]
            combined = pd.concat([home, away])
            if combined.empty: continue
            games    = len(combined)
            acc      = combined["prediction_correct"].mean() * 100
            correct_n = int(combined["prediction_correct"].sum())
            pred_wins   = int((home["prediction"]==True).sum() + (away["prediction"]==False).sum())
            actual_wins = int((home["winner"]==True).sum() + (away["winner"]==False).sum())
            team_odds = combined.dropna(subset=["home_odds","away_odds"])
            if len(team_odds):
                pnl = sum(profit(1, r["home_odds"] if r["prediction"] else r["away_odds"],
                                 bool(r["prediction_correct"])) for _, r in team_odds.iterrows())
                roi = pnl / len(team_odds) * 100
            else:
                pnl, roi = None, None
            team_rows.append(dict(Team=team, Games=games, Accuracy=round(acc,1), Correct=correct_n,
                                   Predicted=f"{pred_wins}-{games-pred_wins}",
                                   Actual=f"{actual_wins}-{games-actual_wins}",
                                   PnL=round(pnl,2) if pnl is not None else None,
                                   ROI=round(roi,1) if roi is not None else None))

        team_df = pd.DataFrame(team_rows).sort_values("Games", ascending=False)

        # ── Accuracy chart (horizontal bar) ───────────────────────────────────
        section_header("Team Accuracy Ranking", f"· {len(team_df)} teams")
        col_chart, col_table = st.columns([1, 1])

        with col_chart:
            top20 = team_df.nlargest(20, "Games").sort_values("Accuracy")
            colors = ["#34d399" if a >= 60 else "#3b82f6" if a >= 55 else "#f87171" for a in top20["Accuracy"]]
            fig_ta = go.Figure(go.Bar(
                x=top20["Accuracy"], y=top20["Team"], orientation="h",
                marker_color=colors,
                text=[f"{a:.1f}%" for a in top20["Accuracy"]], textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig_ta.add_vline(x=50, line_dash="dot", line_color="rgba(248,113,113,0.4)")
            fig_ta.update_layout(**PLOTLY_LAYOUT, xaxis_range=[30,100],
                                  xaxis_title="Accuracy %", yaxis_title="", height=520)
            fig_ta.update_layout(margin=dict(t=12,b=36,l=60,r=60))
            st.plotly_chart(fig_ta, width='stretch')

        with col_table:
            st.dataframe(
                team_df.rename(columns={"Predicted":"Pred W-L","Actual":"Actual W-L","PnL":"P&L (€)","ROI":"ROI %"}),
                width='stretch', hide_index=True, height=520,
            )

        divider()

        # ── Team deep-dive ────────────────────────────────────────────────────
        section_header("Team Deep Dive")
        team_pick = st.selectbox("Select team", all_involved_teams, key="team_pick")
        team_hist = hist[(hist["team"]==team_pick) | (hist["opponent"]==team_pick)].copy()
        team_hist = team_hist.sort_values("date")

        # KPIs for selected team
        t_games = len(team_hist)
        t_acc   = team_hist["prediction_correct"].mean() * 100 if t_games else 0
        t_row   = team_df[team_df["Team"]==team_pick].iloc[0] if team_pick in team_df["Team"].values else None

        kpi_cols = st.columns(4)
        with kpi_cols[0]: kpi("Games tracked", str(t_games))
        with kpi_cols[1]: kpi("Model accuracy", f"{t_acc:.1f}%", color=accuracy_color(t_acc))
        with kpi_cols[2]: kpi("Actual W-L", t_row["Actual"] if t_row is not None else "—")
        with kpi_cols[3]: kpi("Predicted W-L", t_row["Predicted"] if t_row is not None else "—")

        # Form guide — last 10 predictions involving this team
        st.markdown("<div style='margin-top:16px'>", unsafe_allow_html=True)
        last10 = team_hist.tail(10)
        form_dots = "".join([
            f'<span class="form-dot {"win" if r["prediction_correct"] else "loss"}" title="{r["date"].strftime("%m/%d")} vs {r["opponent"] if r["team"]==team_pick else r["team"]}">'
            f'{"W" if r["prediction_correct"] else "L"}</span>'
            for _, r in last10.iterrows()
        ])
        st.markdown(f'<div style="margin-bottom:16px"><div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Last {len(last10)} predictions form</div>'
                    f'<div class="form-guide">{form_dots}</div></div>', unsafe_allow_html=True)

        # Opponent breakdown (H2H accuracy)
        section_header("Head-to-Head vs Opponents", "· prediction accuracy by matchup")
        h2h_rows = []
        for opp in sorted(set(team_hist["team"].tolist() + team_hist["opponent"].tolist()) - {team_pick}):
            g = team_hist[(team_hist["team"]==opp) | (team_hist["opponent"]==opp)]
            if len(g) < 2: continue
            h2h_rows.append(dict(
                Opponent=opp, Games=len(g),
                Accuracy=round(g["prediction_correct"].mean()*100, 1),
                ModelPicks=f'{int((g["predicted_winner"]==team_pick).sum())}-{int((g["predicted_winner"]==opp).sum())}',
            ))
        if h2h_rows:
            h2h_df = pd.DataFrame(h2h_rows).sort_values("Games", ascending=False)
            st.dataframe(h2h_df, width='stretch', hide_index=True)

        # Team prediction history
        section_header("Prediction History", f"· {t_games} games")
        thd = team_hist[["date","team","opponent","predicted_winner","confidence",
                          "home_odds","away_odds","prediction_correct"]].copy()
        thd["date"] = thd["date"].dt.strftime("%Y-%m-%d")
        thd["home_odds"] = thd["home_odds"].round(2)
        thd["away_odds"] = thd["away_odds"].round(2)
        thd["prediction_correct"] = thd["prediction_correct"].map({True:"✅", False:"❌"})
        thd.columns = ["Date","Home","Away","Predicted Winner","Confidence %","Home Odds","Away Odds","Correct"]
        st.dataframe(thd.sort_values("Date", ascending=False), width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — UPSET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if hist.empty:
        st.markdown('<div class="empty-state"><div class="icon">🔍</div>No data after applying filters.</div>', unsafe_allow_html=True)
    else:
        odds_hist = hist.dropna(subset=["home_odds","away_odds"]).copy()
        if odds_hist.empty:
            st.markdown('<div class="empty-state"><div class="icon">📊</div>No odds data in current filter selection.</div>', unsafe_allow_html=True)
        else:
            odds_hist["home_is_fav"]   = odds_hist["home_odds"] < odds_hist["away_odds"]
            odds_hist["model_picks_fav"] = (
                (odds_hist["prediction"]==True)  &  odds_hist["home_is_fav"]
            ) | ((odds_hist["prediction"]==False) & ~odds_hist["home_is_fav"])
            odds_hist["fav_won"]  = (
                (odds_hist["home_is_fav"]==True) & (odds_hist["winner"]==True)
            ) | ((odds_hist["home_is_fav"]==False) & (odds_hist["winner"]==False))
            odds_hist["upset"] = ~odds_hist["fav_won"]

            # Model confidence vs implied probability (from odds)
            # predicted team's implied prob
            odds_hist["pred_implied_prob"] = odds_hist.apply(
                lambda r: r["home_prob"] if r["prediction"] else r["away_prob"], axis=1
            )

            fav_games = odds_hist[odds_hist["model_picks_fav"]]
            dog_games = odds_hist[~odds_hist["model_picks_fav"]]
            fav_acc   = fav_games["prediction_correct"].mean()*100 if len(fav_games) else 0
            dog_acc   = dog_games["prediction_correct"].mean()*100 if len(dog_games) else 0
            upset_rate = odds_hist["upset"].mean()*100

            section_header("Upset KPIs")
            c = st.columns(4)
            with c[0]: kpi("Accuracy (picks fav)", f"{fav_acc:.1f}%", color=accuracy_color(fav_acc),
                            tooltip="How often the model was correct when it picked the betting favourite "
                                    "(the team with lower decimal odds).")
            with c[1]: kpi("Accuracy (picks dog)", f"{dog_acc:.1f}%", color=accuracy_color(dog_acc),
                            tooltip="How often the model was correct when it picked the underdog "
                                    "(higher decimal odds team).")
            with c[2]: kpi("Overall upset rate", f"{upset_rate:.1f}%", color="yellow",
                            tooltip="% of games where the betting underdog (higher decimal odds) actually won. "
                                    "The NBA typically sees ~30-35% upsets per season.")
            with c[3]: kpi("Games with odds", str(len(odds_hist)),
                            tooltip="Number of games in the current filter that have both home and away decimal odds available. "
                                    "Early-season records may be missing odds.")

            divider()

            col_left, col_right = st.columns(2)

            # ── Implied prob vs model confidence scatter ───────────────────────
            with col_left:
                section_header("Implied Probability vs Model Confidence")
                st.caption("Each dot = one game. X = market-implied win probability for predicted team. "
                           "Y = model confidence. Green = correct, red = incorrect.")
                scatter_df = odds_hist.copy()
                scatter_df["result_color"] = scatter_df["prediction_correct"].map({True:"#34d399", False:"#f87171"})
                scatter_df["result_label"] = scatter_df["prediction_correct"].map({True:"Correct","False":"Incorrect"})
                fig_sc = go.Figure()
                for correct_val, color, label in [(True,"#34d399","Correct"),(False,"#f87171","Incorrect")]:
                    sub = scatter_df[scatter_df["prediction_correct"]==correct_val]
                    fig_sc.add_trace(go.Scatter(
                        x=sub["pred_implied_prob"], y=sub["confidence"],
                        mode="markers", name=label,
                        marker=dict(color=color, size=6, opacity=0.6, line=dict(color="rgba(0,0,0,0)")),
                        hovertemplate=f"{label}<br>Mkt implied: %{{x:.1f}}%<br>Model conf: %{{y:.1f}}%<br>"
                                      f"%{{customdata[0]}} vs %{{customdata[1]}}<extra></extra>",
                        customdata=sub[["team","opponent"]].values,
                    ))
                # y=x diagonal
                fig_sc.add_trace(go.Scatter(x=[45,100], y=[45,100], mode="lines", name="y = x (agree)",
                                             line=dict(color="rgba(148,163,184,0.3)", dash="dash", width=1),
                                             showlegend=True))
                fig_sc.update_layout(**PLOTLY_LAYOUT,
                                      xaxis_title="Market-implied win probability (%)",
                                      yaxis_title="Model confidence (%)",
                                      legend=dict(bgcolor="#0f172a", orientation="h", x=0, y=1.1),
                                      xaxis_range=[40,105], yaxis_range=[40,105])
                st.plotly_chart(fig_sc, width='stretch')

            # ── Top upset teams ────────────────────────────────────────────────
            with col_right:
                section_header("Top Teams by Upset Rate", "· min 5 games")
                team_upset = []
                for team in sorted(set(odds_hist["team"]) | set(odds_hist["opponent"])):
                    g = odds_hist[(odds_hist["team"]==team) | (odds_hist["opponent"]==team)]
                    if len(g) < 5: continue
                    team_upset.append(dict(Team=team, Upsets=int(g["upset"].sum()),
                                           Games=len(g), Upset_Rate=round(g["upset"].mean()*100,1)))
                if team_upset:
                    udf = pd.DataFrame(team_upset).sort_values("Upset_Rate", ascending=False)
                    fig_tu = go.Figure(go.Bar(
                        x=udf.head(15)["Upset_Rate"], y=udf.head(15)["Team"],
                        orientation="h", marker_color="#f472b6",
                        text=[f"{v:.1f}%" for v in udf.head(15)["Upset_Rate"]],
                        textposition="outside",
                        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
                    ))
                    fig_tu.update_layout(**PLOTLY_LAYOUT, xaxis_range=[0,80],
                                          xaxis_title="Upset rate %", yaxis_title="", height=420)
                    fig_tu.update_layout(margin=dict(t=12,b=36,l=60,r=60))
                    st.plotly_chart(fig_tu, width='stretch')

            divider()

            # ── Confidence bucket vs upset rate ───────────────────────────────
            section_header("Model Confidence vs Upset Rate", "· ideally decreasing left-to-right")
            bins, bucket_labels = [49,60,70,80,90,101], ["50-60","61-70","71-80","81-90","90+"]
            odds_hist["conf_bucket"] = pd.cut(
                (odds_hist["confidence"] + 0.5).astype(int),
                bins=bins, labels=bucket_labels, right=True,
            )
            conf_upset = (
                odds_hist.groupby("conf_bucket", observed=True)
                .agg(upset_rate_b=("upset","mean"), n=("upset","count")).reset_index()
            )
            conf_upset["upset_rate_pct"] = conf_upset["upset_rate_b"] * 100
            fig_cu = go.Figure()
            bucket_colors = ["#a78bfa","#a78bfa","#3b82f6","#22d3ee","#34d399"]
            for i, (_, row) in enumerate(conf_upset.iterrows()):
                fig_cu.add_trace(go.Bar(
                    x=[str(row["conf_bucket"])], y=[row["upset_rate_pct"]],
                    text=[f"{row['upset_rate_pct']:.1f}%<br>n={row['n']}"],
                    texttemplate="%{text}", textposition="outside",
                    marker_color=bucket_colors[i], showlegend=False,
                    hovertemplate=f"{row['conf_bucket']}: %{{y:.1f}}%  (n={row['n']})<extra></extra>",
                ))
            fig_cu.update_layout(**PLOTLY_LAYOUT, yaxis_range=[0,80],
                                  xaxis_title="Model confidence bucket (%)", yaxis_title="Upset rate (%)")
            st.plotly_chart(fig_cu, width='stretch')

            # Edge insight
            high_conf = conf_upset[conf_upset["conf_bucket"]=="90+"]["upset_rate_pct"].values
            low_conf  = conf_upset[conf_upset["conf_bucket"]=="50-60"]["upset_rate_pct"].values
            if len(high_conf) and len(low_conf):
                diff = low_conf[0] - high_conf[0]
                insight(f"In the 50-60% confidence bucket, upsets occur <b>{low_conf[0]:.1f}%</b> of the time. "
                        f"In the 90%+ bucket: <b>{high_conf[0]:.1f}%</b>. "
                        f"That's a <b>{diff:.1f}pp</b> spread, showing {'good' if diff > 5 else 'limited'} confidence signal.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ODDS & BETTING
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    odds_df = hist.dropna(subset=["home_odds","away_odds"]).copy()

    if odds_df.empty:
        st.markdown('<div class="empty-state"><div class="icon">💰</div>No odds data after applying filters.</div>', unsafe_allow_html=True)
    else:
        odds_df = odds_df.sort_values("date")

        odds_df["profit_model"]   = odds_df.apply(lambda r: profit(1, r["home_odds"] if r["prediction"] else r["away_odds"], r["prediction_correct"]), axis=1)
        odds_df["profit_against"] = odds_df.apply(lambda r: profit(1, r["away_odds"] if r["prediction"] else r["home_odds"], not r["prediction_correct"]), axis=1)
        odds_df["profit_home"]    = odds_df.apply(lambda r: profit(1, r["home_odds"], bool(r["winner"])), axis=1)
        # Always favourite: bet on the team with better (lower) odds regardless of model
        def _profit_fav(r):
            is_home_fav = r["home_odds"] < r["away_odds"]
            fav_odds    = r["home_odds"] if is_home_fav else r["away_odds"]
            fav_correct = bool(r["winner"]) if is_home_fav else not bool(r["winner"])
            return profit(1, fav_odds, fav_correct)
        odds_df["profit_fav"] = odds_df.apply(_profit_fav, axis=1)

        # Daily aggregation
        daily_pnl = (
            odds_df.groupby(odds_df["date"].dt.date)
            .agg(pnl_model=("profit_model","sum"), pnl_against=("profit_against","sum"),
                 pnl_home=("profit_home","sum"), pnl_fav=("profit_fav","sum"))
            .reset_index().rename(columns={"date":"ds"}).sort_values("ds")
        )
        daily_pnl["cum_model"]   = daily_pnl["pnl_model"].cumsum()
        daily_pnl["cum_against"] = daily_pnl["pnl_against"].cumsum()
        daily_pnl["cum_home"]    = daily_pnl["pnl_home"].cumsum()
        daily_pnl["cum_fav"]     = daily_pnl["pnl_fav"].cumsum()

        n_odds        = len(odds_df)
        correct_odds  = int(odds_df["prediction_correct"].sum())
        earliest_odds = odds_df["date"].min().strftime("%Y-%m-%d")
        total_staked  = n_odds
        roi_model   = daily_pnl["cum_model"].iloc[-1]   / total_staked * 100
        roi_against = daily_pnl["cum_against"].iloc[-1] / total_staked * 100
        roi_home    = daily_pnl["cum_home"].iloc[-1]    / total_staked * 100
        roi_fav     = daily_pnl["cum_fav"].iloc[-1]     / total_staked * 100

        # Max drawdown: largest peak-to-trough drop on the cumulative curve
        def max_drawdown(cum_series):
            running_max = cum_series.cummax()
            return (cum_series - running_max).min()

        dd_model   = max_drawdown(daily_pnl["cum_model"])
        dd_against = max_drawdown(daily_pnl["cum_against"])
        dd_home    = max_drawdown(daily_pnl["cum_home"])
        dd_fav     = max_drawdown(daily_pnl["cum_fav"])

        # Best/worst single day
        best_day  = daily_pnl.loc[daily_pnl["pnl_model"].idxmax()]
        worst_day = daily_pnl.loc[daily_pnl["pnl_model"].idxmin()]

        section_header("Betting KPIs", f"· {n_odds:,} games with odds · €1 flat stake")
        st.caption(
            "Simulation assumes a flat €1 stake on every game where odds are available. "
            "**P&L** (profit & loss) per game = odds × €1 − €1 if correct, else −€1. "
        )
        c = st.columns(6)
        with c[0]: kpi("Odds tracking since", earliest_odds,
                        tooltip="Date of the first game for which betting odds are available.")
        with c[1]: kpi("Model ROI %", f"{roi_model:+.1f}%", color="green" if roi_model>0 else "red",
                        tooltip="Return on investment: net profit ÷ total staked × 100. Positive = profitable.")
        with c[2]: kpi("Against-model ROI %", f"{roi_against:+.1f}%", color="green" if roi_against>0 else "red",
                        tooltip="What you'd earn by always betting the OPPOSITE of the model's pick.")
        with c[3]: kpi("Always-home ROI %", f"{roi_home:+.1f}%", color="green" if roi_home>0 else "red",
                        tooltip="What you'd earn by always betting on the home team, ignoring the model.")
        with c[4]: kpi("Max drawdown (model)", f"€{dd_model:.2f}", color="red",
                        tooltip="Largest peak-to-trough loss on the model's cumulative profit curve. "
                                "e.g. €-8.50 means at worst you were €8.50 below your previous high point.")
        with c[5]: kpi("Correct predictions", f"{correct_odds:,}",
                        tooltip="Number of model predictions that were correct, on games where odds are available.")

        c2 = st.columns(4)
        with c2[0]: kpi("Total wagered", f"€{total_staked:,}",
                         tooltip="Total amount staked across all games (€1 per game).")
        with c2[1]: kpi("Always-fav ROI %", f"{roi_fav:+.1f}%", color="green" if roi_fav>0 else "red",
                         tooltip="What you'd earn by always betting on the market favourite (lower decimal odds). "
                                 "This is the market benchmark — beating it means the model adds value beyond just picking favourites.")
        with c2[2]: kpi("Best day (model)", f"€{best_day['pnl_model']:+.2f}",
                         delta=str(best_day['ds']), delta_dir="pos")
        with c2[3]: kpi("Worst day (model)", f"€{worst_day['pnl_model']:+.2f}",
                         delta=str(worst_day['ds']), delta_dir="neg")

        # Auto-insight
        strategies = [("Model", roi_model), ("Against-model", roi_against), ("Always-home", roi_home), ("Always-fav", roi_fav)]
        best_strategy = max(strategies, key=lambda x: x[1])
        if best_strategy[1] > 0:
            insight(f"Best strategy over this period: <b>{best_strategy[0]}</b> with <b>{best_strategy[1]:+.1f}%</b> ROI "
                    f"on €{total_staked} staked. Net profit: <b>€{best_strategy[1]/100*total_staked:.2f}</b>.")
        else:
            insight(f"All strategies are currently unprofitable over this filter window. "
                    f"Best ROI: <b>{best_strategy[0]}</b> at <b>{best_strategy[1]:+.1f}%</b>.")

        divider()

        # ── Cumulative profit chart ────────────────────────────────────────────
        section_header("Cumulative Profit", "· daily aggregated · €1 per game")
        fig_bet = go.Figure()
        fig_bet.add_trace(go.Scatter(x=daily_pnl["ds"], y=daily_pnl["cum_model"],
                                      name="Model", line=dict(color="#3b82f6", width=2.5), mode="lines",
                                      fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"))
        fig_bet.add_trace(go.Scatter(x=daily_pnl["ds"], y=daily_pnl["cum_against"],
                                      name="Against model", line=dict(color="#f472b6", width=2), mode="lines"))
        fig_bet.add_trace(go.Scatter(x=daily_pnl["ds"], y=daily_pnl["cum_home"],
                                      name="Always home", line=dict(color="#34d399", width=2), mode="lines"))
        fig_bet.add_trace(go.Scatter(x=daily_pnl["ds"], y=daily_pnl["cum_fav"],
                                      name="Always favourite", line=dict(color="#fbbf24", width=2, dash="dot"), mode="lines"))
        fig_bet.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,0.3)", annotation_text="Break even")
        _annotate_chart(fig_bet, daily_pnl["ds"].min(), daily_pnl["ds"].max())
        fig_bet.update_layout(**PLOTLY_LAYOUT, xaxis_title="Date", yaxis_title="Cumulative profit (€)",
                               legend=dict(bgcolor="#0f172a", orientation="h", x=0, y=1.08))
        st.plotly_chart(fig_bet, width='stretch')

        # ── Daily P&L bar chart ────────────────────────────────────────────────
        section_header("Daily P&L by Strategy")
        fig_bar = go.Figure()
        for col, name, color in [("pnl_model","Model","#3b82f6"),("pnl_against","Against model","#f472b6"),
                                   ("pnl_home","Always home","#34d399"),("pnl_fav","Always favourite","#fbbf24")]:
            fig_bar.add_trace(go.Bar(x=daily_pnl["ds"], y=daily_pnl[col], name=name, marker_color=color, opacity=0.75))
        fig_bar.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,0.3)")
        fig_bar.update_layout(**PLOTLY_LAYOUT, barmode="group", xaxis_title="Date", yaxis_title="Daily profit (€)",
                               legend=dict(bgcolor="#0f172a", orientation="h", x=0, y=1.08))
        st.plotly_chart(fig_bar, width='stretch')

        divider()

        # ── Monthly P&L summary ────────────────────────────────────────────────
        section_header("Monthly P&L Summary", "· model strategy")
        odds_df["month"] = odds_df["date"].dt.to_period("M").astype(str)
        monthly = (
            odds_df.groupby("month")
            .agg(games=("profit_model","count"),
                 pnl_model=("profit_model","sum"),
                 pnl_against=("profit_against","sum"),
                 pnl_home=("profit_home","sum"),
                 correct=("prediction_correct","sum"))
            .reset_index()
        )
        monthly["acc"]     = (monthly["correct"] / monthly["games"] * 100).round(1)
        monthly["roi"]     = (monthly["pnl_model"] / monthly["games"] * 100).round(1)
        monthly["pnl_model"]   = monthly["pnl_model"].round(2)
        monthly["pnl_against"] = monthly["pnl_against"].round(2)
        monthly["pnl_home"]    = monthly["pnl_home"].round(2)
        monthly.columns = ["Month","Games","P&L Model €","P&L Against €","P&L Home €","Correct","Accuracy %","Model ROI %"]
        monthly = monthly[["Month","Games","Accuracy %","Model ROI %","P&L Model €","P&L Against €","P&L Home €"]]
        st.dataframe(monthly.sort_values("Month", ascending=False), width='stretch', hide_index=True)

        divider()

        # ── Per-game P&L table ────────────────────────────────────────────────
        section_header("Per-game P&L")
        pnl = odds_df[["date","team","opponent","predicted_winner","confidence","prediction_correct","profit_model","profit_against","profit_home"]].copy()
        pnl["date"] = pnl["date"].dt.strftime("%Y-%m-%d")
        pnl["prediction_correct"] = pnl["prediction_correct"].map({True:"✅", False:"❌"})
        pnl.columns = ["Date","Home","Away","Predicted Winner","Conf %","Correct","P&L Model €","P&L Against €","P&L Home €"]
        st.dataframe(pnl.sort_values("Date", ascending=False), width='stretch', hide_index=True)
