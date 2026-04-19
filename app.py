"""
NBA Prediction Analytics Dashboard
====================================
Entry point. Handles page config, CSS, sidebar, data loading, and tab routing.

Data: Cloudflare R2 bucket "nbaprediction"
    history/prediction_history.json   — resolved predictions
    current/current_predictions.json  — today's unresolved predictions
"""

from datetime import date

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from data.loader import load_current, load_history
from ui.components import ver_badge
from ui.styles import inject_css
from tabs import overview, model_performance, teams, upset_analysis, odds_betting

load_dotenv()

st.set_page_config(
    page_title="NBA Prediction Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Data loading ──────────────────────────────────────────────────────────────
try:
    hist_raw = load_history()
    curr_raw = load_current()
except Exception as exc:
    st.error(f"Failed to load data from R2: {exc}")
    st.info("Set R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME.")
    st.stop()

all_versions: list[str] = sorted(hist_raw["model_version"].unique().tolist())
all_teams:    list[str] = sorted(set(hist_raw["team"]) | set(hist_raw["opponent"]))

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

    st.markdown('<div style="font-size:0.7rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Model version</div>', unsafe_allow_html=True)
    default_ver = "Logistic reg V2.2"
    sel_versions: list[str] = [
        v for v in all_versions
        if st.checkbox(v, value=(v == default_ver if default_ver in all_versions else True), key=f"ver_{v}")
    ]
    if not sel_versions:
        sel_versions = all_versions

    st.markdown('<hr style="border:none;height:1px;background:rgba(148,163,184,0.08);margin:12px 0">', unsafe_allow_html=True)

    conf_range: tuple[int, int] = st.slider("Confidence range (%)", 50, 100, (50, 100),
                                             help="Only include predictions where the model's confidence falls in this range.")

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

    st.markdown('<hr style="border:none;height:1px;background:rgba(148,163,184,0.08);margin:12px 0">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Season type</div>', unsafe_allow_html=True)
    all_season_types = ["Regular Season", "Play-In", "Playoffs"]
    sel_season_types: list[str] = [
        t for t in all_season_types
        if st.checkbox(t, value=(t == "Regular Season"), key=f"season_{t}")
    ]
    if not sel_season_types:
        sel_season_types = all_season_types

    st.markdown('<hr style="border:none;height:1px;background:rgba(148,163,184,0.1);margin:16px 0">', unsafe_allow_html=True)

    active = []
    if sorted(sel_versions) != sorted(all_versions): active.append(f"{len(sel_versions)} version(s)")
    if conf_range != (50, 100):                      active.append(f"conf {conf_range[0]}–{conf_range[1]}%")
    if sel_team != "All teams":                      active.append(sel_team)
    if (d_start, d_end) != (min_date, max_date):    active.append("date range")
    if sorted(sel_season_types) != sorted(all_season_types): active.append(" · ".join(sel_season_types))

    if active:
        st.markdown(f'<div style="font-size:0.72rem;color:#475569">Active filters: <span style="color:#60a5fa">{" · ".join(active)}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.72rem;color:#475569">No active filters — showing all data</div>', unsafe_allow_html=True)

    st.caption("Refreshes every 5 min · R2 backed")


# ── Filter ────────────────────────────────────────────────────────────────────
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

# ── Page header ───────────────────────────────────────────────────────────────
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

with tab1:
    overview.render(hist, curr_raw)

with tab2:
    model_performance.render(hist)

with tab3:
    teams.render(hist)

with tab4:
    upset_analysis.render(hist)

with tab5:
    odds_betting.render(hist)
