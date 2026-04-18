import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.charts import PLOTLY_LAYOUT, annotate_chart
from ui.components import divider, insight, kpi, section_header
from ui.components import profit as calc_profit


def _max_drawdown(cum_series: pd.Series) -> float:
    running_max = cum_series.cummax()
    return (cum_series - running_max).min()


def render(hist: pd.DataFrame) -> None:
    odds_df = hist.dropna(subset=["home_odds", "away_odds"]).copy()

    if odds_df.empty:
        st.markdown('<div class="empty-state"><div class="icon">💰</div>No odds data after applying filters.</div>', unsafe_allow_html=True)
        return

    odds_df = odds_df.sort_values("date")

    odds_df["profit_model"]   = odds_df.apply(
        lambda r: calc_profit(1, r["home_odds"] if r["prediction"] else r["away_odds"], r["prediction_correct"]), axis=1
    )
    odds_df["profit_against"] = odds_df.apply(
        lambda r: calc_profit(1, r["away_odds"] if r["prediction"] else r["home_odds"], not r["prediction_correct"]), axis=1
    )
    odds_df["profit_home"]    = odds_df.apply(
        lambda r: calc_profit(1, r["home_odds"], bool(r["winner"])), axis=1
    )

    def _profit_fav(r):
        is_home_fav = r["home_odds"] < r["away_odds"]
        fav_odds    = r["home_odds"] if is_home_fav else r["away_odds"]
        fav_correct = bool(r["winner"]) if is_home_fav else not bool(r["winner"])
        return calc_profit(1, fav_odds, fav_correct)
    odds_df["profit_fav"] = odds_df.apply(_profit_fav, axis=1)

    def _profit_rec(r):
        if pd.isna(r.get("baseline_correct")):
            return 0.0
        rec_odds = r["home_odds"] if bool(r["baseline_picks_home"]) else r["away_odds"]
        return calc_profit(1, rec_odds, bool(r["baseline_correct"]))
    odds_df["profit_record"] = odds_df.apply(_profit_rec, axis=1)

    # Daily aggregation
    daily_pnl = (
        odds_df.groupby(odds_df["date"].dt.date)
        .agg(pnl_model=("profit_model", "sum"), pnl_against=("profit_against", "sum"),
             pnl_home=("profit_home", "sum"), pnl_fav=("profit_fav", "sum"),
             pnl_record=("profit_record", "sum"))
        .reset_index().rename(columns={"date": "ds"}).sort_values("ds")
    )
    daily_pnl["cum_model"]   = daily_pnl["pnl_model"].cumsum()
    daily_pnl["cum_against"] = daily_pnl["pnl_against"].cumsum()
    daily_pnl["cum_home"]    = daily_pnl["pnl_home"].cumsum()
    daily_pnl["cum_fav"]     = daily_pnl["pnl_fav"].cumsum()
    daily_pnl["cum_record"]  = daily_pnl["pnl_record"].cumsum()

    n_odds        = len(odds_df)
    correct_odds  = int(odds_df["prediction_correct"].sum())
    earliest_odds = odds_df["date"].min().strftime("%Y-%m-%d")
    total_staked  = n_odds
    roi_model   = daily_pnl["cum_model"].iloc[-1]   / total_staked * 100
    roi_against = daily_pnl["cum_against"].iloc[-1] / total_staked * 100
    roi_home    = daily_pnl["cum_home"].iloc[-1]    / total_staked * 100
    roi_fav     = daily_pnl["cum_fav"].iloc[-1]     / total_staked * 100
    roi_record  = daily_pnl["cum_record"].iloc[-1]  / total_staked * 100

    dd_model   = _max_drawdown(daily_pnl["cum_model"])
    dd_against = _max_drawdown(daily_pnl["cum_against"])
    dd_home    = _max_drawdown(daily_pnl["cum_home"])
    dd_fav     = _max_drawdown(daily_pnl["cum_fav"])
    dd_record  = _max_drawdown(daily_pnl["cum_record"])

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
    with c[1]: kpi("Model ROI %", f"{roi_model:+.1f}%", color="green" if roi_model > 0 else "red",
                    tooltip="Return on investment: net profit ÷ total staked × 100. Positive = profitable.")
    with c[2]: kpi("Against-model ROI %", f"{roi_against:+.1f}%", color="green" if roi_against > 0 else "red",
                    tooltip="What you'd earn by always betting the OPPOSITE of the model's pick.")
    with c[3]: kpi("Always-home ROI %", f"{roi_home:+.1f}%", color="green" if roi_home > 0 else "red",
                    tooltip="What you'd earn by always betting on the home team, ignoring the model.")
    with c[4]: kpi("Max drawdown (model)", f"€{dd_model:.2f}", color="red",
                    tooltip="Largest peak-to-trough loss on the model's cumulative profit curve.")
    with c[5]: kpi("Correct predictions", f"{correct_odds:,}",
                    tooltip="Number of model predictions that were correct, on games where odds are available.")

    c2 = st.columns(5)
    with c2[0]: kpi("Total wagered", f"€{total_staked:,}",
                     tooltip="Total amount staked across all games (€1 per game).")
    with c2[1]: kpi("Always-fav ROI %", f"{roi_fav:+.1f}%", color="green" if roi_fav > 0 else "red",
                     tooltip="What you'd earn by always betting on the market favourite (lower decimal odds).")
    with c2[2]: kpi("Better-record ROI %", f"{roi_record:+.1f}%", color="green" if roi_record > 0 else "red",
                     tooltip="What you'd earn by always betting on the team with the better season record going into each game. Ties go to home team.")
    with c2[3]: kpi("Best day (model)", f"€{best_day['pnl_model']:+.2f}",
                     delta=str(best_day['ds']), delta_dir="pos")
    with c2[4]: kpi("Worst day (model)", f"€{worst_day['pnl_model']:+.2f}",
                     delta=str(worst_day['ds']), delta_dir="neg")

    strategies    = [("Model", roi_model), ("Against-model", roi_against), ("Always-home", roi_home),
                     ("Always-fav", roi_fav), ("Better-record", roi_record)]
    best_strategy = max(strategies, key=lambda x: x[1])
    if best_strategy[1] > 0:
        insight(f"Best strategy over this period: <b>{best_strategy[0]}</b> with <b>{best_strategy[1]:+.1f}%</b> ROI "
                f"on €{total_staked} staked. Net profit: <b>€{best_strategy[1]/100*total_staked:.2f}</b>.")
    else:
        insight(f"All strategies are currently unprofitable over this filter window. "
                f"Best ROI: <b>{best_strategy[0]}</b> at <b>{best_strategy[1]:+.1f}%</b>.")

    divider()

    # ── Cumulative profit chart ────────────────────────────────────────────────
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
    fig_bet.add_trace(go.Scatter(x=daily_pnl["ds"], y=daily_pnl["cum_record"],
                                  name="Better record", line=dict(color="#22d3ee", width=2, dash="dot"), mode="lines"))
    fig_bet.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,0.3)", annotation_text="Break even")
    annotate_chart(fig_bet, daily_pnl["ds"].min(), daily_pnl["ds"].max())
    fig_bet.update_layout(**PLOTLY_LAYOUT, xaxis_title="Date", yaxis_title="Cumulative profit (€)",
                           legend=dict(bgcolor="#0f172a", orientation="h", x=0, y=1.08))
    st.plotly_chart(fig_bet, width='stretch')

    # ── Daily P&L bar chart ────────────────────────────────────────────────────
    section_header("Daily P&L by Strategy")
    fig_bar = go.Figure()
    for col, name, color in [("pnl_model", "Model", "#3b82f6"), ("pnl_against", "Against model", "#f472b6"),
                               ("pnl_home", "Always home", "#34d399"), ("pnl_fav", "Always favourite", "#fbbf24"),
                               ("pnl_record", "Better record", "#22d3ee")]:
        fig_bar.add_trace(go.Bar(x=daily_pnl["ds"], y=daily_pnl[col], name=name, marker_color=color, opacity=0.75))
    fig_bar.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,0.3)")
    fig_bar.update_layout(**PLOTLY_LAYOUT, barmode="group", xaxis_title="Date", yaxis_title="Daily profit (€)",
                           legend=dict(bgcolor="#0f172a", orientation="h", x=0, y=1.08))
    st.plotly_chart(fig_bar, width='stretch')

    divider()

    # ── Monthly P&L summary ────────────────────────────────────────────────────
    section_header("Monthly P&L Summary", "· model strategy")
    odds_df["month"] = odds_df["date"].dt.to_period("M").astype(str)
    monthly = (
        odds_df.groupby("month")
        .agg(games=("profit_model", "count"),
             pnl_model=("profit_model", "sum"),
             pnl_against=("profit_against", "sum"),
             pnl_home=("profit_home", "sum"),
             pnl_record=("profit_record", "sum"),
             correct=("prediction_correct", "sum"))
        .reset_index()
    )
    monthly["acc"] = (monthly["correct"] / monthly["games"] * 100).round(1)
    monthly["roi"] = (monthly["pnl_model"] / monthly["games"] * 100).round(1)
    monthly["pnl_model"]   = monthly["pnl_model"].round(2)
    monthly["pnl_against"] = monthly["pnl_against"].round(2)
    monthly["pnl_home"]    = monthly["pnl_home"].round(2)
    monthly["pnl_record"]  = monthly["pnl_record"].round(2)
    monthly.columns = ["Month", "Games", "P&L Model €", "P&L Against €", "P&L Home €", "P&L Record €",
                        "Correct", "Accuracy %", "Model ROI %"]
    monthly = monthly[["Month", "Games", "Accuracy %", "Model ROI %", "P&L Model €", "P&L Against €",
                        "P&L Home €", "P&L Record €"]]
    st.dataframe(monthly.sort_values("Month", ascending=False), width='stretch', hide_index=True)

    divider()

    # ── Per-game P&L table ────────────────────────────────────────────────────
    section_header("Per-game P&L")
    pnl = odds_df[["date", "team", "opponent", "predicted_winner", "confidence", "prediction_correct",
                    "profit_model", "profit_against", "profit_home", "profit_record"]].copy()
    pnl["date"]               = pnl["date"].dt.strftime("%Y-%m-%d")
    pnl["prediction_correct"] = pnl["prediction_correct"].map({True: "✅", False: "❌"})
    pnl.columns = ["Date", "Home", "Away", "Predicted Winner", "Conf %", "Correct",
                    "P&L Model €", "P&L Against €", "P&L Home €", "P&L Record €"]
    st.dataframe(pnl.sort_values("Date", ascending=False), width='stretch', hide_index=True)
