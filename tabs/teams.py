import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.charts import PLOTLY_LAYOUT
from ui.components import accuracy_color, divider, kpi, section_header
from ui.components import profit as calc_profit


def render(hist: pd.DataFrame) -> None:
    if hist.empty:
        st.markdown('<div class="empty-state"><div class="icon">🔍</div>No data after applying filters.</div>', unsafe_allow_html=True)
        return

    all_involved_teams = sorted(set(hist["team"]) | set(hist["opponent"]))

    team_rows = []
    for team in all_involved_teams:
        home     = hist[hist["team"] == team]
        away     = hist[hist["opponent"] == team]
        combined = pd.concat([home, away])
        if combined.empty:
            continue
        games         = len(combined)
        acc           = combined["prediction_correct"].mean() * 100
        correct_n     = int(combined["prediction_correct"].sum())
        pred_wins     = int((home["prediction"] == True).sum() + (away["prediction"] == False).sum())
        actual_wins   = int((home["winner"] == True).sum() + (away["winner"] == False).sum())
        team_odds     = combined.dropna(subset=["home_odds", "away_odds"])
        if len(team_odds):
            pnl = sum(calc_profit(1, r["home_odds"] if r["prediction"] else r["away_odds"],
                                  bool(r["prediction_correct"])) for _, r in team_odds.iterrows())
            roi = pnl / len(team_odds) * 100
        else:
            pnl, roi = None, None
        team_rows.append(dict(
            Team=team, Games=games, Accuracy=round(acc, 1), Correct=correct_n,
            Predicted=f"{pred_wins}-{games - pred_wins}",
            Actual=f"{actual_wins}-{games - actual_wins}",
            PnL=round(pnl, 2) if pnl is not None else None,
            ROI=round(roi, 1) if roi is not None else None,
        ))

    team_df = pd.DataFrame(team_rows).sort_values("Games", ascending=False)

    # ── Accuracy chart ────────────────────────────────────────────────────────
    section_header("Team Accuracy Ranking", f"· {len(team_df)} teams")
    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        top20  = team_df.nlargest(20, "Games").sort_values("Accuracy")
        colors = ["#34d399" if a >= 60 else "#3b82f6" if a >= 55 else "#f87171" for a in top20["Accuracy"]]
        fig_ta = go.Figure(go.Bar(
            x=top20["Accuracy"], y=top20["Team"], orientation="h",
            marker_color=colors,
            text=[f"{a:.1f}%" for a in top20["Accuracy"]], textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig_ta.add_vline(x=50, line_dash="dot", line_color="rgba(248,113,113,0.4)")
        fig_ta.update_layout(**PLOTLY_LAYOUT, xaxis_range=[30, 100],
                              xaxis_title="Accuracy %", yaxis_title="", height=520)
        fig_ta.update_layout(margin=dict(t=12, b=36, l=60, r=60))
        st.plotly_chart(fig_ta, width='stretch')

    with col_table:
        st.dataframe(
            team_df.rename(columns={"Predicted": "Pred W-L", "Actual": "Actual W-L",
                                    "PnL": "P&L (€)", "ROI": "ROI %"}),
            width='stretch', hide_index=True, height=520,
        )

    divider()

    # ── Team deep-dive ────────────────────────────────────────────────────────
    section_header("Team Deep Dive")
    team_pick = st.selectbox("Select team", all_involved_teams, key="team_pick")
    team_hist = hist[(hist["team"] == team_pick) | (hist["opponent"] == team_pick)].copy()
    team_hist = team_hist.sort_values("date")

    t_games = len(team_hist)
    t_acc   = team_hist["prediction_correct"].mean() * 100 if t_games else 0
    t_row   = team_df[team_df["Team"] == team_pick].iloc[0] if team_pick in team_df["Team"].values else None

    kpi_cols = st.columns(4)
    with kpi_cols[0]: kpi("Games tracked", str(t_games))
    with kpi_cols[1]: kpi("Model accuracy", f"{t_acc:.1f}%", color=accuracy_color(t_acc))
    with kpi_cols[2]: kpi("Actual W-L", t_row["Actual"] if t_row is not None else "—")
    with kpi_cols[3]: kpi("Predicted W-L", t_row["Predicted"] if t_row is not None else "—")

    st.markdown("<div style='margin-top:16px'>", unsafe_allow_html=True)
    last10     = team_hist.tail(10)
    form_dots  = "".join([
        f'<span class="form-dot {"win" if r["prediction_correct"] else "loss"}" '
        f'title="{r["date"].strftime("%m/%d")} vs {r["opponent"] if r["team"] == team_pick else r["team"]}">'
        f'{"W" if r["prediction_correct"] else "L"}</span>'
        for _, r in last10.iterrows()
    ])
    st.markdown(
        f'<div style="margin-bottom:16px">'
        f'<div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">'
        f'Last {len(last10)} predictions form</div>'
        f'<div class="form-guide">{form_dots}</div></div>',
        unsafe_allow_html=True,
    )

    section_header("Head-to-Head vs Opponents", "· prediction accuracy by matchup")
    h2h_rows = []
    for opp in sorted(set(team_hist["team"].tolist() + team_hist["opponent"].tolist()) - {team_pick}):
        g = team_hist[(team_hist["team"] == opp) | (team_hist["opponent"] == opp)]
        if len(g) < 2:
            continue
        h2h_rows.append(dict(
            Opponent=opp, Games=len(g),
            Accuracy=round(g["prediction_correct"].mean() * 100, 1),
            ModelPicks=f'{int((g["predicted_winner"] == team_pick).sum())}-{int((g["predicted_winner"] == opp).sum())}',
        ))
    if h2h_rows:
        h2h_df = pd.DataFrame(h2h_rows).sort_values("Games", ascending=False)
        st.dataframe(h2h_df, width='stretch', hide_index=True)

    section_header("Prediction History", f"· {t_games} games")
    thd = team_hist[["date", "team", "opponent", "predicted_winner", "confidence",
                      "home_odds", "away_odds", "prediction_correct"]].copy()
    thd["date"]               = thd["date"].dt.strftime("%Y-%m-%d")
    thd["home_odds"]          = thd["home_odds"].round(2)
    thd["away_odds"]          = thd["away_odds"].round(2)
    thd["prediction_correct"] = thd["prediction_correct"].map({True: "✅", False: "❌"})
    thd.columns = ["Date", "Home", "Away", "Predicted Winner", "Confidence %", "Home Odds", "Away Odds", "Correct"]
    st.dataframe(thd.sort_values("Date", ascending=False), width='stretch', hide_index=True)
