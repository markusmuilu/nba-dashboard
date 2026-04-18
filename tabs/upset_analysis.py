import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.charts import PLOTLY_LAYOUT
from ui.components import accuracy_color, divider, insight, kpi, section_header


def render(hist: pd.DataFrame) -> None:
    if hist.empty:
        st.markdown('<div class="empty-state"><div class="icon">🔍</div>No data after applying filters.</div>', unsafe_allow_html=True)
        return

    odds_hist = hist.dropna(subset=["home_odds", "away_odds"]).copy()
    if odds_hist.empty:
        st.markdown('<div class="empty-state"><div class="icon">📊</div>No odds data in current filter selection.</div>', unsafe_allow_html=True)
        return

    odds_hist["home_is_fav"]    = odds_hist["home_odds"] < odds_hist["away_odds"]
    odds_hist["model_picks_fav"] = (
        (odds_hist["prediction"] == True)  &  odds_hist["home_is_fav"]
    ) | ((odds_hist["prediction"] == False) & ~odds_hist["home_is_fav"])
    odds_hist["fav_won"] = (
        (odds_hist["home_is_fav"] == True)  & (odds_hist["winner"] == True)
    ) | ((odds_hist["home_is_fav"] == False) & (odds_hist["winner"] == False))
    odds_hist["upset"] = ~odds_hist["fav_won"]

    odds_hist["pred_implied_prob"] = odds_hist.apply(
        lambda r: r["home_prob"] if r["prediction"] else r["away_prob"], axis=1
    )

    fav_games  = odds_hist[odds_hist["model_picks_fav"]]
    dog_games  = odds_hist[~odds_hist["model_picks_fav"]]
    fav_acc    = fav_games["prediction_correct"].mean() * 100 if len(fav_games) else 0
    dog_acc    = dog_games["prediction_correct"].mean() * 100 if len(dog_games) else 0
    upset_rate = odds_hist["upset"].mean() * 100

    section_header("Upset KPIs")
    c = st.columns(4)
    with c[0]: kpi("Accuracy (picks fav)", f"{fav_acc:.1f}%", color=accuracy_color(fav_acc),
                    tooltip="How often the model was correct when it picked the betting favourite.")
    with c[1]: kpi("Accuracy (picks dog)", f"{dog_acc:.1f}%", color=accuracy_color(dog_acc),
                    tooltip="How often the model was correct when it picked the underdog.")
    with c[2]: kpi("Overall upset rate", f"{upset_rate:.1f}%", color="yellow",
                    tooltip="% of games where the betting underdog actually won. NBA typically ~30-35%.")
    with c[3]: kpi("Games with odds", str(len(odds_hist)),
                    tooltip="Number of games in the current filter with both home and away decimal odds.")

    divider()

    col_left, col_right = st.columns(2)

    with col_left:
        section_header("Implied Probability vs Model Confidence")
        st.caption("Each dot = one game. X = market-implied win probability for predicted team. "
                   "Y = model confidence. Green = correct, red = incorrect.")
        scatter_df = odds_hist.copy()
        fig_sc = go.Figure()
        for correct_val, color, label in [(True, "#34d399", "Correct"), (False, "#f87171", "Incorrect")]:
            sub = scatter_df[scatter_df["prediction_correct"] == correct_val]
            fig_sc.add_trace(go.Scatter(
                x=sub["pred_implied_prob"], y=sub["confidence"],
                mode="markers", name=label,
                marker=dict(color=color, size=6, opacity=0.6, line=dict(color="rgba(0,0,0,0)")),
                hovertemplate=f"{label}<br>Mkt implied: %{{x:.1f}}%<br>Model conf: %{{y:.1f}}%<br>"
                              f"%{{customdata[0]}} vs %{{customdata[1]}}<extra></extra>",
                customdata=sub[["team", "opponent"]].values,
            ))
        fig_sc.add_trace(go.Scatter(x=[45, 100], y=[45, 100], mode="lines", name="y = x (agree)",
                                    line=dict(color="rgba(148,163,184,0.3)", dash="dash", width=1),
                                    showlegend=True))
        fig_sc.update_layout(**PLOTLY_LAYOUT,
                              xaxis_title="Market-implied win probability (%)",
                              yaxis_title="Model confidence (%)",
                              legend=dict(bgcolor="#0f172a", orientation="h", x=0, y=1.1),
                              xaxis_range=[40, 105], yaxis_range=[40, 105])
        st.plotly_chart(fig_sc, width='stretch')

    with col_right:
        section_header("Top Teams by Upset Rate", "· min 5 games")
        team_upset = []
        for team in sorted(set(odds_hist["team"]) | set(odds_hist["opponent"])):
            g = odds_hist[(odds_hist["team"] == team) | (odds_hist["opponent"] == team)]
            if len(g) < 5:
                continue
            team_upset.append(dict(Team=team, Upsets=int(g["upset"].sum()),
                                   Games=len(g), Upset_Rate=round(g["upset"].mean() * 100, 1)))
        if team_upset:
            udf = pd.DataFrame(team_upset).sort_values("Upset_Rate", ascending=False)
            fig_tu = go.Figure(go.Bar(
                x=udf.head(15)["Upset_Rate"], y=udf.head(15)["Team"],
                orientation="h", marker_color="#f472b6",
                text=[f"{v:.1f}%" for v in udf.head(15)["Upset_Rate"]],
                textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig_tu.update_layout(**PLOTLY_LAYOUT, xaxis_range=[0, 80],
                                  xaxis_title="Upset rate %", yaxis_title="", height=420)
            fig_tu.update_layout(margin=dict(t=12, b=36, l=60, r=60))
            st.plotly_chart(fig_tu, width='stretch')

    divider()

    # ── Confidence bucket vs upset rate ──────────────────────────────────────
    section_header("Model Confidence vs Upset Rate", "· ideally decreasing left-to-right")
    bins, bucket_labels = [49, 60, 70, 80, 90, 101], ["50-60", "61-70", "71-80", "81-90", "90+"]
    odds_hist["conf_bucket"] = pd.cut(
        (odds_hist["confidence"] + 0.5).astype(int),
        bins=bins, labels=bucket_labels, right=True,
    )
    conf_upset = (
        odds_hist.groupby("conf_bucket", observed=True)
        .agg(upset_rate_b=("upset", "mean"), n=("upset", "count")).reset_index()
    )
    conf_upset["upset_rate_pct"] = conf_upset["upset_rate_b"] * 100
    fig_cu = go.Figure()
    bucket_colors = ["#a78bfa", "#a78bfa", "#3b82f6", "#22d3ee", "#34d399"]
    for i, (_, row) in enumerate(conf_upset.iterrows()):
        fig_cu.add_trace(go.Bar(
            x=[str(row["conf_bucket"])], y=[row["upset_rate_pct"]],
            text=[f"{row['upset_rate_pct']:.1f}%<br>n={row['n']}"],
            texttemplate="%{text}", textposition="outside",
            marker_color=bucket_colors[i], showlegend=False,
            hovertemplate=f"{row['conf_bucket']}: %{{y:.1f}}%  (n={row['n']})<extra></extra>",
        ))
    fig_cu.update_layout(**PLOTLY_LAYOUT, yaxis_range=[0, 80],
                          xaxis_title="Model confidence bucket (%)", yaxis_title="Upset rate (%)")
    st.plotly_chart(fig_cu, width='stretch')

    high_conf = conf_upset[conf_upset["conf_bucket"] == "90+"]["upset_rate_pct"].values
    low_conf  = conf_upset[conf_upset["conf_bucket"] == "50-60"]["upset_rate_pct"].values
    if len(high_conf) and len(low_conf):
        diff = low_conf[0] - high_conf[0]
        insight(f"In the 50-60% confidence bucket, upsets occur <b>{low_conf[0]:.1f}%</b> of the time. "
                f"In the 90%+ bucket: <b>{high_conf[0]:.1f}%</b>. "
                f"That's a <b>{diff:.1f}pp</b> spread, showing {'good' if diff > 5 else 'limited'} confidence signal.")
