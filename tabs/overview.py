import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.charts import PLOTLY_LAYOUT, annotate_chart
from ui.components import accuracy_color, divider, insight, kpi, section_header


def render(hist: pd.DataFrame, curr_raw: pd.DataFrame) -> None:
    ml_hist = hist.dropna(subset=["prediction_correct"])
    total:        int   = len(ml_hist)
    correct:      int   = int(ml_hist["prediction_correct"].sum()) if total else 0
    accuracy:     float = correct / total * 100 if total else 0.0
    home_win_pct: float = hist["winner"].mean() * 100 if len(hist) else 0.0
    earliest = ml_hist["date"].min().strftime("%Y-%m-%d") if total else "—"
    latest   = ml_hist["date"].max().strftime("%Y-%m-%d") if total else "—"

    last30 = ml_hist[ml_hist["date"] >= (ml_hist["date"].max() - pd.Timedelta(days=30))]
    acc30  = last30["prediction_correct"].mean() * 100 if len(last30) else accuracy
    delta_acc = acc30 - accuracy
    delta_str = f"{'▲' if delta_acc >= 0 else '▼'} {abs(delta_acc):.1f}% vs overall (30-day)"
    delta_dir = "pos" if delta_acc >= 0 else "neg"

    streak_val, streak_type = 0, ""
    if total:
        recent = ml_hist.sort_values("date")["prediction_correct"].tolist()
        streak_val = 1
        streak_type = "W" if recent[-1] else "L"
        for r in reversed(recent[:-1]):
            if bool(r) == (streak_type == "W"):
                streak_val += 1
            else:
                break

    n_today = len(curr_raw)

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
            ml_hist.groupby(ml_hist["date"].dt.date)
            .agg(total_preds=("prediction_correct", "count"), correct_preds=("prediction_correct", "sum"))
            .reset_index().rename(columns={"date": "ds"}).sort_values("ds")
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
        annotate_chart(fig, daily["ds"].min(), daily["ds"].max())
        fig.update_layout(**PLOTLY_LAYOUT, legend=dict(bgcolor="#0f172a", x=0, y=1.1, orientation="h"))
        fig.update_yaxes(title_text="Predictions", secondary_y=False, gridcolor="rgba(148,163,184,0.07)")
        fig.update_yaxes(title_text="Accuracy %",  secondary_y=True,  range=[0, 100], gridcolor="rgba(148,163,184,0.07)")
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
    hist_disp = hist[["date", "team", "opponent", "predicted_winner", "confidence", "prediction_correct", "model_version"]].copy()
    hist_disp["date"] = hist_disp["date"].dt.strftime("%Y-%m-%d")
    hist_disp["prediction_correct"] = hist_disp["prediction_correct"].map({True: "✅", False: "❌"})
    hist_disp.columns = ["Date", "Home", "Away", "Predicted Winner", "Confidence %", "Correct", "Model"]
    hist_disp = hist_disp.sort_values("Date", ascending=False)

    page_size = 25
    total_pages = max(1, (len(hist_disp) - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, label_visibility="collapsed")
    st.dataframe(hist_disp.iloc[(page - 1) * page_size: page * page_size], width='stretch', hide_index=True)
    st.caption(f"Page {page} of {total_pages}  ·  {len(hist_disp):,} records total")
