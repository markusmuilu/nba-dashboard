import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score, precision_score, recall_score

from config.constants import VERSION_COLOURS
from ui.charts import PLOTLY_LAYOUT
from ui.components import divider, insight, section_header, ver_badge


def render(hist: pd.DataFrame) -> None:
    if hist.empty:
        st.markdown('<div class="empty-state"><div class="icon">🔍</div>No data after applying filters.</div>', unsafe_allow_html=True)
        return

    # ── Per-version metrics ───────────────────────────────────────────────────
    ml_hist = hist.dropna(subset=["prediction_correct"])
    rows = []
    for ver, grp in ml_hist.groupby("model_version"):
        y_true = grp["winner"].astype(int).tolist()
        y_pred = grp["prediction"].astype(int).tolist()
        acc  = grp["prediction_correct"].mean() * 100
        f1   = f1_score(y_true, y_pred, zero_division=0) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec  = recall_score(y_true, y_pred, zero_division=0) * 100
        rows.append(dict(Version=ver, Games=len(grp),
                         Accuracy=round(acc, 1), F1=round(f1, 1),
                         Precision=round(prec, 1), Recall=round(rec, 1)))
    metrics_df = pd.DataFrame(rows)

    section_header("Metrics by Version")
    with st.expander("📖 What do these metrics mean? (click to expand)"):
        st.markdown("""
**Accuracy** — The simplest measure: out of every game the model predicted, what % did it get right?
*Example: 62% accuracy means 62 out of 100 predictions were correct.*

**Precision** — When the model predicted a home win, how often was it actually right?
High precision = when the model says "home wins", you can trust it.
*Example: 70% precision means 7 out of every 10 home-win predictions were correct.*

**Recall** — Out of all the games where the home team actually won, how many did the model correctly predict?
High recall = the model catches most home wins and doesn't miss them.
*Example: 60% recall means the model spotted 6 out of every 10 real home wins.*

**F1 Score** — A single number that balances precision and recall. Useful when you want both to be good, not just one.
*Think of it as the average of precision and recall, penalising large gaps between the two.*
        """)
    st.dataframe(metrics_df, width='stretch', hide_index=True)

    melted = metrics_df.melt(id_vars="Version", value_vars=["Accuracy", "F1", "Precision", "Recall"])
    fig_m = px.bar(melted, x="variable", y="value", color="Version", barmode="group",
                   color_discrete_map=VERSION_COLOURS,
                   labels={"value": "%", "variable": "Metric", "Version": "Model"})
    fig_m.update_layout(**PLOTLY_LAYOUT, legend=dict(bgcolor="#0f172a", orientation="h", x=0, y=1.08),
                         title="Classification Metrics by Model Version")
    fig_m.update_traces(hovertemplate="%{x} — %{fullData.name}: %{y:.1f}%<extra></extra>")
    st.plotly_chart(fig_m, width='stretch')

    best_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
    insight(f"Best accuracy: {ver_badge(best_row['Version'])} at <b>{best_row['Accuracy']:.1f}%</b> "
            f"over {best_row['Games']} games · F1 {best_row['F1']:.1f}%")

    divider()

    # ── Calibration + Confusion Matrix ────────────────────────────────────────
    col_cal, col_cm = st.columns(2)

    with col_cal:
        section_header("Calibration Chart")
        st.caption(
            "Bars show actual accuracy per confidence bucket. "
            "The dashed line is the mean model confidence within each bucket. "
            "Green = accuracy exceeds the bucket's lower bound. "
            "Red = accuracy falls below the bucket's lower bound."
        )
        bins   = [49, 60, 70, 80, 90, 101]
        labels = ["50-60", "61-70", "71-80", "81-90", "90+"]
        _lower_map = {"50-60": 50, "61-70": 61, "71-80": 71, "81-90": 81, "90+": 90}
        hist = hist.copy()
        hist["conf_bucket"] = pd.cut(
            (hist["confidence"] + 0.5).astype(int),
            bins=bins, labels=labels, right=True,
        )
        cal = (
            hist.groupby("conf_bucket", observed=True)
            .agg(acc=("prediction_correct", "mean"), n=("prediction_correct", "count"),
                 mean_conf=("confidence", "mean"))
            .reset_index()
        )
        cal["acc_pct"] = cal["acc"] * 100
        cal["lower"]   = cal["conf_bucket"].astype(str).map(_lower_map)
        cal["colour"]  = cal.apply(
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
            tp = int(((grp_cm["prediction"] == True)  & (grp_cm["winner"] == True)).sum())
            fp = int(((grp_cm["prediction"] == True)  & (grp_cm["winner"] == False)).sum())
            fn = int(((grp_cm["prediction"] == False) & (grp_cm["winner"] == True)).sum())
            tn = int(((grp_cm["prediction"] == False) & (grp_cm["winner"] == False)).sum())
            total_cm = tp + fp + fn + tn
            cm_text = [[f"TP\n{tp}\n{tp/total_cm*100:.0f}%", f"FP\n{fp}\n{fp/total_cm*100:.0f}%"],
                       [f"FN\n{fn}\n{fn/total_cm*100:.0f}%", f"TN\n{tn}\n{tn/total_cm*100:.0f}%"]]
            fig_cm = go.Figure(go.Heatmap(
                z=[[tp, fp], [fn, tn]], text=cm_text, texttemplate="%{text}",
                x=["Pred Home Win", "Pred Away Win"], y=["Actual Home Win", "Actual Away Win"],
                colorscale=[[0, "#0a0f1e"], [0.5, "#1e3a5f"], [1, "#3b82f6"]], showscale=False,
                hovertemplate="%{text}<extra></extra>",
            ))
            fig_cm.update_layout(**PLOTLY_LAYOUT)
            fig_cm.update_layout(margin=dict(t=20, b=36, l=48, r=24))
            ppv = tp / (tp + fp) * 100 if (tp + fp) else 0
            npv = tn / (tn + fn) * 100 if (tn + fn) else 0
            st.plotly_chart(fig_cm, width='stretch')
            st.caption(f"PPV (precision): {ppv:.1f}%  ·  NPV: {npv:.1f}%  ·  {total_cm} games")
            with st.expander("📖 What do TP / FP / FN / TN mean?"):
                st.markdown("""
**TP – True Positive** · Predicted home win ✓ and home team actually won ✓

**FP – False Positive** · Predicted home win ✓ but away team won ✗ *(false alarm)*

**FN – False Negative** · Predicted away win ✗ but home team actually won ✗ *(missed it)*

**TN – True Negative** · Predicted away win ✓ and away team actually won ✓

*Ideal matrix: large TP and TN, small FP and FN.*
                """)

    divider()

    # ── Model vs baselines ────────────────────────────────────────────────────
    divider()
    section_header("Model vs Baselines", "· how does the model compare to simple strategies?")
    ml_hist2          = hist.dropna(subset=["prediction_correct"])
    overall_acc       = ml_hist2["prediction_correct"].mean() * 100
    home_baseline     = hist["winner"].mean() * 100
    away_baseline     = (1 - hist["winner"].mean()) * 100
    better_record_acc = hist["baseline_correct"].mean() * 100

    base_labels  = ["Model", "Better record", "Always home", "Always away"]
    base_values  = [overall_acc, better_record_acc, home_baseline, away_baseline]
    base_colours = ["#3b82f6", "#22d3ee", "#94a3b8", "#a78bfa"]

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
                            showlegend=False, height=300)
    st.plotly_chart(fig_base, width='stretch')
