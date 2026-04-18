import pandas as pd
import plotly.graph_objects as go

from config.constants import ALL_EVENTS


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


def annotate_chart(fig: go.Figure, ds_min, ds_max) -> None:
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
