import streamlit as st

from config.constants import VERSION_BADGE_CLASS


def kpi(label: str, value: str, color: str = "", delta: str = "", delta_dir: str = "neu", tooltip: str = "") -> None:
    """Render a styled KPI card. color: 'blue'|'green'|'red'|'yellow'|''."""
    tip_html = (
        f'<span class="kpi-tooltip">'
        f'<span class="kpi-tooltip-icon">i</span>'
        f'<span class="kpi-tooltip-text">{tooltip}</span>'
        f'</span>'
    ) if tooltip else ""
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
