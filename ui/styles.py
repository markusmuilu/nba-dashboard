import streamlit as st

_CSS = """
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
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
