from datetime import datetime, date

import pandas as pd


# ── Model version mapping ─────────────────────────────────────────────────────

def model_version(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return "Unknown"
    if d <= date(2025, 12, 5):    return "Logistic reg V1"
    elif d <= date(2025, 12, 14): return "Logistic reg V2.1"
    elif d <= date(2026, 1, 8):   return "Custom NN V1"
    else:                          return "Logistic reg V2.2"


VERSION_COLOURS: dict[str, str] = {
    "Logistic reg V1":   "#94a3b8",
    "Logistic reg V2.1": "#a78bfa",
    "Custom NN V1":      "#fbbf24",
    "Logistic reg V2.2": "#34d399",
}

VERSION_BADGE_CLASS: dict[str, str] = {
    "Logistic reg V1":   "ver-v1",
    "Logistic reg V2.1": "ver-v21",
    "Custom NN V1":      "ver-nn",
    "Logistic reg V2.2": "ver-v22",
}


# ── Season type classification ────────────────────────────────────────────────

PLAYIN_STARTS: dict[int, pd.Timestamp] = {
    2025: pd.Timestamp("2025-04-15"),
    2026: pd.Timestamp("2026-04-14"),
}
PLAYOFF_STARTS: dict[int, pd.Timestamp] = {
    2025: pd.Timestamp("2025-04-19"),
    2026: pd.Timestamp("2026-04-19"),
}


def get_season_type(d: pd.Timestamp) -> str:
    # NBA seasons end in June; Oct-Dec belong to the season ending *next* year
    season_year = d.year + 1 if d.month >= 7 else d.year
    playoff = PLAYOFF_STARTS.get(season_year)
    if playoff and d >= playoff:
        return "Playoffs"
    playin = PLAYIN_STARTS.get(season_year)
    if playin and d >= playin:
        return "Play-In"
    return "Regular Season"


# ── Chart event annotations ───────────────────────────────────────────────────
# Each tuple: (ISO date, label, hex colour)

ALL_EVENTS: list[tuple[str, str, str]] = [
    # Model version changes
    ("2025-12-06", "→ V2.1",   "#a78bfa"),
    ("2025-12-15", "→ NN V1",  "#fbbf24"),
    ("2026-01-09", "→ V2.2",   "#34d399"),
    # NBA calendar
    ("2026-02-06", "Trade deadline",   "#f472b6"),
    ("2026-02-13", "All-Star break",   "#94a3b8"),
    ("2026-04-13", "Reg. season ends", "#64748b"),
    ("2026-04-14", "Playoffs start",   "#22d3ee"),
]
