"""
Collects Finnish men's basketball results into one table: .cache/finland/matches.csv.

Leagues, from the federation's results service (research/finland/torneopal.py):
  KL     Korisliiga
  I-A    Miesten I divisioona A
  I-B    Miesten I divisioona B
  M2D    Miesten II divisioona, one category per region (Etelainen, Keskinen, Lantinen, ...)

Columns: season, league, region (M2D only), category_id, group (regular season, playoffs, ...), match_id,
date, time, home/away team id and name, final score, the four quarter scores and overtime (home and away),
the halftime score, attendance, status, whether the statistics level is "full" (Korisliiga and I divisioona
have player and team statistics; M2D has only the match record).

"A" in the service is the home team and "B" the away team (checked against venues and the order in the
published standings). A match counts as finished when both final scores are present.

Run: python -m finland.scrape        (about 10 minutes the first time, cached afterwards)
"""

import re

import pandas as pd

from finland import CACHE_DIR
from finland.torneopal import TorneoPal

OUT = CACHE_DIR / "matches.csv"
SEASONS = ["2022-2023", "2023-2024", "2024-2025", "2025-2026", "2026-2027"]
TOP = {"Korisliiga": "KL", "Miesten I divisioona A": "I-A", "Miesten I divisioona B": "I-B"}
M2D_RE = re.compile(r"^Miesten II divisioona(?: (.+))?$")


def wanted(category):
    """League code and region for a category, or None."""
    name = (category.get("category_name") or "").strip()
    if name in TOP and category.get("category_gender") == "M":
        return TOP[name], ""
    m = M2D_RE.match(name)
    if m and category.get("category_gender") == "M":
        return "M2D", (m.group(1) or "").strip()
    return None


def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def rows_for(tp, season):
    cats = tp.call("getCategories", season_id=season)["data"].get("categories", [])
    out = []
    for c in cats:
        w = wanted(c)
        if not w:
            continue
        league, region = w
        matches = tp.call("getMatches", competition_id=c["competition_id"], category_id=c["category_id"])["data"].get("matches", [])
        for m in matches:
            fs_a, fs_b = num(m.get("fs_A")), num(m.get("fs_B"))
            out.append({
                "season": season, "league": league, "region": region, "category_id": c["category_id"],
                "competition_id": c["competition_id"], "group": m.get("group_name"), "round": m.get("round_name"),
                "match_id": m.get("match_id"), "date": m.get("date") or None, "time": m.get("time") or None,
                "home_id": m.get("team_A_id"), "home": m.get("team_A_name"), "away_id": m.get("team_B_id"), "away": m.get("team_B_name"),
                "home_club": m.get("club_A_name"), "away_club": m.get("club_B_name"),
                "home_pts": fs_a, "away_pts": fs_b,
                **{f"{side}_q{q}": num(m.get(f"p{q}s_{s}")) for q in range(1, 6) for side, s in (("home", "A"), ("away", "B"))},
                "home_half": num(m.get("hts_A")), "away_half": num(m.get("hts_B")),
                "attendance": num(m.get("attendance")), "status": m.get("status"),
                "walkover": m.get("walkover"), "forfeit": f"{m.get('forfeit_A')}/{m.get('forfeit_B')}",
                "stats_level": m.get("statistics_level") or m.get("stats_level"),
                "venue": m.get("venue_name"), "city": (m.get("venue_city_name") or "").strip(),
                "finished": fs_a is not None and fs_b is not None,
            })
    return out


def main():
    tp = TorneoPal()
    rows = []
    for season in SEASONS:
        part = rows_for(tp, season)
        rows += part
        df = pd.DataFrame(part)
        print(season, len(df), "matches;", df.groupby("league").size().to_dict() if len(df) else {}, flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, len(df), "rows;", int(df.finished.sum()), "finished")


if __name__ == "__main__":
    main()
