"""
Builds the data behind the dashboard's Finland tab: refreshes the results, fits the rating models, and returns a
JSON-ready dictionary (ratings, upcoming fixtures with win probabilities, season-so-far, backtest, a few facts).

Two pools, each its own model:
  national  Korisliiga + Miesten I divisioona A and B, one rating pool (a promoted team keeps its rating)
  m2d       Miesten II divisioona (regional groups), its own model and settings

For each pool every combination of rating family (Elo, margin rating) and margin definition (final, after three
quarters, halftime, mix) is tuned on 2023-24 and 2024-25 and scored on 2025-26. The published model is the one
with the best tuning score; all combinations are listed on the page.

Run: python -m finland.pipeline [--no-refresh]      writes site/finland.json
"""

import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from finland import model as M
from finland.metrics import calibration_table, classification_metrics

LABELS = {"national": "Korisliiga and I divisioona", "m2d": "II divisioona (M2D)"}
LEAGUE_NAMES = {"KL": "Korisliiga", "I-A": "I divisioona A", "I-B": "I divisioona B", "M2D": "II divisioona"}
N_SEARCH = 1500
UPCOMING_DAYS = 21
FOCUS_TEAM = "Aalto-Basket"      # marked with a star on the page and given its own card; set to None to switch off


def paired_gain(y, p_base, p_new, n_boot=2000, seed=0):
    y = np.asarray(y, float)
    ll = lambda p: -(y * np.log(np.clip(p, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1 - 1e-9)))
    d = ll(np.asarray(p_base)) - ll(np.asarray(p_new))
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}


def margin_points(games, family, pm):
    """Predicted margin in points. The margin family predicts points; for Elo, regress the margin on the rating gap."""
    if family == "margin":
        return pm
    m = games.season.isin(M.TUNE_SEASONS) & games.finished
    a, b = np.polyfit(pm[m], games.m_final[m], 1)
    return a * pm + b


def run_pool(pool, now):
    games = M.load_games(pool)
    const, pick = M.baselines(games)
    fin = games.finished
    split = {"tune": games.season.isin(M.TUNE_SEASONS) & fin, "test": (games.season == M.TEST_SEASON) & fin,
             "live": (games.season == M.LIVE_SEASON) & fin}
    combos, store = [], {}
    for family in ("elo", "margin"):
        for mode in M.MARGIN_MODES:
            params, tune_ll, _ = M.tune(games, family, pool, mode, n=N_SEARCH)
            prob, pm, rating = M.run(games, family, params, mode)
            met = {k: classification_metrics(games.home_win[m], prob[m]) for k, m in split.items() if m.sum() > 5}
            combos.append({"family": family, "mode": mode, "tune": met["tune"]["log_loss"], "test": met["test"]["log_loss"],
                           "test_accuracy": met["test"]["accuracy"], "test_brier": met["test"]["brier"]})
            store[(family, mode)] = (prob, pm, rating, params, met)
    best = min(combos, key=lambda c: c["tune"])
    best["chosen"] = True
    family, mode = best["family"], best["mode"]
    prob, pm, rating, params, met = store[(family, mode)]
    pts = margin_points(games, family, pm)

    out = games.assign(p_home=prob, pred_margin=pts, const=const)
    test_m = split["test"]
    by_league = {lg: classification_metrics(out.home_win[test_m & (out.league == lg)], out.p_home[test_m & (out.league == lg)])
                 for lg in sorted(out.league.unique()) if (test_m & (out.league == lg)).sum() > 20}
    base_test = classification_metrics(games.home_win[test_m], const[test_m])
    result = {
        "label": LABELS[pool], "chosen": {"family": family, "margin_mode": mode},
        "params": {k: round(float(v), 4) for k, v in params.items()},
        "n_games_total": int(fin.sum()), "seasons": sorted(games.season.unique()),
        "metrics": {k: v for k, v in met.items()},
        "by_league_test": by_league,
        "baseline_test": {"constant": base_test, "record_pick_accuracy": float((pick[test_m] == games.home_win[test_m]).mean())},
        "gain_vs_constant_test": paired_gain(games.home_win[test_m], const[test_m], prob[test_m]),
        "calibration_test": calibration_table(games.home_win[test_m], prob[test_m], bins=6).round(4).to_dict("records"),
        "combos": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in combos],
    }

    # Ratings of the teams in the live season
    live = games[games.season == M.LIVE_SEASON]
    ids = set(live.home_id) | set(live.away_id)
    names = pd.concat([games[["home_id", "home", "league", "when"]].rename(columns={"home_id": "id", "home": "name"}),
                       games[["away_id", "away", "league", "when"]].rename(columns={"away_id": "id", "away": "name"})]).sort_values("when")
    last = names.groupby("id").last()
    done = live[live.finished]
    rec = {}
    for g in done.itertuples():
        for tid, won in ((g.home_id, g.home_win == 1), (g.away_id, g.home_win == 0)):
            w, l = rec.get(tid, (0, 0))
            rec[tid] = (w + won, l + (not won))
    unit = "points" if family == "margin" else "Elo points"
    result["rating_unit"] = unit
    result["ratings"] = sorted([{"team": last.loc[t, "name"], "league": last.loc[t, "league"], "rating": round(float(rating.get(t, 0.0)), 2),
                                 "won": int(rec.get(t, (0, 0))[0]), "lost": int(rec.get(t, (0, 0))[1])} for t in ids],
                               key=lambda r: -r["rating"])

    # Upcoming fixtures
    horizon = now + timedelta(days=UPCOMING_DAYS)
    up = out[(~out.finished) & (out["when"] >= pd.Timestamp(now.replace(tzinfo=None) - timedelta(hours=3))) & (out["when"] <= pd.Timestamp(horizon.replace(tzinfo=None)))]
    result["upcoming"] = [{"when": r.when.strftime("%Y-%m-%d %H:%M"), "league": r.league, "home": r.home, "away": r.away,
                           "p_home": round(float(r.p_home), 3), "margin": round(float(r.pred_margin), 1)} for r in up.itertuples()]

    # Season so far, and a few facts
    d = out[(out.season == M.LIVE_SEASON) & out.finished].sort_values("when", ascending=False)
    d = d.assign(correct=((d.p_home >= 0.5) == (d.home_win == 1)))
    result["recent"] = [{"when": r.when.strftime("%Y-%m-%d"), "league": r.league, "home": r.home, "away": r.away,
                         "score": f"{int(r.home_pts)}-{int(r.away_pts)}", "p_home": round(float(r.p_home), 3), "correct": bool(r.correct)}
                        for r in d.head(60).itertuples()]
    result["season_so_far"] = {"games": int(len(d)), "correct": int(d.correct.sum()),
                               "accuracy": float(d.correct.mean()) if len(d) else None}
    facts = {}
    prev = out[(out.season == M.TEST_SEASON) & out.finished]
    facts["avg_home_margin"] = {lg: round(float(prev[prev.league == lg].m_final.mean()), 1) for lg in sorted(prev.league.unique())}
    facts["home_win_rate"] = {lg: round(float((prev[prev.league == lg].home_win).mean()), 3) for lg in sorted(prev.league.unique())}
    if len(d):
        won_p = np.where(d.home_win == 1, d.p_home, 1 - d.p_home)
        u = d.assign(winner_p=won_p).sort_values("winner_p").iloc[0]
        winner = u.home if u.home_win == 1 else u.away
        facts["biggest_upset"] = {"winner": winner, "home": u.home, "away": u.away, "score": f"{int(u.home_pts)}-{int(u.away_pts)}",
                                  "winner_probability": round(float(u.winner_p), 3), "league": u.league, "date": u["when"].strftime("%Y-%m-%d")}
    result["facts"] = facts

    # One team to follow: its rank, record and next games, wherever they fall
    result["focus"] = None
    if FOCUS_TEAM:
        names_in_ratings = [r["team"] for r in result["ratings"]]
        if FOCUS_TEAM in names_in_ratings:
            i = names_in_ratings.index(FOCUS_TEAM)
            row = result["ratings"][i]
            mine = out[(~out.finished) & ((out.home == FOCUS_TEAM) | (out.away == FOCUS_TEAM)) & (out["when"] >= pd.Timestamp(now.replace(tzinfo=None) - timedelta(hours=3)))].sort_values("when")
            result["focus"] = {"team": FOCUS_TEAM, "rank": i + 1, "of": len(names_in_ratings), "rating": row["rating"],
                               "won": row["won"], "lost": row["lost"], "league": row["league"],
                               "next": [{"when": r.when.strftime("%Y-%m-%d %H:%M"), "home": r.home, "away": r.away,
                                         "p_win": round(float(r.p_home if r.home == FOCUS_TEAM else 1 - r.p_home), 3),
                                         "venue": "home" if r.home == FOCUS_TEAM else "away"} for r in mine.head(3).itertuples()]}
    return result


def build(refresh=True, now=None):
    if refresh:
        from finland import scrape
        scrape.main()
    now = now or datetime.now(timezone(timedelta(hours=3)))          # Finnish time, close enough for a 21-day window
    return {"generated_at": now.strftime("%Y-%m-%d %H:%M"), "league_names": LEAGUE_NAMES,
            "pools": {pool: run_pool(pool, now) for pool in ("national", "m2d")}}


def main():
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="reuse cached results instead of calling the results service")
    args = ap.parse_args()
    data = build(refresh=not args.no_refresh)
    out = Path(__file__).resolve().parents[1] / "site" / "finland.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), default=str))
    snap = Path(__file__).resolve().parents[1] / "site_data" / "finland_snapshot.json"
    snap.write_text(json.dumps(data, separators=(",", ":"), default=str))
    for p, r in data["pools"].items():
        print(p, r["chosen"], "test accuracy", round(r["metrics"]["test"]["accuracy"], 3), "| upcoming", len(r["upcoming"]), "| ratings", len(r["ratings"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
