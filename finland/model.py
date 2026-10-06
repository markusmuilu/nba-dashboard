"""
Rating models for Finnish men's basketball, using only team results (no player data exists for most of it).

Two families, both strictly point-in-time (a game's prediction uses only earlier games):

ELO     Rating difference plus a home bonus gives the win probability 1 / (1 + 10^(-d / scale)); after the
        game both ratings move by K * margin multiplier * (result - expected). As in the NBA work.

MARGIN  Each team's rating is its expected point margin against an average team. The predicted margin of a
        game is home rating - away rating + home advantage (in points); after the game both ratings move by
        K * (observed margin - predicted margin); the win probability is Phi(predicted margin / sigma).
        This uses the whole score, not just who won, which matters when a team plays only ~25 games a season.

The "observed margin" can be defined from the quarter scores, which exist for almost every match:
  final    the final margin
  q3       the margin after three quarters (less affected by fouling and garbage time at the end)
  half     the halftime margin
  mix      the average of final and q3
Overtime games are scored by the margin after regulation for q3/half/mix, so overtime is not rewarded.

Seasons: at each new season every rating is pulled by `carry` towards the level of the league the team last played in, and a team seen for
the first time starts at its league's starting value. In the national model (Korisliiga, I divisioona A and B
in one pool) the starting values differ by league, so a promoted team keeps the rating it earned and a new
team enters at the level of its league.

Everything is tuned by random search on the log loss of two seasons (2023-24 and 2024-25), with 2022-23 as
warm-up, and then scored once on 2025-26. 2026-27 is the live season: its finished games are scored the same
way and its fixtures get predictions.
"""

import json
import math

import numpy as np
import pandas as pd

from finland import CACHE_DIR, SEED
from finland.metrics import classification_metrics

DATA = CACHE_DIR / "matches.csv"
TUNE_SEASONS = ["2023-2024", "2024-2025"]
TEST_SEASON = "2025-2026"
LIVE_SEASON = "2026-2027"
MARGIN_MODES = ["final", "q3", "half", "mix"]


def load_games(pool):
    """pool: 'national' (KL + I-A + I-B) or 'm2d'. Returns finished and upcoming games in date order."""
    df = pd.read_csv(DATA)
    leagues = {"national": ["KL", "I-A", "I-B"], "m2d": ["M2D"]}[pool]
    df = df[df.league.isin(leagues)].copy()
    df["when"] = pd.to_datetime(df["date"].fillna("") + " " + df["time"].fillna("00:00").astype(str), errors="coerce")
    df = df[df["when"].notna()].sort_values(["when", "match_id"]).reset_index(drop=True)
    for side in ("home", "away"):
        reg = sum(df[f"{side}_q{q}"] for q in (1, 2, 3, 4))
        df[f"{side}_reg"] = reg                               # points in regulation (NaN if a quarter is missing)
        df[f"{side}_q3sum"] = sum(df[f"{side}_q{q}"] for q in (1, 2, 3))
        df[f"{side}_halfsum"] = df[f"{side}_q1"] + df[f"{side}_q2"]
    final = df.home_pts - df.away_pts
    df["m_final"] = final
    df["m_q3"] = (df.home_q3sum - df.away_q3sum).fillna(final)
    df["m_half"] = (df.home_halfsum - df.away_halfsum).fillna(final)
    df["m_mix"] = (df.m_final.where(df.home_reg.isna(), df.home_reg - df.away_reg) + df.m_q3) / 2     # regulation margin, mixed
    df["m_final_reg"] = (df.home_reg - df.away_reg).fillna(final)
    df["home_win"] = (df.home_pts > df.away_pts).astype(float)
    df["season_idx"] = df.season.map({s: i for i, s in enumerate(sorted(df.season.unique()))})
    return df


def run(games, family, params, margin_mode, league_init=None):
    """
    Returns arrays aligned with games: win probability before the game, predicted margin before the game, and the
    final rating table. Unfinished games get a prediction but no update.
    """
    K, home, carry, scale = params["K"], params["home"], params["carry"], params["scale"]
    mov_exp = params.get("mov_exp", 0.8)
    init = {"KL": 0.0, "I-A": params.get("init_ia", 0.0), "I-B": params.get("init_ib", 0.0), "M2D": 0.0}
    rating, team_league, last_season = {}, {}, None
    n = len(games)
    prob, pred_margin = np.zeros(n), np.zeros(n)
    mode_col = {"final": "m_final_reg", "q3": "m_q3", "half": "m_half", "mix": "m_mix"}[margin_mode]
    cols = games[["home_id", "away_id", "season", "league", "finished", "home_win", "m_final", mode_col]].to_numpy(object)
    mean_by_league = {}
    for i, (h, a, season, league, finished, win, m_final, m_obs) in enumerate(cols):
        if season != last_season:
            for t in rating:                                       # pull towards the level of the league the team last played in,
                base = init[team_league[t]]                        # so a strong I divisioona team cannot drift above Korisliiga teams
                rating[t] = base + carry * (rating[t] - base)
            last_season = season
        rh = rating.setdefault(h, init[league])
        ra = rating.setdefault(a, init[league])
        team_league[h] = team_league[a] = league
        if family == "elo":
            d = rh + home - ra
            p = 1.0 / (1.0 + 10.0 ** (-max(-1500.0, min(1500.0, d)) / scale))
            pm = d                                                # rating gap; turned into points by a regression later
        else:
            pm = rh + home - ra
            p = 0.5 * (1.0 + math.erf(pm / scale / math.sqrt(2.0)))
        prob[i], pred_margin[i] = p, pm
        if not finished:
            continue
        if family == "elo":
            mult = (abs(m_obs) + 3.0) ** mov_exp / (7.5 + 0.006 * (d if win == 1 else -d)) if mov_exp > 0 else 1.0
            shift = K * mult * (win - p)
        else:
            shift = K * (m_obs - pm)
        rating[h], rating[a] = rh + shift, ra - shift
    return prob, pred_margin, rating


def score(games, prob, seasons):
    m = games.season.isin(seasons) & games.finished
    return classification_metrics(games.home_win[m], prob[m])


def sample(family, pool, rng):
    p = {"K": float(np.exp(rng.uniform(np.log(2), np.log(40)))) if family == "elo" else float(rng.uniform(0.01, 0.30)),
         "carry": float(rng.uniform(0.3, 1.0)),
         "scale": float(rng.uniform(250, 700)) if family == "elo" else float(rng.uniform(8, 18)),
         "home": float(rng.uniform(0, 100)) if family == "elo" else float(rng.uniform(0, 8)),
         "mov_exp": float(rng.choice([0.0, 0.5, 0.8, 1.0]))}
    if pool == "national":
        spread = 150 if family == "elo" else 8
        p["init_ia"], p["init_ib"] = float(rng.uniform(-spread, 0)), float(rng.uniform(-spread, 0))
    return p


def tune(games, family, pool, margin_mode, n=300, seed=SEED):
    rng = np.random.default_rng(seed)
    best = None
    rows = []
    for _ in range(n):
        p = sample(family, pool, rng)
        prob, _, _ = run(games, family, p, margin_mode)
        s = score(games, prob, TUNE_SEASONS)
        rows.append({**p, "tune_log_loss": s["log_loss"], "tune_accuracy": s["accuracy"]})
        if best is None or s["log_loss"] < best[0]:
            best = (s["log_loss"], p)
    return best[1], best[0], rows


def baselines(games):
    """
    1. constant: the home win rate of all EARLIER seasons (0.57 for the first season), so no look-ahead;
    2. record pick: the team with the better win percentage so far wins, ties go to the home team (a pick, so accuracy only).
    """
    prior = {}
    for s in sorted(games.season.unique()):
        past = games[(games.season < s) & games.finished]
        prior[s] = float(past.home_win.mean()) if len(past) else 0.57
    const = games.season.map(prior).to_numpy()
    wins, played, pick = {}, {}, np.zeros(len(games))
    for i, g in enumerate(games.itertuples()):
        ph = wins.get(g.home_id, 0) / played[g.home_id] if played.get(g.home_id) else 0.5
        pa = wins.get(g.away_id, 0) / played[g.away_id] if played.get(g.away_id) else 0.5
        pick[i] = 1.0 if ph >= pa else 0.0
        if g.finished:
            wins[g.home_id] = wins.get(g.home_id, 0) + g.home_win
            wins[g.away_id] = wins.get(g.away_id, 0) + 1 - g.home_win
            played[g.home_id] = played.get(g.home_id, 0) + 1
            played[g.away_id] = played.get(g.away_id, 0) + 1
    return const, pick
