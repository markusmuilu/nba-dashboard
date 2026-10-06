"""
Builds the static dashboard: site/data.json (all numbers) next to site/index.html (the page).

Why static: the Streamlit app slept after inactivity, and visitors of the portfolio
saw a sleep screen instead of data. A page made of plain files has nothing to sleep.
A scheduled GitHub Action runs this script once a day and publishes the result.

Data sources:
- The prediction history and today's predictions, from Cloudflare R2 (read only),
  the same two JSON files the Streamlit app read.
- The model comparison, from site_data/model_comparison.json. That file is a copy of
  research/results/metrics.json from the Predicting-Nba repo (branch player-model).
  It is a snapshot because the comparison is a fixed experiment on the 2025-26 season.

Run:
    python build_site.py                      # reads R2, needs the four R2_* variables
    python build_site.py --history path.json  # offline: a local copy of prediction_history.json
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
CURRENT_SEASON = "2026-27"
LAST_SEASON = "2025-26"

# Which production model made the prediction on a given day. Same dates as config/constants.py.
MODEL_VERSIONS = [
    ("2025-12-05", "Logistic reg V1"),
    ("2025-12-14", "Logistic reg V2.1"),
    ("2026-01-08", "Custom NN V1"),
    ("9999-12-31", "Logistic reg V2.2"),
]
EXCLUDED_TEAMS = {"STARS", "STRIPES", "WORLD", "NO_GAMES_TODAY"}  # All-Star games, placeholder


def model_version(date_str):
    for last_day, name in MODEL_VERSIONS:
        if date_str <= last_day:
            return name
    return "Unknown"


def season_of(date_str):
    d = pd.Timestamp(date_str)
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}-{str(start + 1)[2:]}"


# ── Loading ───────────────────────────────────────────────────────────────────

def read_r2(key):
    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    bucket = os.environ.get("R2_BUCKET_NAME", "nbaprediction")
    return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())


def prepare(rows):
    """Raw history rows -> one tidy frame. p_home is the model's home win probability."""
    df = pd.DataFrame(rows)
    df = df[~df["team"].isin(EXCLUDED_TEAMS) & ~df["opponent"].isin(EXCLUDED_TEAMS)].copy()
    df = df.drop_duplicates("gameId").sort_values(["date", "gameId"]).reset_index(drop=True)
    # Early-season rows carry a result but no stored prediction (the service was not
    # yet predicting those games). They cannot be scored, so they are dropped here and counted.
    df.attrs["n_without_prediction"] = int(df["confidence"].isna().sum())
    df = df[df["confidence"].notna()].copy()
    for col in ("home_odds", "away_odds"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    conf = df["confidence"].astype(float) / 100          # confidence in the predicted side
    df["pred_home"] = df["prediction"].astype(bool)
    df["p_home"] = np.where(df["pred_home"], conf, 1 - conf)
    df["home_win"] = df["winner"].astype(bool)
    df["correct"] = df["pred_home"] == df["home_win"]
    df["season"] = df["date"].map(season_of)
    df["model_version"] = df["date"].map(model_version)
    return df


# ── Summaries ─────────────────────────────────────────────────────────────────

def scores(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))) if len(y) else None,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else None,
    }


def calibration(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return [{"mean_pred": float(p[idx == b].mean()), "observed": float(y[idx == b].mean()), "count": int((idx == b).sum())}
            for b in range(bins) if (idx == b).any()]


def running_accuracy(df):
    """Cumulative accuracy after each game day, plus the 20-game rolling accuracy."""
    out = df.sort_values(["date", "gameId"]).reset_index(drop=True)
    out["cum_acc"] = out["correct"].expanding().mean()
    out["roll20"] = out["correct"].rolling(20, min_periods=20).mean()
    last_per_day = out.groupby("date").tail(1)
    return [{"date": r.date, "n": int(i + 1), "cumulative": round(float(r.cum_acc), 4),
             "rolling20": None if pd.isna(r.roll20) else round(float(r.roll20), 4)}
            for i, r in last_per_day.iterrows()]


def season_block(df):
    block = {"scores": scores(df.home_win, df.p_home), "home_win_rate": float(df.home_win.mean()) if len(df) else None,
             "calibration": calibration(df.home_win, df.p_home) if len(df) else [],
             "running": running_accuracy(df) if len(df) else []}
    return block


def games_table(df, limit=None):
    cols = ["date", "team", "opponent", "p_home", "home_win", "correct", "home_odds", "away_odds"]
    t = df.sort_values(["date", "gameId"], ascending=[False, False])[cols]
    if limit:
        t = t.head(limit)
    return json.loads(t.round(4).to_json(orient="records"))


def team_accuracy(df):
    long = pd.concat([df[["team", "correct"]].rename(columns={"team": "t"}),
                      df[["opponent", "correct"]].rename(columns={"opponent": "t"})])
    g = long.groupby("t")["correct"].agg(["mean", "count"]).reset_index().sort_values("mean", ascending=False)
    return [{"team": r.t, "accuracy": round(float(r["mean"]), 4), "games": int(r["count"])} for _, r in g.iterrows()]


def odds_archive(df):
    """Flat 1-unit stake on the predicted side, for games that had Pinnacle odds stored."""
    o = df[df.home_odds.notna() & df.away_odds.notna()].copy().sort_values(["date", "gameId"])
    if o.empty:
        return None
    price = np.where(o.pred_home, o.home_odds, o.away_odds)
    o["profit"] = np.where(o.correct, price - 1, -1.0)
    imp_home = (1 / o.home_odds) / (1 / o.home_odds + 1 / o.away_odds)
    favourite_is_home = imp_home >= 0.5
    picks_fav = o.pred_home == favourite_is_home
    daily = o.groupby("date")["profit"].sum().cumsum()
    return {
        "n": int(len(o)), "first_date": o.date.min(),
        "roi": float(o.profit.mean()), "total_profit": float(o.profit.sum()),
        "accuracy_when_picking_favourite": float(o.correct[picks_fav].mean()),
        "n_picking_favourite": int(picks_fav.sum()),
        "accuracy_when_picking_underdog": float(o.correct[~picks_fav].mean()) if (~picks_fav).any() else None,
        "n_picking_underdog": int((~picks_fav).sum()),
        "cumulative": [{"date": d, "profit": round(float(v), 3)} for d, v in daily.items()],
    }


# ── Build ─────────────────────────────────────────────────────────────────────

def build(history_rows, current_rows, comparison):
    df = prepare(history_rows)
    n_unscored = df.attrs.get("n_without_prediction", 0)
    this = df[df.season == CURRENT_SEASON]
    last = df[df.season == LAST_SEASON]

    cur = pd.DataFrame(current_rows) if current_rows else pd.DataFrame()
    if not cur.empty and "team" in cur:
        cur = cur[~cur["team"].isin(EXCLUDED_TEAMS)]
    today = [] if cur.empty else json.loads(cur[["date", "team", "opponent", "confidence", "prediction"]].to_json(orient="records"))

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "current_season": CURRENT_SEASON,
        "last_season": LAST_SEASON,
        "history_range": {"first": df.date.min(), "last": df.date.max(), "n": int(len(df)),
                          "n_without_prediction": n_unscored},
        "season_now": {**season_block(this), "games": games_table(this, 200), "today": today},
        "season_last": {**season_block(last), "odds": odds_archive(last), "teams": team_accuracy(last),
                        "games": games_table(last, 300),
                        "by_version": {v: scores(g.home_win, g.p_home)
                                       for v, g in last.groupby("model_version")}},
        "comparison": comparison,
    }


def comparison_block(path):
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    test = m["splits"]["test"]
    games = pd.DataFrame(m["test_games"])
    curves = {}
    for key in m["models"]:
        col = f"p_{key}"
        if col in games and games[col].notna().all():
            curves[key] = calibration(games.home_win, games[col])
    if games["p_market"].notna().any():
        odds = games[games.p_market.notna()]
        curves["market"] = calibration(odds.home_win, odds.p_market)
    return {
        "models": m["models"],
        "test": {"season": test["season"], "n_games": test["n_games"], "metrics": test["metrics"],
                 "paired_vs_logreg": test["paired_vs_logreg"],
                 "paired_vs_scalars": test.get("paired_vs_scalars", {}), "odds": test.get("odds"),
                 "live_production": test.get("live_production")},
        "validation": {"season": m["splits"]["validation"]["season"], "metrics": m["splits"]["validation"]["metrics"]},
        "calibration": curves,
        "jev_available": any(k.startswith("jev") for k in m["models"]),
    }


def build_finland():
    """
    The Finland tab: refresh results from the Finnish Basketball Association's service and refit the rating models.
    If that fails (the service is down, or blocks the runner), the last committed snapshot is published instead,
    and the page shows its own timestamp, so stale data is visible as stale.
    """
    snapshot = HERE / "site_data" / "finland_snapshot.json"
    try:
        from finland.pipeline import build
        data = build(refresh=True)
        snapshot.write_text(json.dumps(data, separators=(",", ":"), default=str))
        source = "live"
    except Exception as e:
        print(f"Finland pipeline failed ({type(e).__name__}: {e}); using the committed snapshot")
        if not snapshot.exists():
            return
        data, source = json.loads(snapshot.read_text()), "snapshot"
    (SITE / "finland.json").write_text(json.dumps(data, separators=(",", ":"), default=str))
    print(f"Wrote site/finland.json ({source}, generated {data['generated_at']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", help="local prediction_history.json instead of R2")
    ap.add_argument("--current", help="local current_predictions.json instead of R2")
    ap.add_argument("--skip-finland", action="store_true", help="do not refresh the Finland tab")
    args = ap.parse_args()

    history = json.loads(Path(args.history).read_text()) if args.history else read_r2("history/prediction_history.json")
    if args.current:
        current = json.loads(Path(args.current).read_text())
    elif args.history:
        current = []
    else:
        current = read_r2("current/current_predictions.json")

    data = build(history, current, comparison_block(HERE / "site_data" / "model_comparison.json"))
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    if not args.skip_finland:
        build_finland()
    print(f"Wrote site/data.json: {data['history_range']['n']} games, "
          f"{data['season_now']['scores']['n']} in {CURRENT_SEASON}, {data['season_last']['scores']['n']} in {LAST_SEASON}")


if __name__ == "__main__":
    main()
