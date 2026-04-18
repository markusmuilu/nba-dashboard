import os
import json

import boto3
import numpy as np
import pandas as pd
import streamlit as st

from config.constants import model_version, get_season_type


def _r2_client():
    endpoint   = os.environ.get("R2_ENDPOINT")          or st.secrets.get("R2_ENDPOINT", "")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")     or st.secrets.get("R2_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY") or st.secrets.get("R2_SECRET_ACCESS_KEY", "")
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise raw JSON data into a clean DataFrame with expected columns."""
    EXPECTED_COLS = ["team", "opponent", "date", "prediction", "confidence",
                     "gameId", "model_version", "home_odds", "away_odds",
                     "predicted_winner", "home_prob", "away_prob", "season_type"]

    if df.empty or "team" not in df.columns:
        return pd.DataFrame(columns=EXPECTED_COLS)

    df = df[df["team"] != "NO_GAMES_TODAY"].copy()

    # Filter out All-Star games (team names contain "star")
    allstar_mask = (
        df["team"].str.contains("star", case=False, na=False) |
        df["opponent"].str.contains("star", case=False, na=False)
    )
    df = df[~allstar_mask].copy()

    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLS)

    df["date"]          = pd.to_datetime(df["date"])
    df["model_version"] = df["date"].dt.strftime("%Y-%m-%d").apply(model_version)
    df["season_type"]   = df["date"].apply(get_season_type)
    df["home_odds"]     = pd.to_numeric(df.get("home_odds", np.nan), errors="coerce")
    df["away_odds"]     = pd.to_numeric(df.get("away_odds", np.nan), errors="coerce")
    df["predicted_winner"] = np.where(
        df["prediction"].isna(), np.nan,
        np.where(df["prediction"], df["team"], df["opponent"])
    )
    mask = df["home_odds"].notna() & df["away_odds"].notna()
    if mask.any():
        total = 1 / df.loc[mask, "home_odds"] + 1 / df.loc[mask, "away_odds"]
        df.loc[mask, "home_prob"] = (1 / df.loc[mask, "home_odds"]) / total * 100
        df.loc[mask, "away_prob"] = 100 - df.loc[mask, "home_prob"]
    return df


def _compute_better_record_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the team with the better cumulative win % going into each game.
    Home team wins ties. Returns baseline_correct and baseline_picks_home columns."""
    df_sorted = df.sort_values("date")
    records: dict = {}
    baseline_correct: dict = {}
    baseline_picks_home: dict = {}

    for idx, row in df_sorted.iterrows():
        home, away = row["team"], row["opponent"]
        winner = row.get("winner")

        hw, hg = records.get(home, [0, 0])
        aw, ag = records.get(away, [0, 0])
        home_pct = hw / hg if hg > 0 else 0.5
        away_pct = aw / ag if ag > 0 else 0.5

        picks_home = home_pct >= away_pct
        baseline_picks_home[idx] = picks_home

        if pd.notna(winner):
            baseline_correct[idx] = picks_home == bool(winner)
            records[home] = [hw + int(bool(winner)), hg + 1]
            records[away] = [aw + int(not bool(winner)), ag + 1]
        else:
            baseline_correct[idx] = np.nan

    return pd.DataFrame({
        "baseline_correct":    pd.Series(baseline_correct),
        "baseline_picks_home": pd.Series(baseline_picks_home),
    })


@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
    bucket = os.environ.get("R2_BUCKET_NAME") or st.secrets.get("R2_BUCKET_NAME", "nbaprediction")
    obj = _r2_client().get_object(Bucket=bucket, Key="history/prediction_history.json")
    df  = pd.DataFrame(json.loads(obj["Body"].read()))
    df  = _normalise(df)
    bl  = _compute_better_record_baseline(df)
    df["baseline_correct"]    = bl["baseline_correct"]
    df["baseline_picks_home"] = bl["baseline_picks_home"]
    return df


@st.cache_data(ttl=300)
def load_current() -> pd.DataFrame:
    bucket = os.environ.get("R2_BUCKET_NAME") or st.secrets.get("R2_BUCKET_NAME", "nbaprediction")
    obj = _r2_client().get_object(Bucket=bucket, Key="current/current_predictions.json")
    df  = pd.DataFrame(json.loads(obj["Body"].read()))
    return _normalise(df)
