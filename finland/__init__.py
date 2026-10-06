"""
Finnish men's basketball ratings and predictions (Korisliiga, I divisioona, II divisioona).

Self-contained copy of the pipeline developed in the Predicting-Nba repository (research/finland), so the daily
GitHub Action can refresh the Finland tab without that repository. Data comes from the Finnish Basketball
Association's public results service; nothing is stored in the repository except a snapshot of the last good output.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "finland"
SEED = 42
