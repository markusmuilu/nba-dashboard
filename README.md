# NBA Prediction Analytics Dashboard

A Streamlit dashboard for analysing NBA game predictions, model performance,
team stats, upset patterns, and betting simulation — backed by data stored in
Cloudflare R2.

## Project Evolution

### V1 — Power BI (Nov 2025 – Apr 2026)
The original analytics layer was a Power BI dashboard embedded directly in the portfolio website. It visualised the same prediction data but required a Power BI Pro licence.

### V2 — Streamlit (Apr 2026–present)
When the Power BI free trial ended, the dashboard was rebuilt from scratch in Streamlit. Benefits:
- Permanently free on Streamlit Community Cloud
- Full code ownership and version control
- Tighter integration with the Cloudflare R2 data schema
- Custom visualisations using Plotly (calibration charts, drawdown curves, upset scatter)

The Streamlit dashboard is embedded via iframe in the portfolio at [markusmuilu.page](https://markusmuilu.page).

---

## Project structure

```
nba-dashboard/
├── app.py                      # Entry point: page config, sidebar, tab routing
├── requirements.txt
│
├── config/
│   └── constants.py            # Model version mapping, season type logic, chart events
│
├── data/
│   └── loader.py               # R2 client, data normalisation, better-record baseline, caching
│
├── ui/
│   ├── styles.py               # All CSS injected into Streamlit
│   ├── components.py           # kpi(), section_header(), divider(), insight(), profit()
│   └── charts.py               # PLOTLY_LAYOUT, annotate_chart()
│
└── tabs/
    ├── overview.py             # Tab 1 — Overview
    ├── model_performance.py    # Tab 2 — Model Performance
    ├── teams.py                # Tab 3 — Teams
    ├── upset_analysis.py       # Tab 4 — Upset Analysis
    └── odds_betting.py         # Tab 5 — Odds & Betting
```

---

## Tabs

| # | Tab | What it shows |
|---|-----|---------------|
| 1 | **Overview** | KPIs, 7-day rolling accuracy, today's predictions, full paginated history |
| 2 | **Model Performance** | Classification metrics by version with plain-English explanations, calibration chart, confusion matrix with TP/FP/FN/TN guide, baseline comparison |
| 3 | **Teams** | Per-team accuracy, P&L, ROI, form guide, H2H breakdown, prediction history with odds |
| 4 | **Upset Analysis** | Favourite vs underdog accuracy, implied probability scatter, confidence vs upset rate |
| 5 | **Odds & Betting** | Flat-stake P&L simulation for 5 strategies, cumulative profit chart with bucket filter and crosshair hover, daily/monthly breakdowns, per-game table |

---

## Sidebar filters

All five tabs update simultaneously when any filter changes.

- **Model version** — checkboxes, one per version; tick multiple to compare
- **Confidence range** — slider restricting to predictions in a confidence band
- **Team** — filter to games involving a specific team
- **Date range** — restricts to the selected window
- **Season type** — checkboxes for Regular Season / Play-In / Playoffs; tick multiple to combine

---

## Baselines

The dashboard computes three baselines for comparison against the ML model:

| Baseline | Logic |
|---|---|
| **Always home** | Always predicts the home team wins |
| **Always away** | Always predicts the away team wins |
| **Better record** | Picks the team with the better cumulative win % going into that game; home team wins ties |

The **Better Record** baseline is evaluated on all resolved games (including games without an ML prediction), while ML metrics only count games where a prediction was made.

---

## Confidence bucket filter

On the Odds & Betting tab, five checkboxes sit directly above the cumulative profit chart (one per confidence tier: 50–60, 61–70, 71–80, 81–90, 90+). Toggling buckets instantly refilters the chart so you can compare strategy performance within any confidence tier. KPIs and tables always reflect the full dataset.

---

## Local development

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file (or fill in .streamlit/secrets.toml)
cp .env.example .env   # then edit values

# 4. Run
streamlit run app.py
```

### Environment variables

| Variable | Description |
|---|---|
| `R2_ENDPOINT` | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | R2 API token access key |
| `R2_SECRET_ACCESS_KEY` | R2 API token secret |
| `R2_BUCKET_NAME` | Bucket name (default: `nbaprediction`) |

For local runs put these in a `.env` file (loaded by `python-dotenv`) or in `.streamlit/secrets.toml`.  
**Never commit real credentials to git.**

---

## Deploying to Streamlit Community Cloud (free)

1. **Push to GitHub**

   ```bash
   git init && git add . && git commit -m "Initial commit"
   gh repo create nba-dashboard --public --source=. --push
   ```

2. **Connect at [share.streamlit.io](https://share.streamlit.io)** — New app → select repo → main file: `app.py`

3. **Add secrets** in the app settings:

   ```toml
   R2_ENDPOINT = "https://<account-id>.r2.cloudflarestorage.com"
   R2_ACCESS_KEY_ID = "<your-access-key-id>"
   R2_SECRET_ACCESS_KEY = "<your-secret-access-key>"
   R2_BUCKET_NAME = "nbaprediction"
   ```

4. **Deploy** — Streamlit Cloud installs `requirements.txt` automatically.

Data is fetched from R2 on every page load and cached for 5 minutes (`@st.cache_data(ttl=300)`). Force a refresh via the hamburger menu → **Clear cache**.

Keep secrets out of git — add to `.gitignore`:
```
.streamlit/secrets.toml
.env
```

---

## Data schema

### `history/prediction_history.json`

Array of resolved game records. Games without an ML prediction have `prediction: null` and `prediction_correct: null` — they are included so the Better Record baseline can be computed across the full season.

```json
{
  "team": "CLE",                 // home team
  "opponent": "TOR",             // away team
  "date": "2025-11-13",
  "prediction": true,            // true = model predicts home win; null = no ML prediction
  "winner": false,               // true = home team actually won
  "prediction_correct": false,   // null when prediction is null
  "confidence": 54.42,           // model confidence %; null when no prediction
  "gameId": "401810080",
  "home_odds": 3.16,             // decimal odds; may be null for early records
  "away_odds": 1.4
}
```

### `current/current_predictions.json`

Today's unresolved predictions (no `winner` / `prediction_correct`):

```json
{
  "team": "PHI",
  "opponent": "ORL",
  "date": "2026-04-15",
  "prediction": false,
  "confidence": 51.52,
  "gameId": "401866757",
  "home_odds": 1.84,
  "away_odds": 2.06
}
```

---

## Model versions

| Date range | Version |
|---|---|
| 2025-11-10 – 2025-12-05 | Logistic reg V1 |
| 2025-12-06 – 2025-12-14 | Logistic reg V2.1 |
| 2025-12-15 – 2026-01-08 | Custom NN V1 |
| 2026-01-09 onwards | Logistic reg V2.2 |

Version changes are annotated as dotted vertical lines on the rolling accuracy chart and the cumulative profit chart.

---

## Betting simulation methodology

- **Stake:** flat €1 per game (only games where both home and away odds are available)
- **P&L per game:** `odds × €1 − €1` if correct, else `−€1`
- **ROI %:** `net profit ÷ total staked × 100`
- **Strategies compared:** Model · Against model · Always home · Always favourite · Better record
- **Max drawdown:** largest peak-to-trough loss on the cumulative P&L curve
- **Bucket filter:** cumulative chart can be restricted to any combination of confidence tiers

---

## Season type classification

| Type | Date range |
|---|---|
| Regular Season | Oct – mid Apr (before Play-In) |
| Play-In | ~Apr 14–18 |
| Playoffs | Apr 19 onwards |

| Season | Play-In start | Playoffs start |
|---|---|---|
| 2024–25 | 2025-04-15 | 2025-04-19 |
| 2025–26 | 2026-04-14 | 2026-04-19 |
