# NBA Prediction Analytics Dashboard

A Streamlit dashboard for analysing NBA game predictions, model performance,
team stats, upset patterns, and betting simulation — backed by data stored in
Cloudflare R2.

## Tabs

| # | Tab | What it shows |
|---|-----|---------------|
| 1 | **Overview** | KPIs, rolling accuracy, today's predictions, full history table |
| 2 | **Model Performance** | Classification metrics by version, calibration chart, confusion matrix, baseline comparison |
| 3 | **Teams** | Per-team accuracy, P&L, ROI, form guide, H2H breakdown, prediction history with odds |
| 4 | **Upset Analysis** | Favourite vs underdog model accuracy, implied probability scatter, upset KPIs |
| 5 | **Odds & Betting** | Flat-stake P&L simulation, cumulative profit chart, drawdown, monthly summary |

## Sidebar filters

- **Date range** — restricts all tabs to the selected window
- **Model version** — multi-select (Logistic reg V1, V2.1, Custom NN V1, V2.2)
- **Season type** — Regular Season / Play-In / Playoffs

---

## Local development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create a .env file (or fill in .streamlit/secrets.toml)
cp .env.example .env   # then edit values

# 3. Run
streamlit run app.py
```

### Environment variables

| Variable | Description |
|---|---|
| `R2_ENDPOINT` | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | R2 API token access key |
| `R2_SECRET_ACCESS_KEY` | R2 API token secret |
| `R2_BUCKET_NAME` | Bucket name (default: `nbaprediction`) |

For local runs you can put these in a `.env` file (loaded by `python-dotenv`)
or in `.streamlit/secrets.toml`.  
**Never commit real credentials to git.**

---

## Deploying to Streamlit Community Cloud (free)

### Prerequisites
- A GitHub account
- A Streamlit Community Cloud account at [streamlit.io/cloud](https://streamlit.io/cloud)

### Steps

1. **Push this folder to GitHub**

   ```bash
   git init
   git add .
   git commit -m "Initial NBA dashboard"
   gh repo create nba-dashboard --public --source=. --push
   ```

2. **Connect to Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New app**
   - Select your GitHub repo, branch (`main`), and set the main file path to `app.py`

3. **Add secrets**
   - In the app settings on Streamlit Cloud, open **Secrets**
   - Paste the following (replacing placeholder values):

   ```toml
   R2_ENDPOINT = "https://<account-id>.r2.cloudflarestorage.com"
   R2_ACCESS_KEY_ID = "<your-access-key-id>"
   R2_SECRET_ACCESS_KEY = "<your-secret-access-key>"
   R2_BUCKET_NAME = "nbaprediction"
   ```

4. **Deploy** — Streamlit Cloud will install `requirements.txt` automatically
   and launch the app.

### Data refresh

Data is fetched from R2 on every page load and cached for 5 minutes
(`@st.cache_data(ttl=300)`).  To force an immediate refresh, open the
hamburger menu inside the app and click **Clear cache**.

### Keeping `.streamlit/secrets.toml` out of git

Add this to your `.gitignore`:

```
.streamlit/secrets.toml
.env
```

The `config.toml` (dark theme settings) is safe to commit.

---

## Data schema

### `history/prediction_history.json`

Array of resolved prediction records:

```json
{
  "team": "CLE",           // home team
  "opponent": "TOR",       // away team
  "date": "2025-11-13",
  "prediction": true,      // true = model predicts home win
  "winner": false,         // true = home team actually won
  "prediction_correct": false,
  "confidence": 54.42,
  "gameId": "401810080",
  "home_odds": 3.16,       // may be null for early records
  "away_odds": 1.4
}
```

### `current/current_predictions.json`

Array of today's unresolved predictions (no `winner` / `prediction_correct`):

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

Version change lines are rendered as dotted vertical annotations on the Overview
rolling accuracy chart and the Odds & Betting cumulative profit chart.

---

## Betting simulation methodology

- **Stake:** flat €1 per game (only games where both home and away odds are available)
- **P&L per game:** `odds × €1 − €1` if prediction correct, else `−€1`
- **ROI %:** `net profit ÷ total staked × 100`
- **Strategies compared:** Model · Against model · Always home · Always favourite
- **Max drawdown:** largest peak-to-trough loss on the cumulative P&L curve

---

## Season type classification

| Type | Date range |
|---|---|
| Regular Season | Oct – mid Apr (before Play-In) |
| Play-In | ~Apr 14–18 |
| Playoffs | Apr 19 onwards |

Exact start dates per season:

| Season | Play-In start | Playoffs start |
|---|---|---|
| 2024–25 | 2025-04-15 | 2025-04-19 |
| 2025–26 | 2026-04-14 | 2026-04-19 |
