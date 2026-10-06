# NBA Prediction Analytics Dashboard

A static dashboard for the [Predicting-Nba](https://github.com/markusmuilu/Predicting-Nba) prediction service.
Three views:

- **2026-27 season**: the production model's daily predictions against results, running accuracy, calibration.
  Empty until the regular season has scored games, and says so.
- **Model comparison**: the production logistic regression, the PyTorch player model and (once it has been run) Jev,
  all on the same 2025-26 test season, with a market benchmark where Pinnacle odds were stored.
- **2025-26 archive**: last season's record of the production model, by version, by team, favourites vs underdogs, flat-stake result.

## Why it is static

The previous version was a Streamlit app on Streamlit Community Cloud. Free apps there go to sleep after a period
without visitors, and the portfolio site embeds the app, so visitors often saw a sleep screen instead of data.
A page made of plain files on GitHub Pages has no process that can sleep. A GitHub Action rebuilds the data once a day.

What this gives up: the sidebar filters of the old app (the page filters nothing server-side), and the data is up to a
day old. Both are fine for a model that predicts once a day.

## How it works

```
R2 bucket ──(read only, once a day)──> build_site.py ──> site/data.json
                                                         site/index.html  (static, Plotly drawn in the browser)
site_data/model_comparison.json  (snapshot of the Predicting-Nba experiment)
```

- `build_site.py` reads `history/prediction_history.json` and `current/current_predictions.json` from R2, scores them, writes `site/data.json`.
- `site/index.html` is hand-written HTML and JavaScript. Plotly is vendored in `site/vendor/`, so there is no third-party script to break.
- `site_data/model_comparison.json` is a copy of `research/results/metrics.json` from the Predicting-Nba `player-model` branch.
  Re-copy it when that experiment is re-run.

## Run locally

```bash
pip install -r requirements-build.txt
python build_site.py --history path/to/prediction_history.json   # offline, from a local copy
# or, with R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME set:
python build_site.py
cd site && python -m http.server 8000      # open http://localhost:8000
```

`data.json` is loaded with `fetch`, so open the page through a local server, not as a `file://` URL.

## Deploy (GitHub Pages)

1. Merge the `static-site` branch into `master`.
2. Repository **Settings > Pages > Build and deployment > Source = GitHub Actions**.
3. **Settings > Secrets and variables > Actions**, add four repository secrets:
   `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` (use an R2 API token limited to read access on this bucket).
4. **Actions > Build and deploy dashboard > Run workflow** once. The page appears at `https://markusmuilu.github.io/nba-dashboard/`.
5. It then rebuilds every day at 10:30 UTC.

Known limit: GitHub disables scheduled workflows in a public repository after 60 days without repository activity.
If the page stops updating, re-enable the workflow under the Actions tab (or push any commit).

## Changing the portfolio embed

The portfolio (`src/pages/NbaPrediction.js`) embeds `https://nba-ml-dashboard.streamlit.app/?embed=true` in an iframe and links to
`https://nba-ml-dashboard.streamlit.app/` in two more places. After the Pages site is live, those URLs would change to
`https://markusmuilu.github.io/nba-dashboard/` (the `?embed=true` suffix is Streamlit-specific and can go), and so would the
browser-bar label text `nba-ml-dashboard.streamlit.app` in the same file. GitHub Pages sends no frame-blocking headers, so the
iframe works. Nothing in the portfolio was changed as part of this work.

## Legacy Streamlit app

`app.py`, `tabs/`, `ui/`, `data/` and `config/` are the earlier Streamlit version. They still run locally
(`pip install -r requirements.txt && streamlit run app.py`) but are no longer the deployed dashboard.
Its README, including the data schema of the two R2 files, is kept in [docs/streamlit-legacy.md](docs/streamlit-legacy.md).
