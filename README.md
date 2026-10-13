# Electricity load forecasting for Germany — next 24 hours

A beginner-friendly portfolio MLOps project predicting German electricity demand in MW for the next **24 UTC hours**. Accurate day-ahead forecasts help electricity-market planning, grid balancing and renewable integration. **Bundled data is synthetic, not SMARD observations.** Do not use this prototype for grid operations.

## Architecture

```text
SMARD + Open-Meteo --> CSV --> UTC cleaning --> leakage-safe features
                                         |                 |
                                 chronological split --> baselines / Linear / LightGBM
                                                           |
                                             metrics + MLflow + joblib
                                                           |
                                             FastAPI --> Streamlit test plot
```

## Quick start (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train --data data/sample_data.csv
python -m pytest -q
mlflow ui --backend-store-uri ./mlruns
```

The sample CSV contains 17,520 consecutive UTC hours (2023-01-01 through 2024-12-30), with seasonal, daily, weekly, weather and holiday patterns and deliberately missing values. The train/validation/test split is chronological 70%/15%/15%, with a 191-hour warm-up discarded for lag and rolling features. **Model selection uses validation MAE only.** The test set is reserved for final reporting.

## Download real data (internet required)

```bash
python -m src.download_data --start 2023-01-01 --end 2025-01-01 --output data/real_data.csv
python -m src.train --data data/real_data.csv
```

`download_data.py` requests SMARD's hourly electricity-consumption series (filter 410) and historical hourly temperatures for Berlin, Munich and Hamburg via Open-Meteo archive. The external API calls could not be tested in the offline sandbox; if SMARD changes its API, adjust the endpoint or series filter. Open-Meteo historical weather is **observed**, not a future forecast. The model uses *24-hour-lagged* weather to prevent future-weather leakage. For deployment, replace this with weather forecasts available at prediction time. The nationwide holiday feature uses `holidays.Germany` when installed. The offline fallback includes only five fixed-date holidays and is less complete.

## API, dashboard and Docker

```bash
uvicorn api.main:app --reload
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"data_path":"data/sample_data.csv"}'
streamlit run app/dashboard.py
# In another shell, or instead of running uvicorn locally:
docker build -t germany-load .
docker run --rm -p 8000:8000 germany-load
```

`POST /predict` returns exactly 24 hourly forecasts; its optional `data_path` refers to a CSV accessible to the server. Only use trusted server-side paths in this educational API; restrict or remove this parameter before public deployment. Docker includes the sample training artifacts bundled in this ZIP. `GET /docs` shows the interactive API schema.

## Computed synthetic-data results

All metrics below are from an actual local execution, not invented. Lower is better. MAPE is a percentage.

| Model | Validation MAE (MW) | Test MAE (MW) | Test RMSE (MW) | Test MAPE |
|---|---:|---:|---:|---:|
| Yesterday naive (t-24h) | 2584.71 | 2981.98 | 3949.70 | 5.61% |
| Last-week naive (t-168h) | 1097.23 | 1660.98 | 2124.40 | 3.08% |
| Linear regression | 1040.29 | 1370.74 | 1735.95 | 2.54% |
| **LightGBM** | **901.44** | **1267.97** | **1610.92** | **2.36%** |

LightGBM wins on validation MAE, with the lowest held-out test MAE too. Its nonlinear splits capture interacting seasonal and temperature patterns. However, these synthetic patterns may be easier than real German grid demand. Weekly naive is a strong, interpretable reference. The chart at `artifacts/forecast_vs_actual.png` displays the first test week; the dashboard uses the saved test predictions.

## MLflow and CI

With all requirements installed, training records a separate MLflow run for each model and baseline, including parameters, validation/test metrics, and serialized regression models. Training also saves `artifacts/metrics.json` and `joblib` models so predictions work without the tracking server. In the offline build environment MLflow could not be installed, so its integration was not exercised; local training and artifact saving were verified. CI installs dependencies, runs Ruff's critical-error rules and pytest.

## Limitations and next steps

- No real-data benchmark until SMARD download and training are run locally.
- The cleaning interpolation is appropriate for a learning prototype, but filling long gaps with forward/backward fill can hide outages; production requires quality flags and time-aware imputation.
- Forecasts use lagged weather rather than actual next-day weather forecasts.
- Holidays include nationwide dates, not all state-specific variations.
- Backtesting uses one chronological holdout, not rolling-origin cross-validation.
- Add drift monitoring, scheduled retraining, richer weather forecasts, calendar effects, quantile uncertainty intervals and eventually compare deep-learning models (LSTM/TFT) against these strong baselines.
- No authentication, monitoring or production-grade path restrictions in the demo API.

## What I learned / design decisions

1. Keep timestamps in UTC to avoid duplicate/missing local daylight-saving hours; convert to Europe/Berlin only for calendar features.
2. Predict a 24-hour horizon without consulting future observed loads: every lag is at least 24 hours old.
3. Split by time to prevent training on the future, and choose the winner on validation, not test.
4. Compare complex models with cheap seasonal baselines before claiming improvement.
5. Keep training, evaluation, API and dashboard small and separate for easy explanation in interviews.

## Verification notes

The sample training pipeline ran successfully, 24 forecast rows were produced, and **3 tests passed** in the sandbox. The sandbox ran Python 3.13 and available packages rather than the requested pinned Python 3.11 environment; CI is configured for Python 3.11. Network downloads, MLflow, Streamlit and Ruff require a normal internet-connected installation to validate end to end.

## Upgraded interactive dashboard

From the project root, run `python -m streamlit run app/dashboard.py`.
Use the sidebar to switch between **Next 24 hours** (forecast after the last
available sample timestamp), **Historical evaluation** (held-out actual versus
predicted, adjustable window), and **Model comparison** (validation or test
MAE/RMSE/MAPE). The 24-hour forecast is based on synthetic historical data,
not a live prediction for today's German grid.
