# Germany Electricity Load Forecasting

**[Open the live Streamlit dashboard](https://germany-electricity-load-forecasting-xlsylopzbnksrezhsyskzn.streamlit.app/)** — Explore the historical 24-hour forecast, evaluation, and model comparison. The demo is not a live operational forecast.

End-to-end MLOps portfolio project for **hourly German electricity-load forecasting**, using historical [SMARD](https://www.smard.de/) electricity data and [Open-Meteo](https://open-meteo.com/) weather data.

**Status:** Working historical demonstration, **not a live operational forecast**. The example 24-hour forecast covers **1 January 2025**, immediately after the dataset ends.

## Project at a glance

- **Data:** 17,544 hourly observations covering 2023–2024, with no missing values in the merged training dataset.
- **Weather:** Historical hourly temperatures for Berlin, Hamburg, and Munich; their simple average serves as a rough national proxy.
- **Models:** Linear Regression, yesterday's-load baseline (24-hour lag), and last-week's-load baseline (168-hour lag).
- **Evaluation:** Chronological 70% / 15% / 15% train, validation, and test split after feature generation. Model selection uses **validation MAE**, never test MAE.
- **Tools:** Python, pandas, scikit-learn, MLflow, Streamlit, FastAPI, Docker, GitHub Actions.

## Pipeline

```text
SMARD load ───────────────┐
                          ├─> UTC merge and cleaning ─> lag/calendar/weather features
Open-Meteo temperatures ──┘                                    │
                                                chronological train/val/test
                                                               │
                                          Linear Regression + seasonal baselines
                                                               │
                                               MAE / RMSE / MAPE + MLflow
                                                               │
                                          saved artifacts ─> Streamlit dashboard
```

## Results on real data

These results were obtained by running the real-data pipeline locally on the 2023–2024 dataset.

| Model | Validation MAE (MW) | Test MAE (MW) | Test RMSE (MW) | Test MAPE |
|---|---:|---:|---:|---:|
| Yesterday baseline | 3,748.08 | 3,773.28 | 5,665.14 | 7.03% |
| **Last-week baseline** | **1,549.03** | 2,901.28 | 4,190.11 | 5.58% |
| Linear Regression | 1,718.04 | **2,242.16** | **2,971.89** | **4.30%** |

The **last-week baseline was selected** because it achieved the lowest validation MAE. **Linear Regression achieved the lowest test error**. This difference is a useful reminder that performance can vary across historical periods; rolling-origin backtesting is a planned improvement.

The historical-evaluation dashboard displays the validation-selected last-week baseline. The next-24-hour demonstration currently uses **Linear Regression explicitly**, rather than the metadata-selected baseline.

## Reproduce locally (Python 3.11)

Run commands from the repository root. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows dependency note:** Some installations of LightGBM can fail due to native DLL dependencies. The current `src/train.py` trains Linear Regression and seasonal baselines without importing LightGBM, although the dependency may still appear in `requirements.txt`.

### Download and prepare the data

```powershell
python -m src.download_data
python -m src.download_weather
python -m src.merge_real_data
```

This produces `data/real_training_data.csv` with `timestamp`, `load_mw`, and `temperature_c`. The downloader currently uses fixed dates for **2023-01-01 through 2024-12-31**.

### Train and evaluate

```powershell
python -m src.train --data data/real_training_data.csv --output artifacts_real
```

Outputs include `linear_regression.joblib`, `metadata.joblib`, `metrics.json`, `test_predictions.csv`, and `forecast_vs_actual.png` in `artifacts_real/`.

### View MLflow experiments

```powershell
python -m mlflow ui --backend-store-uri ./mlruns
```

Open http://localhost:5000. The real-data experiment is named `germany-real-electricity-load`.

### Run the Streamlit dashboard

```powershell
python -m streamlit run app/dashboard.py
```

The dashboard has three views:

1. **Next 24 hours:** Historical forecast for 1 January 2025, with a CSV download.
2. **Historical evaluation:** Actual versus predicted values on the held-out test set.
3. **Model comparison:** Validation and test MAE, RMSE, and MAPE for all three models.

The dashboard should be configured to read `data/real_training_data.csv` and `artifacts_real/`. The repository does not include generated datasets or trained artifacts, so run the preparation and training commands first.

## Repository layout

```text
.github/workflows/   CI configuration
api/                 FastAPI service
app/                 Streamlit dashboard
notebooks/           Exploration notebook
src/                 Download, cleaning, features, training, prediction
tests/               Automated tests
Dockerfile           Container configuration
requirements.txt     Dependencies
```

**API/Docker caveat:** The API still defaults to older sample-data paths and the Docker setup has not yet been fully migrated to the real-data pipeline. Do not assume the API or container reproduces the real-data dashboard without further configuration.

## Methodology and limitations

- UTC timestamps avoid daylight-saving ambiguity; local German time is used for calendar features.
- Forecast features use historical load lags and lagged observed temperatures, not future observed load.
- The three-city temperature mean is an approximation, not a population-weighted national weather index.
- Historical lagged weather is **not** a genuine next-day weather forecast.
- The 24-hour output is a historical demonstration, not a current-date forecast.
- One chronological holdout does not establish reliability across all seasons; rolling-origin backtesting is needed.
- The API requires additional validation and security hardening before public deployment.
- Datasets, model artifacts, and local MLflow logs are intentionally excluded from Git.

## Next improvements

Rolling-origin backtesting, consistent model selection across the dashboard and API, stronger weather features, CI checks for the real-data workflow, and deployment with reproducible model artifacts.

## Data acknowledgments

- [SMARD](https://www.smard.de/) — German electricity-load observations
- [Open-Meteo](https://open-meteo.com/) — Historical temperature data
