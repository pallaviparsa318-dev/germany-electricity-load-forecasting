"""Interactive portfolio dashboard for historical evaluation and 24-hour forecasts.

Run from the project root: python -m streamlit run app/dashboard.py
"""
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st




PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_next_24

ARTIFACTS = Path('artifacts_real')
DATA_PATH = Path('data/real_training_data.csv')

DISPLAY_NAMES = {    'naive_yesterday': 'Yesterday (seasonal naive)',
    'naive_last_week': 'Last week (seasonal naive)',
    'linear_regression': 'Linear Regression',
    'lightgbm': 'LightGBM',
}

st.set_page_config(page_title='Germany Load Forecasting', page_icon='⚡', layout='wide')
st.title('⚡ Germany electricity load forecasting')
st.caption("Historical forecasting demonstration using real German electricity load data from SMARD (2023–2024) and Open-Meteo weather. Not a live operational forecast.")

metrics_path = ARTIFACTS / 'metrics.json'
predictions_path = ARTIFACTS / 'test_predictions.csv'
if not metrics_path.exists() or not predictions_path.exists():
    st.warning('Training artifacts are missing. From the project root run: python -m src.train')
    st.stop()

with metrics_path.open(encoding='utf-8') as file:
    report = json.load(file)
winner = report['winner']
results = report['results']

st.sidebar.header('Dashboard controls')
page = st.sidebar.radio('View', ['Next 24 hours', 'Historical evaluation', 'Model comparison'])
st.sidebar.info('Forecasts use historical load and lagged weather, not future weather forecasts.')

if page == 'Next 24 hours':
    st.subheader('24-hour load forecast')
    st.write(
    "Forecast model: **Linear Regression** "
    "(manually selected for the 24-hour forecast)."
)
    if not DATA_PATH.exists():
        st.error(f'Missing {DATA_PATH}. Please restore the sample data.')
        st.stop()
    try:
        forecast = pd.DataFrame(predict_next_24(str(DATA_PATH), str(ARTIFACTS)))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        st.error(f'Cannot generate forecast: {exc}')
        st.stop()
    forecast['timestamp'] = pd.to_datetime(forecast['timestamp'], utc=True)
    forecast['germany_time'] = forecast['timestamp'].dt.tz_convert('Europe/Berlin')
    start = forecast['germany_time'].iloc[0]
    end = forecast['germany_time'].iloc[-1]
    st.caption(f'Forecast horizon: {start:%d %b %Y %H:%M %Z} to {end:%d %b %Y %H:%M %Z}.')
    a, b, c = st.columns(3)
    a.metric('Mean predicted load', f"{forecast['predicted_load_mw'].mean():,.0f} MW")
    b.metric('Peak predicted load', f"{forecast['predicted_load_mw'].max():,.0f} MW")
    c.metric('Minimum predicted load', f"{forecast['predicted_load_mw'].min():,.0f} MW")
    st.line_chart(forecast.set_index('germany_time')['predicted_load_mw'], y_label='Load (MW)')
    st.dataframe(forecast[['germany_time', 'predicted_load_mw']], use_container_width=True, hide_index=True)
    st.download_button('Download 24-hour forecast (CSV)', forecast.to_csv(index=False).encode('utf-8'), 'next_24_hours.csv', 'text/csv')
    st.info("This is a historical next-day forecast for 1 January 2025, generated after the real SMARD dataset ends on 31 December 2024. It is not a forecast for today's date.")

elif page == 'Historical evaluation':
    st.subheader('Predicted vs actual — held-out test set')
    test = pd.read_csv(predictions_path, parse_dates=['timestamp']).sort_values('timestamp')
    test['timestamp'] = pd.to_datetime(test['timestamp'], utc=True)
    days = st.slider('Days to display', min_value=1, max_value=30, value=7)
    window = test.tail(days * 24).set_index('timestamp')
    scores = results[winner]['test']
    a, b, c = st.columns(3)
    a.metric('Test MAE', f"{scores['mae']:,.1f} MW")
    b.metric('Test RMSE', f"{scores['rmse']:,.1f} MW")
    c.metric('Test MAPE', f"{scores['mape']:.2f}%")
    st.line_chart(window[['actual', 'predicted']], y_label='Load (MW)')
    st.dataframe(window.tail(24), use_container_width=True)
    st.caption('The test set was held out chronologically. This plot evaluates the selected model, not a live forecast.')

else:
    st.subheader('Model comparison')
    split = st.selectbox('Evaluation split', ['test', 'validation'])
    rows = [
        {'Model': DISPLAY_NAMES.get(name, name),
         'MAE (MW)': scores[split]['mae'],
         'RMSE (MW)': scores[split]['rmse'],
         'MAPE (%)': scores[split]['mape']}
        for name, scores in results.items()
    ]
    comparison = pd.DataFrame(rows).sort_values('MAE (MW)').reset_index(drop=True)
    st.dataframe(comparison.style.format({'MAE (MW)': '{:,.2f}', 'RMSE (MW)': '{:,.2f}', 'MAPE (%)': '{:.2f}'}), use_container_width=True, hide_index=True)
    st.bar_chart(comparison.set_index('Model')['MAE (MW)'], y_label='MAE (MW)')
    st.caption(f'Model selection used validation MAE. Selected model: {DISPLAY_NAMES.get(winner, winner)}. Test results are reported only after selection.')
    st.info("All metrics were computed using real German electricity load data from SMARD (2023–2024) and historical Open-Meteo weather. Model selection used validation MAE, and test results were evaluated on a chronologically held-out period.")
