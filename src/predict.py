"""Predict a complete next-day horizon using historical observations."""
from pathlib import Path
import joblib
import pandas as pd
from src.data_cleaning import clean_data
from src.features import FEATURES, holiday_flag


def predict_next_24(data_path: str = 'data/sample_data.csv', artifact_dir: str = 'artifacts') -> list[dict]:
    """Forecast next 24 UTC hours, using lagged load and lagged temperature.

    All 24 horizon features use only observed history. Rolling features are
    anchored at the forecast origin; no future actual loads are consulted.
    """
    df = clean_data(pd.read_csv(data_path)).set_index('timestamp')
    if len(df) < 192:
        raise ValueError('At least 192 historical hours are required')
    meta = joblib.load(Path(artifact_dir) / 'metadata.joblib')
    winner = 'linear_regression'
    model = None if winner.startswith('naive_') else joblib.load(Path(artifact_dir) / f'{winner}.joblib')
    last = df.index[-1]
    horizon = pd.date_range(last + pd.Timedelta(hours=1), periods=24, freq='h', tz='UTC')
    rows = []
    for target in horizon:
        local = target.tz_convert('Europe/Berlin')
        values = {'hour': local.hour, 'weekday': local.dayofweek, 'month': local.month,
                  'holiday': int(holiday_flag(pd.Series([local])).iloc[0]),
                  'lag_24': df.loc[target - pd.Timedelta(hours=24), 'load_mw'],
                  'lag_48': df.loc[target - pd.Timedelta(hours=48), 'load_mw'],
                  'lag_168': df.loc[target - pd.Timedelta(hours=168), 'load_mw'],
                  'rolling_24': df.loc[target - pd.Timedelta(hours=47):target - pd.Timedelta(hours=24), 'load_mw'].mean(),
                  'rolling_168': df.loc[target - pd.Timedelta(hours=191):target - pd.Timedelta(hours=24), 'load_mw'].mean(),
                  'temperature_lag_24': df.loc[target - pd.Timedelta(hours=24), 'temperature_c']}
        if winner == 'naive_yesterday':
            prediction = values['lag_24']
        elif winner == 'naive_last_week':
            prediction = values['lag_168']
        else:
            prediction = model.predict(pd.DataFrame([values])[FEATURES])[0]
        rows.append({'timestamp': target.isoformat(), 'predicted_load_mw': round(float(prediction), 2)})
    return rows
