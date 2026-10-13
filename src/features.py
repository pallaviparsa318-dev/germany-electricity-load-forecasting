"""Leakage-safe historical features for direct 24-hour-ahead forecasts."""
import pandas as pd


def holiday_flag(timestamps: pd.Series) -> pd.Series:
    """German nationwide public holidays, with a small offline fallback."""
    try:
        import holidays
        years = sorted(set(timestamps.dt.year))
        de = holidays.Germany(years=years)
        return timestamps.dt.date.map(lambda day: int(day in de)).astype(int)
    except ImportError:
        # Used only for the bundled offline smoke test; install requirements for full calendar.
        return timestamps.dt.strftime('%m-%d').isin(['01-01', '05-01', '10-03', '12-25', '12-26']).astype(int)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Each row predicts load at t using values known at or before t-24h.

    Weather is intentionally represented by its 24h lag: future observed
    weather would leak information; production can substitute forecasts.
    """
    df = frame.copy().sort_values('timestamp').reset_index(drop=True)
    ts = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Europe/Berlin')
    df['hour'] = ts.dt.hour
    df['weekday'] = ts.dt.dayofweek
    df['month'] = ts.dt.month
    df['holiday'] = holiday_flag(ts)
    for lag in [24, 48, 168]:
        df[f'lag_{lag}'] = df['load_mw'].shift(lag)
    df['rolling_24'] = df['load_mw'].shift(24).rolling(24).mean()
    df['rolling_168'] = df['load_mw'].shift(24).rolling(168).mean()
    df['temperature_lag_24'] = df['temperature_c'].shift(24)
    return df.dropna().reset_index(drop=True)

FEATURES = ['hour', 'weekday', 'month', 'holiday', 'lag_24', 'lag_48',
            'lag_168', 'rolling_24', 'rolling_168', 'temperature_lag_24']
