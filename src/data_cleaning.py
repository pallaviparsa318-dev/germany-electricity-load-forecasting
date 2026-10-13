"""UTC-first hourly cleaning: DST is a display concern, not a missing UTC hour."""
import numpy as np
import pandas as pd


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex UTC hours, interpolate short gaps, and clip implausible spikes."""
    df = frame.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
    df = df.asfreq('h')
    df['load_mw'] = pd.to_numeric(df['load_mw'], errors='coerce')
    df.loc[df['load_mw'] <= 0, 'load_mw'] = np.nan
    median = df['load_mw'].median()
    mad = (df['load_mw'] - median).abs().median()
    if mad > 0:
        df['load_mw'] = df['load_mw'].clip(median - 8 * mad, median + 8 * mad)
    df['load_mw'] = df['load_mw'].interpolate(limit=6, limit_direction='both')
    df['load_mw'] = df['load_mw'].ffill().bfill()
    for column in ['temperature_c']:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors='coerce').interpolate(limit_direction='both').ffill().bfill()
    return df.reset_index()
