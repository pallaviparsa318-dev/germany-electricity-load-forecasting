import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app
from src.data_cleaning import clean_data
from src.features import make_features


def sample(n=240):
    return pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC'),
                         'load_mw': 50000 + np.arange(n) * 2., 'temperature_c': 10.})


def test_cleaning_gap_and_duplicate():
    df = sample(100).drop(index=5)
    df = pd.concat([df, df.iloc[[0]]])
    cleaned = clean_data(df)
    assert len(cleaned) == 100
    assert cleaned['load_mw'].notna().all()


def test_features_no_leakage():
    df = sample()
    feat = make_features(df)
    first = feat.iloc[0]
    original = df.set_index('timestamp')
    assert first['lag_24'] == original.loc[first['timestamp'] - pd.Timedelta(hours=24), 'load_mw']
    assert len(feat) > 0


def test_health_and_validation():
    client = TestClient(app)
    assert client.get('/health').json() == {'status': 'ok'}
    assert client.post('/predict', json={'data_path': ''}).status_code == 422
    assert client.post('/predict', json={'data_path': 'does-not-exist.csv'}).status_code == 400
