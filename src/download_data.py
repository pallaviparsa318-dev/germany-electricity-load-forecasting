
"""Download real hourly German electricity load from SMARD."""

from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://www.smard.de/app/chart_data"
FILTER = "410"  # Total German grid load
REGION = "DE"
RESOLUTION = "hour"

START_DATE = "2023-01-01"
END_DATE = "2024-12-31"


def download_smard() -> pd.DataFrame:
    """Fetch historical load and return hourly UTC observations."""

    index_url = (
        f"{BASE_URL}/{FILTER}/{REGION}/"
        f"index_{RESOLUTION}.json"
    )

    response = requests.get(index_url, timeout=30)
    response.raise_for_status()

    timestamps = response.json()["timestamps"]

    # Index timestamps identify available data windows.
    # Include neighboring windows to cover date boundaries.
    start_ms = pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000
    end_ms = (
        pd.Timestamp(END_DATE, tz="UTC")
        + pd.Timedelta(days=1)
    ).timestamp() * 1000

    selected = [
        ts for ts in timestamps
        if start_ms - 8 * 86400 * 1000 <= ts < end_ms
    ]

    records = []

    for ts in selected:
        url = (
            f"{BASE_URL}/{FILTER}/{REGION}/"
            f"{FILTER}_{REGION}_{RESOLUTION}_{ts}.json"
        )

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        records.extend(response.json()["series"])

    if not records:
        raise RuntimeError("No SMARD records were downloaded.")

    df = pd.DataFrame(records, columns=["timestamp", "load_mw"])

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], unit="ms", utc=True
    )

    df["load_mw"] = pd.to_numeric(
        df["load_mw"], errors="coerce"
    )

    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp")

    df = df[
        (df["timestamp"] >= pd.Timestamp(START_DATE, tz="UTC"))
        & (
            df["timestamp"]
            < pd.Timestamp(END_DATE, tz="UTC")
            + pd.Timedelta(days=1)
        )
    ]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    data = download_smard()

    output = Path("data/real_data.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(output, index=False)

    print(f"Downloaded {len(data):,} hourly rows")
    print(f"Missing load values: {data['load_mw'].isna().sum():,}")
    print(f"Saved to: {output}")
    print(data.head())
