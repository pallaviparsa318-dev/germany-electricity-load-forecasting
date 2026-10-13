
"""Download hourly historical temperatures for German cities."""

from pathlib import Path

import pandas as pd
import requests

CITIES = {
    "berlin": (52.52, 13.41),
    "hamburg": (53.55, 10.00),
    "munich": (48.14, 11.58),
}

URL = "https://archive-api.open-meteo.com/v1/archive"


def download_weather() -> pd.DataFrame:
    """Download city temperatures and average them by UTC hour."""

    temperatures = []

    for city, (latitude, longitude) in CITIES.items():
        print(f"Downloading weather for {city}...")

        response = requests.get(
            URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "hourly": "temperature_2m",
                "timezone": "GMT",
            },
            timeout=120,
        )
        response.raise_for_status()

        hourly = response.json()["hourly"]

        city_data = pd.DataFrame({
            "timestamp": pd.to_datetime(
                hourly["time"], utc=True
            ),
            city: pd.to_numeric(
                hourly["temperature_2m"], errors="coerce"
            ),
        })

        temperatures.append(city_data)

    result = temperatures[0]

    for city_data in temperatures[1:]:
        result = result.merge(
            city_data, on="timestamp", how="outer"
        )

    city_columns = list(CITIES)

    result["temperature_c"] = result[city_columns].mean(
        axis=1
    )

    return result.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    weather = download_weather()

    output = Path("data/weather.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    weather.to_csv(output, index=False)

    print(f"Downloaded {len(weather):,} hourly weather rows")
    print(f"Saved to: {output}")
    print(weather.head())
