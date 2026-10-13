
"""Merge SMARD electricity load with Open-Meteo weather."""

from pathlib import Path

import pandas as pd


def main() -> None:
    """Create the combined dataset used for model training."""

    load = pd.read_csv(
        "data/real_data.csv", parse_dates=["timestamp"]
    )

    weather = pd.read_csv(
        "data/weather.csv", parse_dates=["timestamp"]
    )

    combined = load.merge(
        weather[["timestamp", "temperature_c"]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    combined = combined.sort_values("timestamp")

    print(f"Total rows: {len(combined):,}")
    print("Missing values:")
    print(combined.isna().sum())

    if combined["temperature_c"].isna().any():
        raise ValueError(
            "Missing weather observations. Check date coverage."
        )

    output = Path("data/real_training_data.csv")
    combined.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(combined.head())


if __name__ == "__main__":
    main()
