
"""Train electricity load models using chronological data splits."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_cleaning import clean_data
from src.features import FEATURES, make_features
from src.evaluate import metrics, plot_forecast


def train(
    data_path: str,
    output_dir: str = "artifacts_real",
) -> dict:
    """Train models and select the winner using validation MAE."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare data
    raw_data = pd.read_csv(data_path)
    clean = clean_data(raw_data)
    df = make_features(clean)

    # 2. Chronological split: 70% / 15% / 15%
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")

    # 3. Train Linear Regression
    model = LinearRegression()

    model.fit(
        train_df[FEATURES],
        train_df["load_mw"],
    )

    joblib.dump(
        model,
        output / "linear_regression.joblib",
    )

    # 4. Generate predictions for each model
    predictions = {
        "naive_yesterday": (
            val_df["lag_24"].to_numpy(),
            test_df["lag_24"].to_numpy(),
        ),
        "naive_last_week": (
            val_df["lag_168"].to_numpy(),
            test_df["lag_168"].to_numpy(),
        ),
        "linear_regression": (
            model.predict(val_df[FEATURES]),
            model.predict(test_df[FEATURES]),
        ),
    }

    # 5. Evaluate each model
    results = {}

    for name, (val_pred, test_pred) in predictions.items():
        results[name] = {
            "validation": metrics(
                val_df["load_mw"], val_pred
            ),
            "test": metrics(
                test_df["load_mw"], test_pred
            ),
        }

    # 6. Select using validation MAE only
    winner = min(
        results,
        key=lambda name: results[name]["validation"]["mae"],
    )

    print(f"Selected model: {winner}")

    # 7. Save metadata
    joblib.dump(
        {
            "winner": winner,
            "features": FEATURES,
        },
        output / "metadata.joblib",
    )

    # 8. Save test predictions
    test_pred = predictions[winner][1]

    test_predictions = pd.DataFrame({
        "timestamp": test_df["timestamp"],
        "actual": test_df["load_mw"],
        "predicted": test_pred,
    })

    test_predictions.to_csv(
        output / "test_predictions.csv",
        index=False,
    )

    plot_forecast(
        test_df["timestamp"].to_numpy(),
        test_df["load_mw"].to_numpy(),
        test_pred,
        str(output / "forecast_vs_actual.png"),
    )

    report = {
        "winner": winner,
        "results": results,
    }

    (output / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # 9. Optional MLflow experiment tracking
    try:
        import mlflow

        mlflow.set_tracking_uri(
            "file:" + str(Path("mlruns").resolve())
        )

        mlflow.set_experiment(
            "germany-real-electricity-load"
        )

        for name in predictions:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("dataset", "SMARD_2023_2024")
                mlflow.log_param("model_name", name)

                for split in ["validation", "test"]:
                    mlflow.log_metrics({
                        f"{split}_{key}": value
                        for key, value in results[name][split].items()
                    })

                if name == "linear_regression":
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                    )

    except ImportError:
        print("MLflow unavailable; local metrics were saved.")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/real_training_data.csv",
    )

    parser.add_argument(
        "--output",
        default="artifacts_real",
    )

    args = parser.parse_args()

    report = train(args.data, args.output)

    print(json.dumps(report, indent=2))
