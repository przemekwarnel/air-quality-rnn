import pandas as pd
import json
import numpy as np
from pathlib import Path

from air_quality_rnn.config import load_config
from air_quality_rnn.datasets import create_datasets
from air_quality_rnn.baselines import naive_forecast, seasonal_naive_forecast 
from air_quality_rnn.evaluate import evaluate_forecast
from air_quality_rnn.utils import convert_numpy, round_metrics


def main():
    """Run baseline forecasts and evaluate them. Save results to reports/baselines.json."""

    # Load config and config parameters
    config = load_config("configs/base.yaml") 

    data_path = config["data"]["path"]

    train_size = config["split"]["train_size"]
    val_size = config["split"]["val_size"]

    feature_columns = config["features"]["feature_columns"]
    target_column = config["features"]["target_column"]

    window_size = config["forecast"]["window_size"]
    horizon = config["forecast"]["horizon"]

    # Load dataset 
    df = pd.read_csv(
        data_path,
        usecols=lambda column: column not in ['No', 'wd', 'station']
    )

    df['Date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('Date', inplace=True)
    df.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)

    # Create datasets
    X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler = create_datasets(
        df=df,
        train_size=train_size,
        val_size=val_size,
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=window_size,
        horizon=horizon
    )

    # Run naive and seasonal naive forecast baseline
    naive_predictions = naive_forecast(
        X_test,
        feature_columns,
        target_column,
        horizon
    )

    seasonal_naive_predictions = seasonal_naive_forecast(
        X_test,
        feature_columns,
        target_column,
        horizon
    )

    # Evaluate baselines and print results
    naive_metrics = evaluate_forecast(y_test, naive_predictions)
    seasonal_naive_metrics = evaluate_forecast(y_test, seasonal_naive_predictions)

    print("Naive baseline:")
    print(round_metrics(naive_metrics))

    print("\nSeasonal naive baseline:")
    print(round_metrics(seasonal_naive_metrics))

    # Save results
    Path("reports").mkdir(exist_ok=True)

    results = {
        "naive": convert_numpy(round_metrics(naive_metrics)),
        "seasonal_naive": convert_numpy(round_metrics(seasonal_naive_metrics)),
    }

    with open("reports/baselines.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
