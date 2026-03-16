import pandas as pd
import json
from pathlib import Path

from air_quality_forecasting.config import load_config
from air_quality_forecasting.datasets import load_data, create_datasets
from air_quality_forecasting.baselines import naive_forecast, seasonal_naive_forecast 
from air_quality_forecasting.evaluate import evaluate_forecast
from air_quality_forecasting.utils import convert_numpy, round_metrics, inverse_scale_targets


def main() -> None:
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
    df = load_data(data_path)

    # Create datasets
    _, _, _, _, X_test, y_test, _, target_scaler = create_datasets(
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

    # Inverse scale predictions and true values for evaluation
    y_test_true = inverse_scale_targets(y_test, target_scaler)
    naive_predictions = inverse_scale_targets(naive_predictions, target_scaler)
    seasonal_naive_predictions = inverse_scale_targets(seasonal_naive_predictions, target_scaler)

    # Evaluate baselines and print results
    naive_metrics = evaluate_forecast(y_test_true, naive_predictions)
    seasonal_naive_metrics = evaluate_forecast(y_test_true, seasonal_naive_predictions)

    print("Naive baseline:")
    print(round_metrics(naive_metrics))

    print("\nSeasonal naive baseline:")
    print(round_metrics(seasonal_naive_metrics))

    # Save results
    Path("reports").mkdir(exist_ok=True)

    results = {
        "naive": convert_numpy(naive_metrics),
        "seasonal_naive": convert_numpy(seasonal_naive_metrics),
    }

    with open("reports/baselines_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
