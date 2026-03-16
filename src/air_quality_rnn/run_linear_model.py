import pandas as pd
import json
from pathlib import Path
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from air_quality_rnn.datasets import create_datasets
from air_quality_rnn.evaluate import evaluate_forecast
from air_quality_rnn.config import load_config
from air_quality_rnn.utils import flatten_windows, convert_numpy, round_metrics, inverse_scale_targets
from air_quality_rnn.visualization import plot_forecast_example


def main():
    """Train and evaluate a Ridge regression forecasting model."""

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
    X_train, y_train, X_val, y_val, X_test, y_test, _feature_scaler, target_scaler = create_datasets(
        df=df,
        train_size=train_size,
        val_size=val_size,
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=window_size,
        horizon=horizon
    )

    # Flatten windows 
    X_train_flat = flatten_windows(X_train)
    X_val_flat = flatten_windows(X_val)
    X_test_flat = flatten_windows(X_test)

    # 
    y_val_true = inverse_scale_targets(y_val, target_scaler)
    y_test_true = inverse_scale_targets(y_test, target_scaler)

    # Train Ridge regression model with different alpha values
    best_alpha = None
    best_metrics = None
    best_rmse = float("inf")

    for alpha in [0.01, 0.1, 1, 10, 100]:
        model = MultiOutputRegressor(Ridge(alpha=alpha))
        model.fit(X_train_flat, y_train)

        y_val_pred = model.predict(X_val_flat)
        y_val_pred_inv = inverse_scale_targets(y_val_pred, target_scaler)  # type: ignore

        metrics = evaluate_forecast(y_val_true, y_val_pred_inv)  

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_alpha = alpha
            best_metrics = metrics
    
    if best_alpha is None or best_metrics is None:
        raise RuntimeError("No best model was selected during alpha tuning.")

    print(f"Best alpha: {best_alpha}")
    print("Validation metrics for best model:")
    print(round_metrics(best_metrics)) 

    # Train best model on combined train and validation sets
    X_train_val_flat = np.vstack([X_train_flat, X_val_flat])
    y_train_val = np.vstack([y_train, y_val])

    final_model = MultiOutputRegressor(Ridge(alpha=best_alpha))  
    final_model.fit(X_train_val_flat, y_train_val)  

    # Predict on test set and inverse scale
    y_test_pred = final_model.predict(X_test_flat)
    y_test_pred_inv = inverse_scale_targets(y_test_pred, target_scaler)  # type: ignore
    
    # Evaluate model and print results
    test_metrics = evaluate_forecast(y_test_true, y_test_pred_inv)
    print("Ridge regression test metrics:")
    print(round_metrics(test_metrics))

    # Save results 
    results = {
        "best_alpha": best_alpha,
        "validation_metrics": convert_numpy(round_metrics(best_metrics)),
        "test_metrics": convert_numpy(round_metrics(test_metrics)),
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    with open("reports/linear_model.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    # Generate and save example forecast plot
    sample_idx = len(y_test) // 2

    plot_forecast_example(
        y_true=y_test_true,
        y_pred=y_test_pred_inv,
        sample_idx=sample_idx,
        save_path="reports/figures/forecast_example.png",
    )


if __name__ == "__main__":
    main()
