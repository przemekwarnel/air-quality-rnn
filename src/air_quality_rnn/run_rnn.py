import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from air_quality_rnn.config import load_config
from air_quality_rnn.datasets import load_data, create_datasets
from air_quality_rnn.models import build_lstm_model
from air_quality_rnn.evaluate import evaluate_forecast
from air_quality_rnn.utils import inverse_scale_targets, convert_numpy, round_metrics
from air_quality_rnn.visualization import plot_forecast_example, plot_training_history

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None: 
    """Train and evaluate an LSTM forecasting model."""

    # Load config and config parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config) 

    seed = config["training"].get("seed", 42)
    set_seed(seed)

    data_path = config["data"]["path"]

    train_size = config["split"]["train_size"]
    val_size = config["split"]["val_size"]

    feature_columns = config["features"]["feature_columns"]
    target_column = config["features"]["target_column"]

    window_size = config["forecast"]["window_size"]
    horizon = config["forecast"]["horizon"]
    
    lstm_units = config["model"]["lstm_units"]
    learning_rate = config["model"]["learning_rate"]
    dropout = config["model"].get("dropout", 0.0)
    l2_strength = config["model"].get("l2_strength", 0.0)

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    patience = config["training"].get("patience", 5)

    # Load dataset 
    df = load_data(data_path)

    # Create datasets
    X_train, y_train, X_val, y_val, X_test, y_test, _, target_scaler = create_datasets(
        df=df,
        train_size=train_size,
        val_size=val_size,
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=window_size,
        horizon=horizon
    )

    # Build model
    model = build_lstm_model(
        window_size=window_size,
        n_features=X_train.shape[2],
        horizon=horizon,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        dropout=dropout,
        l2_strength=l2_strength,
    )

    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Predict on val and test set 
    y_val_pred = model.predict(X_val, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)

    # Inverse scale predictions and true values for evaluation
    y_val_pred_inv = inverse_scale_targets(y_val_pred, target_scaler)
    y_test_pred_inv = inverse_scale_targets(y_test_pred, target_scaler)
    y_val_true = inverse_scale_targets(y_val, target_scaler)
    y_test_true = inverse_scale_targets(y_test, target_scaler)


    # Evaluate model and print results
    val_metrics = evaluate_forecast(y_val_true, y_val_pred_inv)
    test_metrics = evaluate_forecast(y_test_true, y_test_pred_inv)
    
    print("\nLSTM model validation metrics:")
    print(round_metrics(val_metrics))

    print("\nLSTM model test metrics:")
    print(round_metrics(test_metrics))

    # Save results 
    results = {
        "model": "lstm",
        "params": {
            "seed": seed,
            "window_size": window_size,
            "horizon": horizon,
            "lstm_units": lstm_units,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "l2_strength": l2_strength,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "train_size": train_size,
            "val_size": val_size,
        },
        "validation_metrics": convert_numpy(val_metrics),
        "test_metrics": convert_numpy(test_metrics),
    }

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/experiments").mkdir(parents=True, exist_ok=True)

    experiment_name = Path(args.config).stem
    output_path = Path(f"reports/experiments/{experiment_name}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    # Generate and save example forecast plot
    sample_idx = len(y_test) // 2

    plot_forecast_example(
        y_test_true,
        y_test_pred_inv,
        sample_idx=sample_idx,
        save_path=f"reports/figures/{experiment_name}_forecast.png",
        )
    
    # Generate and save training history plot 
    plot_training_history(
        history,
        save_path=f"reports/figures/{experiment_name}_training.png",
    )


if __name__ == "__main__":
    main()