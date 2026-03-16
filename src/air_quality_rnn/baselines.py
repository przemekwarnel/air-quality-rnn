import numpy as np
from typing import List


def naive_forecast(
    X: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    horizon: int,               
) -> np.ndarray:
    """Naive forecasting baseline using the last observed target value."""

    # Find index of the target column
    target_idx = feature_columns.index(target_column)

    # Extract target history
    target_history = X[:, :, target_idx]

    # Last observed value in each window
    last_value = target_history[:, -1]  # shape (num_samples,)

    # Repeat last value across forecast horizon
    predictions = np.repeat(last_value[:, None], horizon, axis=1)  # shape (num_samples, horizon)

    return predictions


def seasonal_naive_forecast(
    X: np.ndarray,
    feature_columns: list[str],
    target_column: str,
    horizon: int,
) -> np.ndarray:
    """Seasonal naive forecasting baseline using the last observed seasonal pattern."""

    if horizon > X.shape[1]:
        raise ValueError("horizon cannot be greater than window size")
    
    # Find index of the target column
    target_idx = feature_columns.index(target_column)

    # Extract target history
    target_history = X[:, :, target_idx]

    # Use the last observed seasonal pattern
    predictions = target_history[:, -horizon:]  # shape (num_samples, horizon)

    return predictions