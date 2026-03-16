import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

from air_quality_rnn.utils import scale_targets
from air_quality_rnn.preprocessing import preprocess_splits, chronological_split


def create_sliding_windows(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    window_size: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a time series DataFrame into sliding windows.

    Returns
    -------
    X : np.ndarray
        Shape (samples, window_size, n_features)
    y : np.ndarray
        Shape (samples, horizon)
    """

    # Validate inputs
    if window_size <= 0 or horizon <= 0:
        raise ValueError("window_size and horizon must be positive integers")
    if feature_columns is None or len(feature_columns) == 0:
        raise ValueError("feature_columns must be a non-empty list of column names")
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in DataFrame")
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"feature_column '{col}' not found in DataFrame")
    if window_size + horizon > len(df):
        raise ValueError("window_size + horizon must be less than or equal to the length of the DataFrame")
    
    X, y = [], []
    for i in range(len(df) - window_size - horizon + 1):
        X.append(df[feature_columns].iloc[i:i + window_size].values)
        y.append(df[target_column].iloc[i + window_size:i + window_size + horizon].values)

    return np.array(X), np.array(y)


def create_datasets(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    horizon: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    StandardScaler
]:
    """Prepare train/validation/test datasets for forecasting."""

    # Split the data into train/validation/test sets
    train_df, val_df, test_df = chronological_split(df, train_size, val_size)

    # Preprocess the splits (impute missing values and scale features)
    train_df, val_df, test_df, feature_scaler, target_scaler = preprocess_splits(
        train_df,
        val_df,
        test_df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    # Create sliding windows for each split
    X_train, y_train = create_sliding_windows(
        train_df,
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=window_size,
        horizon=horizon,
    )
    X_val, y_val = create_sliding_windows(
        val_df,
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=window_size,
        horizon=horizon,
    )
    X_test, y_test = create_sliding_windows(
        test_df,
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=window_size,
        horizon=horizon,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler