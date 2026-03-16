import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

from air_quality_rnn.utils import scale_targets
from air_quality_rnn.preprocessing import preprocess_splits, chronological_split


def create_sliding_windows(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    feature_columns: List[str],
    window_size: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time series data into sliding windows.

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
    
    if X_df.empty or y_series.empty:
        raise ValueError("X_df and y_series must not be empty")
    
    for col in feature_columns:
        if col not in X_df.columns:
            raise ValueError(f"feature_column '{col}' not found in X_df")
    
    if len(X_df) != len(y_series):
        raise ValueError("X_df and y_series must have the same length")
    
    if window_size + horizon > len(X_df):
        raise ValueError("window_size + horizon must be less than or equal to the length of X_df")
    
    X, y = [], []

    for i in range(len(X_df) - window_size - horizon + 1):
        X.append(X_df[feature_columns].iloc[i:i + window_size].values)
        y.append(y_series.iloc[i + window_size:i + window_size + horizon].values)

    return np.array(X), np.array(y)


def create_datasets(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    horizon: int,
    scale_target: bool = True,
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
    train_imputed, val_imputed, test_imputed, train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler = preprocess_splits(
        train_df,
        val_df,
        test_df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    # Create sliding windows for each split
    X_train, y_train = create_sliding_windows(
        X_df=train_scaled,
        y_series=train_imputed[target_column],
        feature_columns=feature_columns,
        window_size=window_size,
        horizon=horizon,
    )
    X_val, y_val = create_sliding_windows(
        X_df=val_scaled,
        y_series=val_imputed[target_column],
        feature_columns=feature_columns,
        window_size=window_size,
        horizon=horizon,
    )
    X_test, y_test = create_sliding_windows(
        X_df=test_scaled,
        y_series=test_imputed[target_column],
        feature_columns=feature_columns,
        window_size=window_size,
        horizon=horizon,
    )

    # Scale target values
    if scale_target:
        y_train = scale_targets(y_train, target_scaler)
        y_val = scale_targets(y_val, target_scaler)
        y_test = scale_targets(y_test, target_scaler)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler