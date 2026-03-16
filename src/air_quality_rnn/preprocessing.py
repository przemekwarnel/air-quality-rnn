import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def chronological_split(
    df: pd.DataFrame,
    train_size=0.7,
    val_size=0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series DataFrame into train/validation/test sets chronologically."""

    # Validate inputs
    if not (0 < train_size < 1) or not (0 < val_size < 1):
        raise ValueError("train_size and val_size must be between 0 and 1")
    if train_size + val_size >= 1.0:
        raise ValueError("train_size and val_size must sum to less than 1.0")
    
    total_size = len(df)
    train_end = int(total_size * train_size)
    val_end = int(total_size * (train_size + val_size))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def preprocess_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,    
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
    StandardScaler,
]:
    """Impute missing values and scale feature columns using statistics fitted on the training split."""

    # Make copies to avoid modifying original DataFrames
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Validate inputs
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Input DataFrames must not be empty")
    if feature_columns is None or len(feature_columns) == 0:
        raise ValueError("feature_columns must be a non-empty list of column names")
    if target_column not in train_df.columns or target_column not in val_df.columns or target_column not in test_df.columns:
        raise ValueError(f"target_column '{target_column}' must be present in all DataFrames")
    for col in feature_columns:
        if col not in train_df.columns or col not in val_df.columns or col not in test_df.columns:
            raise ValueError(f"feature_column '{col}' must be present in all DataFrames")

    # Fill missing values with forward fill, then backward fill as a fallback
    train_imputed = train_df.ffill().bfill()
    val_imputed = val_df.ffill().bfill()
    test_imputed = test_df.ffill().bfill()

    # Check for remaining missing values
    if train_imputed.isnull().any().any() or val_imputed.isnull().any().any() or test_imputed.isnull().any().any():
        raise ValueError("DataFrames contain missing values after filling. Please check the data.")
    
    # Prepare datasets for scaling 
    train_scaled = train_imputed.copy()
    val_scaled = val_imputed.copy()
    test_scaled = test_imputed.copy()

    # Instantiate scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit scalers on training data only
    feature_scaler.fit(train_scaled[feature_columns].to_numpy())
    target_scaler.fit(train_imputed[[target_column]].to_numpy())

    # Transform features
    train_scaled[feature_columns] = feature_scaler.transform(train_scaled[feature_columns])
    val_scaled[feature_columns] = feature_scaler.transform(val_scaled[feature_columns])
    test_scaled[feature_columns] = feature_scaler.transform(test_scaled[feature_columns])

    return train_imputed, val_imputed, test_imputed, train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler