import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_targets(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Scale target array while preserving (samples, horizon) shape."""

    original_shape = y.shape
    y_scaled = scaler.transform(y.reshape(-1, 1))

    return y_scaled.reshape(original_shape)


def inverse_scale_targets(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse scale target array while preserving (samples, horizon) shape."""

    original_shape = y_scaled.shape
    y_inversed = scaler.inverse_transform(y_scaled.reshape(-1, 1))

    return y_inversed.reshape(original_shape)


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """Flatten time-series windows for tabular models."""

    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def round_metrics(metrics: dict, decimals: int = 4) -> dict:
    rounded = {}

    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            rounded[key] = np.round(value, decimals)
        elif isinstance(value, (float, np.floating)):
            rounded[key] = round(float(value), decimals)
        else:
            rounded[key] = value

    return rounded