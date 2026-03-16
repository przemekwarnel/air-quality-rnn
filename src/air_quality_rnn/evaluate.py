from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Evaluate forecast globally and for each horizon step."""

    # Global metrics
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    r2 = r2_score(y_true.flatten(), y_pred.flatten())

    # Per-horizon metrics
    mae_per_horizon = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_per_horizon = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    r2_per_horizon = np.array([
        r2_score(y_true[:, i], y_pred[:, i])
        for i in range(y_true.shape[1])
    ])
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mae_per_horizon": mae_per_horizon,
        "rmse_per_horizon": rmse_per_horizon,
        "r2_per_horizon": r2_per_horizon,
    }