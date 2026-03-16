import matplotlib.pyplot as plt
import numpy as np


def plot_mae_per_horizon(
    naive_metrics: dict,
    ridge_metrics: dict,
    save_path: str,
) -> None:
    """Plot MAE for each forecast horizon step for baseline and Ridge model."""

    horizon = len(naive_metrics["mae_per_horizon"])
    steps = range(1, horizon + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        naive_metrics["mae_per_horizon"],
        label="Naive",
        marker="o",
    )
    
    plt.plot(
        steps,
        ridge_metrics["mae_per_horizon"],
        label="Ridge Regression",
        marker="o"
    )
    
    plt.title("MAE per Forecast Horizon")
    plt.xlabel("Horizon Step (hours ahead)")
    plt.ylabel("MAE")
    plt.xticks(steps)
    plt.xlim(1, horizon)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_r2_per_horizon(
    naive_metrics: dict,
    ridge_metrics: dict,
    save_path: str,
) -> None:
    """Plot R² for each forecast horizon step for baseline and Ridge model."""

    horizon = len(naive_metrics["r2_per_horizon"])
    steps = range(1, horizon + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        naive_metrics["r2_per_horizon"],
        label="Naive",
        marker="o",
    )
    
    plt.plot(
        steps,
        ridge_metrics["r2_per_horizon"],
        label="Ridge Regression",
        marker="o"
    )
    
    plt.title("R² per Forecast Horizon")
    plt.xlabel("Horizon Step (hours ahead)")
    plt.ylabel("R²")
    plt.xticks(steps)
    plt.xlim(1, horizon)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_forecast_example(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_idx: int,
    save_path: str,
) -> None:
    """Plot true vs predicted forecast trajectory for a single sample."""
    
    horizon = y_true.shape[1]
    steps = range(1, horizon + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        y_true[sample_idx],
        label="True PM2.5",
        marker="o",
    )
    
    plt.plot(
        steps,
        y_pred[sample_idx],
        label="Ridge Prediction",
        marker="o"
    )
    
    plt.title("Example Forecast: True vs Predicted PM2.5")
    plt.xlabel("Horizon Step (hours ahead)")
    plt.ylabel("PM2.5")
    plt.xticks(steps)
    plt.xlim(1, horizon)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    