import json
from pathlib import Path

from air_quality_rnn.visualization import plot_mae_per_horizon, plot_r2_per_horizon


def main():
    """Generate MAE and R² horizon plots comparing naive baseline and Ridge model."""

    # Load results 
    with open("reports/baselines.json", "r", encoding="utf-8") as f:
        baselines_results = json.load(f)

    with open("reports/linear_model.json", "r", encoding="utf-8") as f:
        linear_model_results = json.load(f)

    # Select appropriate metrics 
    naive_metrics = baselines_results["naive"]
    ridge_metrics = linear_model_results["test_metrics"]

    # Create figures directory 
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Generate and save plots
    plot_mae_per_horizon(
        naive_metrics=naive_metrics,
        ridge_metrics=ridge_metrics,
        save_path="reports/figures/mae_per_horizon.png",
    )

    plot_r2_per_horizon(
        naive_metrics=naive_metrics,
        ridge_metrics=ridge_metrics,
        save_path="reports/figures/r2_per_horizon.png",
    )


if __name__ == "__main__":
    main()