import argparse
import json
from pathlib import Path

from air_quality_forecasting.visualization import plot_mae_per_horizon, plot_r2_per_horizon


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """Generate horizon plots for baseline, Ridge, and optionally LSTM."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lstm-experiment",
        default=None,
        help="Optional LSTM experiment name, e.g. lstm_baseline",
    )
    args = parser.parse_args()

    baselines_results = load_json("reports/baselines_metrics.json")
    ridge_results = load_json("reports/ridge_metrics.json")

    naive_metrics = baselines_results["naive"]
    ridge_metrics = ridge_results["test_metrics"]

    lstm_metrics = None
    if args.lstm_experiment is not None:
        lstm_path = f"reports/experiments/{args.lstm_experiment}.json"
        lstm_results = load_json(lstm_path)
        lstm_metrics = lstm_results["test_metrics"]

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    plot_mae_per_horizon(
        naive_metrics=naive_metrics,
        ridge_metrics=ridge_metrics,
        lstm_metrics=lstm_metrics,
        save_path="reports/figures/mae_per_horizon.png",
    )

    plot_r2_per_horizon(
        naive_metrics=naive_metrics,
        ridge_metrics=ridge_metrics,
        lstm_metrics=lstm_metrics,
        save_path="reports/figures/r2_per_horizon.png",
    )


if __name__ == "__main__":
    main()