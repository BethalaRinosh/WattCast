#!/usr/bin/env python3
"""Create chronological holdout splits and evaluate simple WattCast baselines."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("wattcast.baselines")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = ROOT / "backend" / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "backend" / "reports" / "baselines"
TARGET = "active_power_kw"


def read_series(path: Path) -> pd.Series:
    """Load one processed CSV into a finite, ordered numeric series."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. Run scripts/preprocess_data.py first."
        )
    frame = pd.read_csv(path)
    missing = {"timestamp", TARGET}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame[TARGET] = frame[TARGET].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["timestamp", TARGET])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    if frame.empty:
        raise ValueError(f"No valid observations found in {path}.")
    return frame.set_index("timestamp")[TARGET].astype(float).sort_index()


def chronological_split(
    series: pd.Series, test_size: float = 0.2
) -> tuple[pd.Series, pd.Series]:
    """Split by time order; the final test_size fraction is held out."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    clean = series.dropna().sort_index()
    if len(clean) < 3:
        raise ValueError("At least 3 valid observations are required for a train/test split.")
    split_at = int(np.floor(len(clean) * (1 - test_size)))
    split_at = min(max(split_at, 1), len(clean) - 1)
    train, test = clean.iloc[:split_at].copy(), clean.iloc[split_at:].copy()
    if train.index.max() >= test.index.min():
        raise ValueError("Chronological split invariant failed: training overlaps test period.")
    return train, test


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Predict every future value using the last training observation."""
    if len(train) == 0 or horizon < 1:
        raise ValueError("Naive forecast requires training data and a positive horizon.")
    return np.full(horizon, float(train.iloc[-1]), dtype=float)


def seasonal_naive_forecast(
    train: pd.Series, horizon: int, season_length: int
) -> np.ndarray:
    """Repeat the final observed training season without using test observations."""
    if season_length < 1:
        raise ValueError("season_length must be a positive integer.")
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    if len(train) < season_length:
        raise ValueError(
            f"Need at least {season_length} training observations for seasonal-naive; "
            f"got {len(train)}."
        )
    last_season = np.asarray(train.iloc[-season_length:], dtype=float)
    return np.resize(last_season, horizon)


def evaluate_forecast(actual: pd.Series, predicted: np.ndarray) -> dict[str, Any]:
    """Calculate MAE/RMSE and zero-safe MAPE with transparent sample counts."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted values must have the same non-zero length.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Actual and predicted values must all be finite.")
    errors = y_true - y_pred
    nonzero = np.abs(y_true) > 1e-8
    mape = (
        float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100)
        if nonzero.any() else None
    )
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape_percent": mape,
        "test_observations": int(len(y_true)),
        "mape_observations": int(nonzero.sum()),
        "mape_zero_or_near_zero_actuals_excluded": int((~nonzero).sum()),
        "mape_note": (
            "MAPE excludes actual values whose absolute magnitude is <= 1e-8; "
            "the excluded count is reported. MAE and RMSE use every test observation."
        ),
    }


def run_baselines(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Evaluate naive and seasonal-naive forecasts on hourly and daily holdouts."""
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_report = processed_dir / "preprocessing_report.json"
    source_type = "unknown"
    if preprocessing_report.exists():
        source_type = json.loads(
            preprocessing_report.read_text(encoding="utf-8")
        ).get("source_type", "unknown")

    summary: dict[str, Any] = {
        "project": "WattCast",
        "source_type": source_type,
        "target": "Mean active power (kW)",
        "split": {
            "method": "chronological; no shuffle",
            "test_size_fraction": test_size,
            "training_period_precedes_test_period": True,
        },
        "series": {},
        "method_notes": [
            "The final chronological fraction is held out; no test observations are used to form forecasts.",
            "Naive predicts the final training value for every test timestamp.",
            "Seasonal-naive repeats the last full season observed in training (24 hours for hourly data, 7 days for daily data).",
            "These are reference baselines, not fitted advanced models.",
            "MAPE excludes zero and near-zero actuals; MAE and RMSE include all test observations.",
        ],
    }
    comparison_rows: list[dict[str, Any]] = []
    for label, filename, season_length, season_name in (
        ("hourly", "hourly_energy.csv", 24, "24-hour"),
        ("daily", "daily_energy.csv", 7, "7-day"),
    ):
        series = read_series(processed_dir / filename)
        train, test = chronological_split(series, test_size=test_size)
        predictions = {
            "naive": naive_forecast(train, len(test)),
            "seasonal_naive": seasonal_naive_forecast(train, len(test), season_length),
        }
        summary["series"][label] = {
            "frequency": "hourly" if label == "hourly" else "daily",
            "season_length": season_length,
            "season_name": season_name,
            "observations": int(len(series)),
            "train_observations": int(len(train)),
            "test_observations": int(len(test)),
            "train_start": train.index.min().isoformat(),
            "train_end": train.index.max().isoformat(),
            "test_start": test.index.min().isoformat(),
            "test_end": test.index.max().isoformat(),
            "models": {},
        }
        for model_name, predicted in predictions.items():
            metrics = evaluate_forecast(test, predicted)
            summary["series"][label]["models"][model_name] = metrics
            comparison_rows.append({
                "series": label,
                "model": model_name,
                **metrics,
            })

        forecast_frame = pd.DataFrame({
            "timestamp": test.index,
            "actual_kw": test.to_numpy(),
            "naive_forecast_kw": predictions["naive"],
            "seasonal_naive_forecast_kw": predictions["seasonal_naive"],
        })
        forecast_frame.to_csv(output_dir / f"{label}_baseline_forecasts.csv", index=False)

        # Plot only a manageable window to keep the artifact readable.
        plot_count = min(len(test), 24 * 14 if label == "hourly" else 90)
        plot_frame = forecast_frame.iloc[:plot_count]
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax.plot(plot_frame["timestamp"], plot_frame["actual_kw"], label="Actual", linewidth=1.0)
        ax.plot(plot_frame["timestamp"], plot_frame["naive_forecast_kw"], label="Naive", linewidth=0.9)
        ax.plot(
            plot_frame["timestamp"], plot_frame["seasonal_naive_forecast_kw"],
            label="Seasonal naive", linewidth=0.9,
        )
        ax.set_title(f"{label.title()} holdout: baseline forecasts")
        ax.set_ylabel("Mean active power (kW)")
        ax.set_xlabel("Timestamp")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.savefig(output_dir / f"{label}_baseline_forecasts.png", dpi=140)
        plt.close(fig)

    pd.DataFrame(comparison_rows).to_csv(output_dir / "baseline_metrics.csv", index=False)
    (output_dir / "baseline_results.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Baseline evaluation complete for source=%s; outputs saved to %s", source_type, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    summary = run_baselines(args.processed_dir, args.output_dir, args.test_size)
    for series_name, result in summary["series"].items():
        LOGGER.info(
            "%s split: %s train / %s test (%s to %s)",
            series_name, result["train_observations"], result["test_observations"],
            result["test_start"], result["test_end"],
        )
        for model_name, metrics in result["models"].items():
            LOGGER.info(
                "%s %s: MAE=%.5f RMSE=%.5f MAPE=%s",
                series_name, model_name, metrics["mae"], metrics["rmse"],
                f"{metrics['mape_percent']:.3f}%" if metrics["mape_percent"] is not None else "n/a",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
