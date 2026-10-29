#!/usr/bin/env python3
"""Fit Holt-Winters, ARIMA and SARIMA candidates on WattCast's chronological holdout."""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from scripts.baseline_forecasting import (
    DEFAULT_PROCESSED_DIR,
    chronological_split,
    evaluate_forecast,
    longest_regular_segment,
    read_series,
)

LOGGER = logging.getLogger("wattcast.statistical_forecasting")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "backend" / "reports" / "statistical_models"


def _fit_candidates(
    train: pd.Series, horizon: int, season_length: int
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Fit each candidate using training data only; isolate model failures."""
    predictions: dict[str, np.ndarray] = {}
    outcomes: dict[str, dict[str, Any]] = {}

    candidates = {
        "holt_winters": lambda: ExponentialSmoothing(
            train.astype(float),
            trend="add",
            seasonal="add",
            seasonal_periods=season_length,
            initialization_method="estimated",
        ).fit(optimized=True),
        "arima_1_1_1": lambda: ARIMA(
            train.astype(float), order=(1, 1, 1), enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(),
        "sarima": lambda: SARIMAX(
            train.astype(float),
            order=(1, 0, 1),
            seasonal_order=(1, 0, 0, season_length),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=30),
    }

    for name, factory in candidates.items():
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = factory()
                forecast = np.asarray(fitted.forecast(steps=horizon), dtype=float)
            if len(forecast) != horizon or not np.isfinite(forecast).all():
                raise ValueError("Model produced a non-finite forecast or wrong horizon length.")
            predictions[name] = forecast
            outcomes[name] = {
                "status": "success",
                "warnings": [str(item.message) for item in caught],
                "model_specification": {
                    "holt_winters": "additive trend + additive seasonality",
                    "arima_1_1_1": "ARIMA(1,1,1)",
                    "sarima": f"SARIMA(1,0,1)x(1,0,0,{season_length})",
                }[name],
            }
        except Exception as exc:  # One candidate failing must not hide other model results.
            outcomes[name] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_specification": {
                    "holt_winters": "additive trend + additive seasonality",
                    "arima_1_1_1": "ARIMA(1,1,1)",
                    "sarima": f"SARIMA(1,0,1)x(1,0,0,{season_length})",
                }[name],
            }
            LOGGER.warning("%s fit failed: %s: %s", name, type(exc).__name__, exc)
    return predictions, outcomes


def run_statistical_forecasting(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Evaluate statistical candidates on the same chronological split as Step 6."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = processed_dir / "preprocessing_report.json"
    source_type = "unknown"
    if provenance_path.exists():
        source_type = json.loads(provenance_path.read_text(encoding="utf-8")).get(
            "source_type", "unknown"
        )

    summary: dict[str, Any] = {
        "project": "WattCast",
        "step": 7,
        "source_type": source_type,
        "target": "Mean active power (kW)",
        "split": {
            "method": "chronological; no shuffle",
            "test_size_fraction": test_size,
            "training_period_precedes_test_period": True,
        },
        "method_notes": [
            "All candidate models are fitted only on the training segment; the chronological holdout is never used for fitting.",
            "Hourly and daily inputs are regularized using the Step 6 policy: interpolate only complete internal gaps of at most three bins, then select the longest contiguous segment.",
            "Holt-Winters uses additive trend and additive seasonality; ARIMA uses order (1,1,1); SARIMA uses order (1,0,1) with a seasonal AR term at period 24 hourly or 7 daily.",
            "Each candidate is isolated: a failed fit is recorded and does not erase successful model results.",
            "Model metrics are comparable only when evaluated on the same series and chronological test interval.",
            "These are initial candidate specifications, not a claim of optimal hyperparameters.",
        ],
        "series": {},
    }
    metric_rows: list[dict[str, Any]] = []
    for label, filename, frequency, season_length in (
        ("hourly", "hourly_energy.csv", "h", 24),
        ("daily", "daily_energy.csv", "D", 7),
    ):
        series = read_series(processed_dir / filename)
        regular, regularization = longest_regular_segment(series, frequency)
        train, test = chronological_split(regular, test_size=test_size)
        if len(train) < 2 * season_length:
            raise ValueError(
                f"{label} training data needs at least {2 * season_length} observations "
                f"for seasonal candidate models; got {len(train)}."
            )
        predictions, outcomes = _fit_candidates(train, len(test), season_length)
        result: dict[str, Any] = {
            "frequency": "hourly" if label == "hourly" else "daily",
            "season_length": season_length,
            "observations": int(len(regular)),
            "train_observations": int(len(train)),
            "test_observations": int(len(test)),
            "train_start": train.index.min().isoformat(),
            "train_end": train.index.max().isoformat(),
            "test_start": test.index.min().isoformat(),
            "test_end": test.index.max().isoformat(),
            "regularization": regularization,
            "models": {},
        }
        forecast_frame = pd.DataFrame({"timestamp": test.index, "actual_kw": test.to_numpy()})
        for model_name in ("holt_winters", "arima_1_1_1", "sarima"):
            outcome = outcomes[model_name]
            if model_name in predictions:
                forecast = predictions[model_name]
                metrics = evaluate_forecast(test, forecast)
                result["models"][model_name] = {**outcome, "metrics": metrics}
                metric_rows.append({"series": label, "model": model_name, **metrics, "status": "success"})
                forecast_frame[f"{model_name}_forecast_kw"] = forecast
            else:
                result["models"][model_name] = outcome
                metric_rows.append({
                    "series": label, "model": model_name, "status": "failed",
                    "error_type": outcome["error_type"], "error": outcome["error"],
                })

        forecast_frame.to_csv(output_dir / f"{label}_statistical_forecasts.csv", index=False)
        plot_count = min(len(test), 24 * 14 if label == "hourly" else 90)
        plot_frame = forecast_frame.iloc[:plot_count]
        fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
        ax.plot(plot_frame["timestamp"], plot_frame["actual_kw"], label="Actual", linewidth=1.2)
        for model_name in predictions:
            column = f"{model_name}_forecast_kw"
            ax.plot(plot_frame["timestamp"], plot_frame[column], label=model_name.replace("_", " "), linewidth=0.9)
        ax.set_title(f"{label.title()} statistical forecasts | {source_type.upper()} DATA")
        ax.set_ylabel("Mean active power (kW)")
        ax.set_xlabel("Timestamp")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.savefig(output_dir / f"{label}_statistical_forecasts.png", dpi=140)
        plt.close(fig)
        summary["series"][label] = result

    pd.DataFrame(metric_rows).to_csv(output_dir / "statistical_model_metrics.csv", index=False)
    (output_dir / "statistical_model_results.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Statistical model evaluation complete for source=%s; outputs saved to %s",
        source_type, output_dir,
    )
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
    summary = run_statistical_forecasting(args.processed_dir, args.output_dir, args.test_size)
    for series_name, result in summary["series"].items():
        LOGGER.info(
            "%s split: %s train / %s test (%s to %s)",
            series_name, result["train_observations"], result["test_observations"],
            result["test_start"], result["test_end"],
        )
        for model_name, outcome in result["models"].items():
            if outcome["status"] == "success":
                metrics = outcome["metrics"]
                LOGGER.info(
                    "%s %s: MAE=%.5f RMSE=%.5f MAPE=%s",
                    series_name, model_name, metrics["mae"], metrics["rmse"],
                    f"{metrics['mape_percent']:.3f}%" if metrics["mape_percent"] is not None else "n/a",
                )
            else:
                LOGGER.warning("%s %s failed: %s", series_name, model_name, outcome["error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
