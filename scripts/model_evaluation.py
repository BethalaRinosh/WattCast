#!/usr/bin/env python3
"""Compare WattCast forecasts, inspect residuals, and cache reproducible evaluation results."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .baseline_forecasting import evaluate_forecast
except ImportError:
    from baseline_forecasting import evaluate_forecast

LOGGER = logging.getLogger("wattcast.model_evaluation")
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "backend" / "reports"
DEFAULT_BASELINE_DIR = REPORTS / "baselines"
DEFAULT_STATISTICAL_DIR = REPORTS / "statistical_models"
DEFAULT_ML_DIR = REPORTS / "ml_forecasting"
DEFAULT_OUTPUT_DIR = REPORTS / "evaluation"
CACHE_VERSION = 1


def _forecast_specs(series: str) -> list[tuple[str, Path, str]]:
    """Return model names, source files, and prediction columns in stable order."""
    baseline = DEFAULT_BASELINE_DIR / f"{series}_baseline_forecasts.csv"
    statistical = DEFAULT_STATISTICAL_DIR / f"{series}_statistical_forecasts.csv"
    ml = DEFAULT_ML_DIR / f"{series}_ml_forecasts.csv"
    specs = [
        ("naive", baseline, "naive_forecast_kw"),
        ("seasonal_naive", baseline, "seasonal_naive_forecast_kw"),
    ]
    for model in ("holt_winters", "arima_1_1_1", "sarima"):
        specs.append((model, statistical, f"{model}_forecast_kw"))
    specs.append(("random_forest_lag_calendar", ml, "random_forest_forecast_kw"))
    return specs


def _fingerprint(paths: list[Path], test_size: float) -> str:
    digest = hashlib.sha256()
    digest.update(f"cache-version={CACHE_VERSION};test-size={test_size:.12g}".encode())
    for path in sorted(paths, key=lambda item: str(item).lower()):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required forecast artifact is missing: {path}. Run Steps 6, 7, and 8 first."
            )
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_forecast(path: Path, prediction_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", "actual_kw", prediction_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in ("actual_kw", prediction_column):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["timestamp", "actual_kw", prediction_column])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    if frame.empty:
        raise ValueError(f"No valid forecast rows remain in {path.name}.")
    return frame[["timestamp", "actual_kw", prediction_column]].rename(
        columns={prediction_column: "predicted_kw"}
    )


def _residual_diagnostics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    residuals = actual - predicted
    centered = residuals - residuals.mean()
    denominator = float(np.sqrt(np.sum(centered[:-1] ** 2) * np.sum(centered[1:] ** 2))) if len(centered) > 1 else 0.0
    lag1 = float(np.sum(centered[:-1] * centered[1:]) / denominator) if denominator > 0 else None
    std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    return {
        "residual_definition": "actual_kw - predicted_kw",
        "mean_error_kw": float(np.mean(residuals)),
        "median_error_kw": float(np.median(residuals)),
        "residual_std_kw": std,
        "mean_absolute_error_kw": float(np.mean(np.abs(residuals))),
        "max_absolute_error_kw": float(np.max(np.abs(residuals))),
        "lag_1_residual_autocorrelation": lag1,
        "fraction_within_two_residual_std": (
            float(np.mean(np.abs(residuals - np.mean(residuals)) <= 2 * std)) if std > 0 else 1.0
        ),
        "interpretation_note": (
            "Positive mean error means the model under-forecast on average; negative means it over-forecast. "
            "Residual autocorrelation may indicate time structure left unexplained."
        ),
    }


def _plot_diagnostics(frame: pd.DataFrame, model: str, series: str, output_dir: Path) -> None:
    actual = frame["actual_kw"].to_numpy(dtype=float)
    predicted = frame["predicted_kw"].to_numpy(dtype=float)
    residuals = actual - predicted
    safe_model = model.replace("/", "_")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(frame["timestamp"], residuals, linewidth=0.8)
    axes[0].axhline(0, linewidth=1, linestyle="--")
    axes[0].set_title(f"{series.title()} residuals over time | {model}")
    axes[0].set_ylabel("Error (kW)")
    axes[0].grid(True, alpha=0.25)
    axes[1].scatter(predicted, residuals, s=8, alpha=0.55)
    axes[1].axhline(0, linewidth=1, linestyle="--")
    axes[1].set_title("Residuals versus predicted values")
    axes[1].set_xlabel("Predicted mean active power (kW)")
    axes[1].set_ylabel("Error (kW)")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(output_dir / f"{series}_{safe_model}_residuals.png", dpi=130)
    plt.close(fig)


def run_model_evaluation(
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    statistical_dir: Path = DEFAULT_STATISTICAL_DIR,
    ml_dir: Path = DEFAULT_ML_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.2,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Compare available model forecasts on identical timestamps and cache by input content."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    baseline_dir, statistical_dir, ml_dir, output_dir = map(
        Path, (baseline_dir, statistical_dir, ml_dir, output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_specs: dict[str, list[tuple[str, Path, str]]] = {}
    for series in ("hourly", "daily"):
        all_specs[series] = [
            (name, (baseline_dir if "baseline" in str(path) else path.parent), column)
            for name, path, column in _forecast_specs(series)
        ]
        all_specs[series] = [
            (name, baseline_dir / path.name if "baseline_forecasts" in path.name else
             statistical_dir / path.name if "statistical_forecasts" in path.name else
             ml_dir / path.name, column)
            for name, path, column in _forecast_specs(series)
        ]
    input_paths = list(dict.fromkeys(path for specs in all_specs.values() for _, path, _ in specs))
    fingerprint = _fingerprint(input_paths, test_size)
    cache_path = output_dir / "evaluation_cache.json"
    results_path = output_dir / "evaluation_results.json"
    cache: dict[str, Any] | None = None
    if use_cache and cache_path.exists():
        try:
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if candidate.get("fingerprint") == fingerprint:
                cache = candidate.get("result")
        except (OSError, json.JSONDecodeError):
            cache = None

    if cache is not None:
        summary = cache
        summary["cache"] = {"hit": True, "fingerprint": fingerprint}
        LOGGER.info("Evaluation cache hit; reusing metrics for unchanged forecast artifacts.")
    else:
        summary = {
            "project": "WattCast",
            "step": 9,
            "source_type": "unknown",
            "target": "Mean active power (kW)",
            "test_size_fraction": test_size,
            "comparison_policy": "Models are compared only after aligning actuals by timestamp; metrics use shared valid timestamps.",
            "series": {},
        }
        metric_rows: list[dict[str, Any]] = []
        for series, specs in all_specs.items():
            models: dict[str, Any] = {}
            reference_actual: pd.Series | None = None
            residual_frames: dict[str, pd.DataFrame] = {}
            for model, path, column in specs:
                frame = _load_forecast(path, column)
                actual = frame.set_index("timestamp")["actual_kw"]
                if reference_actual is None:
                    reference_actual = actual
                else:
                    common = reference_actual.index.intersection(actual.index)
                    if len(common) == 0:
                        raise ValueError(f"No overlapping timestamps between model forecasts for {series}.")
                    reference_actual = reference_actual.loc[common]
                residual_frames[model] = frame.set_index("timestamp")
            assert reference_actual is not None
            aligned: dict[str, pd.DataFrame] = {}
            for model, frame in residual_frames.items():
                common = reference_actual.index.intersection(frame.index)
                aligned_frame = frame.loc[common].copy()
                aligned_frame["actual_kw"] = reference_actual.loc[common]
                aligned_frame = aligned_frame.sort_index()
                if len(aligned_frame) == 0:
                    continue
                if not np.allclose(
                    aligned_frame["actual_kw"].to_numpy(),
                    frame.loc[common, "actual_kw"].to_numpy(),
                    rtol=1e-8, atol=1e-10,
                ):
                    raise ValueError(f"Actual holdout values disagree for {series}/{model}; cannot compare fairly.")
                aligned[model] = aligned_frame.reset_index()
            if len(aligned) < 2:
                raise ValueError(f"At least two successful models are required to compare {series} forecasts.")
            for model, frame in aligned.items():
                actual = frame["actual_kw"].to_numpy(dtype=float)
                predicted = frame["predicted_kw"].to_numpy(dtype=float)
                metrics = evaluate_forecast(pd.Series(actual), predicted)
                residuals = _residual_diagnostics(actual, predicted)
                models[model] = {
                    "status": "success",
                    "metrics": metrics,
                    "residual_diagnostics": residuals,
                    "evaluated_timestamps": int(len(frame)),
                    "evaluation_start": frame["timestamp"].min().isoformat(),
                    "evaluation_end": frame["timestamp"].max().isoformat(),
                }
                metric_rows.append({"series": series, "model": model, "status": "success", **metrics, **{
                    key: value for key, value in residuals.items() if not isinstance(value, str)
                }})
            ranking = sorted(
                ({"model": model, "mae": details["metrics"]["mae"], "rmse": details["metrics"]["rmse"]}
                 for model, details in models.items()),
                key=lambda row: (row["mae"], row["rmse"]),
            )
            summary["series"][series] = {
                "models": models,
                "ranking_by_mae_then_rmse": ranking,
                "best_by_mae": ranking[0]["model"],
                "best_by_rmse": min(models, key=lambda name: models[name]["metrics"]["rmse"]),
                "evaluation_observations": min(details["evaluated_timestamps"] for details in models.values()),
            }
        pd.DataFrame(metric_rows).to_csv(output_dir / "model_evaluation_metrics.csv", index=False)
        summary["cache"] = {"hit": False, "fingerprint": fingerprint}
        cache_path.write_text(json.dumps({"fingerprint": fingerprint, "result": summary}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        LOGGER.info("Evaluation cache miss; computed metrics from current forecast artifacts.")

    # Plotting is cheap and intentionally runs even on a cache hit so deleted images are restored.
    for series, specs in all_specs.items():
        for model, path, column in specs:
            frame = _load_forecast(path, column)
            _plot_diagnostics(frame, model, series, output_dir)
    summary["cache"] = {"hit": cache is not None, "fingerprint": fingerprint}
    results_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--statistical-dir", type=Path, default=DEFAULT_STATISTICAL_DIR)
    parser.add_argument("--ml-dir", type=Path, default=DEFAULT_ML_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-cache", action="store_true", help="Recompute metrics even if inputs are unchanged.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    result = run_model_evaluation(
        args.baseline_dir, args.statistical_dir, args.ml_dir, args.output_dir,
        test_size=args.test_size, use_cache=not args.no_cache,
    )
    for series, details in result["series"].items():
        LOGGER.info("%s: best MAE=%s; best RMSE=%s; observations=%s",
                    series, details["best_by_mae"], details["best_by_rmse"], details["evaluation_observations"])
    LOGGER.info("Evaluation outputs saved to %s (cache hit=%s)", args.output_dir, result["cache"]["hit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
