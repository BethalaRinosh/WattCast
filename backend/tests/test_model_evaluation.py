"""Tests for Step 9 model comparison, residual diagnostics, and cache invalidation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.model_evaluation import run_model_evaluation


def _write_inputs(root, *, offset=0.0):
    baseline = root / "baselines"
    statistical = root / "statistical"
    ml = root / "ml"
    for directory in (baseline, statistical, ml):
        directory.mkdir(parents=True, exist_ok=True)

    for label, count, frequency in (("hourly", 36, "h"), ("daily", 40, "D")):
        timestamps = pd.date_range("2024-01-01", periods=count, freq=frequency)
        actual = 1.5 + 0.2 * np.sin(np.arange(count) * 2 * np.pi / (24 if label == "hourly" else 7))
        baseline_frame = pd.DataFrame({
            "timestamp": timestamps,
            "actual_kw": actual,
            "naive_forecast_kw": actual - 0.10 - offset,
            "seasonal_naive_forecast_kw": actual - 0.08 - offset,
        })
        baseline_frame.to_csv(baseline / f"{label}_baseline_forecasts.csv", index=False)
        statistical_frame = pd.DataFrame({"timestamp": timestamps, "actual_kw": actual})
        for i, name in enumerate(("holt_winters", "arima_1_1_1", "sarima"), start=1):
            statistical_frame[f"{name}_forecast_kw"] = actual - (0.04 * i) - offset
        statistical_frame.to_csv(statistical / f"{label}_statistical_forecasts.csv", index=False)
        pd.DataFrame({
            "timestamp": timestamps,
            "actual_kw": actual,
            "random_forest_forecast_kw": actual - 0.03 - offset,
        }).to_csv(ml / f"{label}_ml_forecasts.csv", index=False)
    return baseline, statistical, ml


def test_evaluation_compares_models_and_writes_diagnostics(tmp_path):
    baseline, statistical, ml = _write_inputs(tmp_path)
    output = tmp_path / "evaluation"

    result = run_model_evaluation(baseline, statistical, ml, output)

    assert result["step"] == 9
    assert result["cache"]["hit"] is False
    for series in ("hourly", "daily"):
        details = result["series"][series]
        assert len(details["models"]) == 6
        assert details["best_by_mae"] == "random_forest_lag_calendar"
        assert details["best_by_rmse"] == "random_forest_lag_calendar"
        rf = details["models"]["random_forest_lag_calendar"]
        assert rf["metrics"]["mae"] == pytest.approx(0.03)
        assert rf["residual_diagnostics"]["mean_error_kw"] == pytest.approx(0.03)
        assert rf["evaluated_timestamps"] > 0
        assert (output / f"{series}_random_forest_lag_calendar_residuals.png").exists()
    assert (output / "model_evaluation_metrics.csv").exists()
    assert (output / "evaluation_cache.json").exists()
    assert (output / "evaluation_results.json").exists()


def test_cache_hit_reuses_metrics_and_changed_input_invalidates_cache(tmp_path):
    baseline, statistical, ml = _write_inputs(tmp_path)
    output = tmp_path / "evaluation"

    first = run_model_evaluation(baseline, statistical, ml, output)
    second = run_model_evaluation(baseline, statistical, ml, output)
    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert second["series"]["hourly"]["best_by_mae"] == first["series"]["hourly"]["best_by_mae"]

    _write_inputs(tmp_path, offset=0.02)
    third = run_model_evaluation(baseline, statistical, ml, output)
    assert third["cache"]["hit"] is False


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1])
def test_evaluation_rejects_invalid_test_size(tmp_path, test_size):
    with pytest.raises(ValueError, match="test_size"):
        run_model_evaluation(tmp_path, tmp_path, tmp_path, tmp_path / "out", test_size=test_size)


def test_missing_forecast_artifact_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run Steps 6, 7, and 8 first"):
        run_model_evaluation(tmp_path, tmp_path, tmp_path, tmp_path / "out")
