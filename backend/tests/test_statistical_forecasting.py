"""Tests for Step 7 statistical forecasting candidates."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.statistical_forecasting import _fit_candidates, run_statistical_forecasting


def test_candidate_forecasts_use_requested_horizon_and_are_finite():
    index = pd.date_range("2024-01-01", periods=24 * 25, freq="h")
    values = 1.5 + 0.25 * np.sin(np.arange(len(index)) * 2 * np.pi / 24)
    train = pd.Series(values, index=index)
    predictions, outcomes = _fit_candidates(train, horizon=12, season_length=24)

    assert set(outcomes) == {"holt_winters", "arima_1_1_1", "sarima"}
    for name, forecast in predictions.items():
        assert len(forecast) == 12
        assert np.isfinite(forecast).all()
        assert outcomes[name]["status"] == "success"


def test_candidate_fitting_records_failure_without_raising(monkeypatch):
    import scripts.statistical_forecasting as module

    class BrokenFit:
        def fit(self, *args, **kwargs):
            raise RuntimeError("deliberate test failure")

    # Make only the first candidate fail while leaving the others untouched.
    monkeypatch.setattr(module, "ExponentialSmoothing", lambda *args, **kwargs: BrokenFit())
    index = pd.date_range("2024-01-01", periods=24 * 20, freq="h")
    train = pd.Series(1.0 + np.sin(np.arange(len(index)) * 2 * np.pi / 24), index=index)
    predictions, outcomes = _fit_candidates(train, horizon=4, season_length=24)
    assert outcomes["holt_winters"]["status"] == "failed"
    assert "deliberate test failure" in outcomes["holt_winters"]["error"]
    assert "arima_1_1_1" in predictions
    assert "sarima" in predictions


def test_run_statistical_forecasting_writes_results_and_preserves_chronology(tmp_path):
    processed = tmp_path / "processed"
    output = tmp_path / "reports"
    processed.mkdir()
    hourly_index = pd.date_range("2024-01-01", periods=24 * 25, freq="h")
    daily_index = pd.date_range("2024-01-01", periods=90, freq="D")
    pd.DataFrame({
        "timestamp": hourly_index,
        "active_power_kw": 1.4 + 0.2 * np.sin(np.arange(len(hourly_index)) * 2 * np.pi / 24),
    }).to_csv(processed / "hourly_energy.csv", index=False)
    pd.DataFrame({
        "timestamp": daily_index,
        "active_power_kw": 1.4 + 0.2 * np.sin(np.arange(len(daily_index)) * 2 * np.pi / 7),
    }).to_csv(processed / "daily_energy.csv", index=False)
    (processed / "preprocessing_report.json").write_text(
        json.dumps({"source_type": "synthetic"}), encoding="utf-8"
    )

    result = run_statistical_forecasting(processed, output, test_size=0.2)
    assert result["source_type"] == "synthetic"
    for label in ("hourly", "daily"):
        details = result["series"][label]
        assert details["train_observations"] + details["test_observations"] == details["observations"]
        assert details["train_end"] < details["test_start"]
        assert set(details["models"]) == {"holt_winters", "arima_1_1_1", "sarima"}
        assert (output / f"{label}_statistical_forecasts.csv").exists()
        assert (output / f"{label}_statistical_forecasts.png").exists()
    assert (output / "statistical_model_metrics.csv").exists()
    assert (output / "statistical_model_results.json").exists()
    saved = json.loads((output / "statistical_model_results.json").read_text(encoding="utf-8"))
    assert saved["step"] == 7


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1])
def test_run_rejects_invalid_test_size(tmp_path, test_size):
    with pytest.raises(ValueError, match="test_size"):
        run_statistical_forecasting(tmp_path, tmp_path / "out", test_size=test_size)
