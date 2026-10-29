"""Tests for chronological splits and simple forecasting baselines."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.baseline_forecasting import (
    chronological_split,
    evaluate_forecast,
    naive_forecast,
    read_series,
    run_baselines,
    seasonal_naive_forecast,
)


def test_chronological_split_keeps_future_values_in_test():
    index = pd.date_range("2024-01-01", periods=10, freq="h")
    series = pd.Series(np.arange(10, dtype=float), index=index)
    train, test = chronological_split(series, test_size=0.3)
    assert len(train) == 7
    assert len(test) == 3
    assert train.index.max() < test.index.min()
    assert train.iloc[-1] == 6
    assert test.iloc[0] == 7


@pytest.mark.parametrize("test_size", [0, 1, -0.2, 1.2])
def test_split_rejects_invalid_test_size(test_size):
    with pytest.raises(ValueError, match="test_size"):
        chronological_split(pd.Series([1.0, 2.0, 3.0]), test_size=test_size)


def test_split_rejects_too_few_valid_observations():
    with pytest.raises(ValueError, match="At least 3"):
        chronological_split(pd.Series([float("nan"), 2.0]))


def test_naive_forecast_repeats_last_training_value():
    train = pd.Series([1.0, 2.5, 3.0])
    assert naive_forecast(train, 4).tolist() == [3.0, 3.0, 3.0, 3.0]


def test_seasonal_naive_repeats_last_training_season():
    train = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert seasonal_naive_forecast(train, horizon=6, season_length=3).tolist() == [
        2.0, 3.0, 4.0, 2.0, 3.0, 4.0
    ]


def test_seasonal_naive_requires_a_full_training_season():
    with pytest.raises(ValueError, match="Need at least"):
        seasonal_naive_forecast(pd.Series([1.0, 2.0]), horizon=2, season_length=3)


def test_metrics_report_zero_safe_mape_and_all_sample_counts():
    actual = pd.Series([2.0, 0.0, 4.0])
    predicted = np.array([1.0, 2.0, 2.0])
    metrics = evaluate_forecast(actual, predicted)
    assert metrics["mae"] == pytest.approx(5 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(3))
    assert metrics["mape_percent"] == pytest.approx(50.0)
    assert metrics["test_observations"] == 3
    assert metrics["mape_observations"] == 2
    assert metrics["mape_zero_or_near_zero_actuals_excluded"] == 1


def test_metrics_return_null_mape_if_all_actuals_are_zero():
    metrics = evaluate_forecast(pd.Series([0.0, 0.0]), np.array([1.0, 2.0]))
    assert metrics["mape_percent"] is None
    assert metrics["mape_observations"] == 0


def test_read_series_cleans_invalid_rows_and_orders_timestamps(tmp_path):
    path = tmp_path / "hourly_energy.csv"
    pd.DataFrame({
        "timestamp": ["2024-01-01 01:00", "bad", "2024-01-01 00:00", "2024-01-01 00:00"],
        "active_power_kw": [2.0, 8.0, 1.0, 9.0],
    }).to_csv(path, index=False)
    series = read_series(path)
    assert len(series) == 2
    assert series.index.is_monotonic_increasing
    assert series.iloc[0] == 1.0


def test_run_baselines_writes_metrics_and_forecast_artifacts(tmp_path):
    processed = tmp_path / "processed"
    output = tmp_path / "reports"
    processed.mkdir()
    hourly_index = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    daily_index = pd.date_range("2024-01-01", periods=60, freq="D")
    pd.DataFrame({
        "timestamp": hourly_index,
        "active_power_kw": 1.5 + np.sin(np.arange(len(hourly_index)) * 2 * np.pi / 24),
    }).to_csv(processed / "hourly_energy.csv", index=False)
    pd.DataFrame({
        "timestamp": daily_index,
        "active_power_kw": 1.5 + np.sin(np.arange(len(daily_index)) * 2 * np.pi / 7),
    }).to_csv(processed / "daily_energy.csv", index=False)
    (processed / "preprocessing_report.json").write_text(
        json.dumps({"source_type": "synthetic"}), encoding="utf-8"
    )

    result = run_baselines(processed, output, test_size=0.2)
    assert result["source_type"] == "synthetic"
    assert result["series"]["hourly"]["train_observations"] == 576
    assert result["series"]["hourly"]["test_observations"] == 144
    assert result["series"]["daily"]["train_observations"] == 48
    assert result["series"]["daily"]["test_observations"] == 12
    assert set(result["series"]["hourly"]["models"]) == {"naive", "seasonal_naive"}
    assert (output / "baseline_results.json").exists()
    assert (output / "baseline_metrics.csv").exists()
    assert (output / "hourly_baseline_forecasts.csv").exists()
    assert (output / "daily_baseline_forecasts.csv").exists()
    assert (output / "hourly_baseline_forecasts.png").exists()
    assert (output / "daily_baseline_forecasts.png").exists()
