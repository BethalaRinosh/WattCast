"""Tests for WattCast exploratory data analysis."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.eda import build_eda_summary, read_series, run_eda


def test_read_series_validates_and_sorts_timestamps(tmp_path):
    path = tmp_path / "hourly_energy.csv"
    pd.DataFrame({
        "timestamp": ["2024-01-01 01:00:00", "2024-01-01 00:00:00"],
        "active_power_kw": [2.0, 1.0],
    }).to_csv(path, index=False)

    series = read_series(path)
    assert series.index.is_monotonic_increasing
    assert series.tolist() == [1.0, 2.0]


def test_read_series_missing_file_has_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess_data.py"):
        read_series(tmp_path / "missing.csv")


def test_summary_labels_synthetic_data_and_profiles():
    hourly_index = pd.date_range("2024-01-01", periods=24 * 14, freq="h")
    hourly = pd.Series([float(ts.hour + 1) for ts in hourly_index], index=hourly_index)
    daily_index = pd.date_range("2024-01-01", periods=14, freq="D")
    daily = pd.Series([float(ts.day) for ts in daily_index], index=daily_index)

    summary = build_eda_summary(hourly, daily, "synthetic")
    assert summary["source_label"] == "SYNTHETIC SIMULATED DATA"
    assert summary["peak_mean_hour"] == 23
    assert summary["lowest_mean_hour"] == 0
    assert len(summary["intraday_mean_power_kw"]) == 24
    assert len(summary["weekday_mean_power_kw"]) == 7
    assert len(summary["monthly_mean_power_kw"]) == 12
    assert summary["monthly_mean_power_kw"]["1"] is not None


def test_run_eda_writes_json_profiles_and_charts(tmp_path):
    processed = tmp_path / "processed"
    output = tmp_path / "eda"
    processed.mkdir()
    hourly_index = pd.date_range("2024-01-01", periods=24 * 14, freq="h")
    daily_index = pd.date_range("2024-01-01", periods=14, freq="D")
    pd.DataFrame({
        "timestamp": hourly_index,
        "active_power_kw": [1.0 + (ts.hour / 24) for ts in hourly_index],
    }).to_csv(processed / "hourly_energy.csv", index=False)
    pd.DataFrame({
        "timestamp": daily_index,
        "active_power_kw": [1.5 + (ts.day / 100) for ts in daily_index],
    }).to_csv(processed / "daily_energy.csv", index=False)
    (processed / "preprocessing_report.json").write_text(
        json.dumps({"source_type": "synthetic"}), encoding="utf-8"
    )

    summary = run_eda(processed, output)
    assert summary["source_type"] == "synthetic"
    assert (output / "eda_summary.json").exists()
    assert (output / "intraday_profile.csv").exists()
    assert (output / "weekday_profile.csv").exists()
    assert (output / "monthly_profile.csv").exists()
    for filename in (
        "hourly_series.png",
        "daily_series.png",
        "intraday_profile.png",
        "weekday_profile.png",
    ):
        assert (output / filename).exists()
        assert (output / filename).stat().st_size > 0
