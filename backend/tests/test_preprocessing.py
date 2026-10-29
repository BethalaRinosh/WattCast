"""Unit tests for WattCast dataset loading and preprocessing."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.data_processing import clean_and_resample, inspect_dataset, load_dataset


def test_synthetic_csv_loads_into_common_schema(tmp_path):
    path = tmp_path / "synthetic_hourly_energy.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=48, freq="h").astype(str),
        "global_active_power_kw": [1.0] * 48,
        "data_source": ["synthetic"] * 48,
    }).to_csv(path, index=False)

    loaded = load_dataset(path, source_type="synthetic")
    assert loaded.index.name == "timestamp"
    assert list(loaded.columns) == ["active_power_kw"]
    assert len(loaded) == 48
    assert loaded["active_power_kw"].iloc[0] == pytest.approx(1.0)


def test_uci_semicolon_file_parses_date_and_missing_values(tmp_path):
    path = tmp_path / "household_power_consumption.txt"
    path.write_text(
        "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n"
        "01/01/2007;00:00:00;1.500;0.100;240.0;6.0;0.0;0.0;0.0\n"
        "01/01/2007;00:01:00;?;?;?;?;?;?;?\n",
        encoding="utf-8",
    )
    loaded = load_dataset(path, source_type="real")
    assert len(loaded) == 2
    assert loaded.index[0] == pd.Timestamp("2007-01-01 00:00:00")
    assert pd.isna(loaded["active_power_kw"].iloc[1])


def test_resampling_produces_hourly_and_daily_series():
    index = pd.date_range("2024-01-01", periods=48, freq="h", name="timestamp")
    frame = pd.DataFrame({"active_power_kw": [1.0] * 24 + [3.0] * 24}, index=index)
    hourly, daily, report = clean_and_resample(frame, "synthetic")
    assert len(hourly) == 48
    assert len(daily) == 2
    assert daily["active_power_kw"].tolist() == pytest.approx([1.0, 3.0])
    assert report["source_type"] == "synthetic"
    assert report["iqr_outliers_detected"]["count"] == 0


def test_inspection_reports_frequency_and_missing_values():
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="timestamp")
    frame = pd.DataFrame({"active_power_kw": [1.0, float("nan"), 2.0]}, index=index)
    report = inspect_dataset(frame, "synthetic")
    assert report["rows"] == 3
    assert report["inferred_frequency"] == "hourly"
    assert report["missing_values"]["active_power_kw"] == 1


def test_missing_input_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_dataset(tmp_path / "missing.csv", source_type="synthetic")
