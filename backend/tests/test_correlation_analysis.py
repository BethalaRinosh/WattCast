"""Tests for Pearson and Spearman correlation analysis."""
import pandas as pd
import pytest

from scripts.correlation_analysis import (
    correlation_payload,
    load_real_measurements,
    resample_measurements,
)


def test_real_loader_preserves_all_uci_numeric_measurements(tmp_path):
    path = tmp_path / "household_power_consumption.txt"
    path.write_text(
        "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n"
        "01/01/2007;00:00:00;1.500;0.100;240.0;6.0;0.0;0.0;0.0\n"
        "01/01/2007;00:01:00;2.500;0.200;241.0;10.0;1.0;0.0;0.0\n",
        encoding="utf-8",
    )
    frame = load_real_measurements(path)
    assert "active_power_kw" in frame
    assert "reactive_power_kvar" in frame
    assert "voltage_v" in frame
    assert "global_intensity_a" in frame
    assert "sub_metering_3_wh" in frame
    assert frame.iloc[1]["active_power_kw"] == pytest.approx(2.5)


def test_resampling_aggregates_each_measurement_without_losing_columns():
    index = pd.date_range("2024-01-01", periods=4, freq="30min", name="timestamp")
    frame = pd.DataFrame({
        "active_power_kw": [1.0, 3.0, 2.0, 4.0],
        "voltage_v": [230.0, 240.0, 232.0, 242.0],
    }, index=index)
    hourly = resample_measurements(frame, "h")
    assert list(hourly.columns) == ["active_power_kw", "voltage_v"]
    assert hourly["active_power_kw"].tolist() == pytest.approx([2.0, 3.0])


def test_pearson_and_spearman_detect_positive_monotonic_relationship():
    frame = pd.DataFrame({
        "active_power_kw": [1.0, 2.0, 3.0, 4.0, 5.0],
        "global_intensity_a": [2.0, 4.0, 6.0, 8.0, 10.0],
        "voltage_v": [240.0, 239.0, 241.0, 238.0, 242.0],
    })
    result = correlation_payload(frame, "hourly", "real")
    assert result["available"] is True
    assert result["pearson"]["active_power_kw"]["global_intensity_a"] == pytest.approx(1.0)
    assert result["spearman"]["active_power_kw"]["global_intensity_a"] == pytest.approx(1.0)
    assert result["target_ranking"][0]["variable"] == "global_intensity_a"


def test_single_variable_synthetic_data_reports_correlation_unavailable():
    frame = pd.DataFrame({"active_power_kw": [1.0, 2.0, 3.0]})
    result = correlation_payload(frame, "hourly", "synthetic")
    assert result["available"] is False
    assert result["source_type"] == "synthetic"
    assert "alongside active power" in result["message"]
