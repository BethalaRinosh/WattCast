"""Tests for WattCast decomposition and time-series diagnostics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.time_series_analysis import (
    _stationarity_result,
    read_series,
    run_time_series_analysis,
)


def _write_processed(processed):
    processed.mkdir(parents=True, exist_ok=True)
    hourly_index = pd.date_range("2024-01-01", periods=24 * 35, freq="h")
    daily_index = pd.date_range("2024-01-01", periods=35, freq="D")
    hourly_values = [
        1.5 + 0.2 * np.sin(2 * np.pi * ts.hour / 24)
        + 0.1 * np.sin(2 * np.pi * ts.dayofweek / 7)
        for ts in hourly_index
    ]
    daily_values = [
        1.5 + 0.1 * np.sin(2 * np.pi * i / 7) + 0.002 * i
        for i in range(len(daily_index))
    ]
    pd.DataFrame({
        "timestamp": hourly_index,
        "active_power_kw": hourly_values,
    }).to_csv(processed / "hourly_energy.csv", index=False)
    pd.DataFrame({
        "timestamp": daily_index,
        "active_power_kw": daily_values,
    }).to_csv(processed / "daily_energy.csv", index=False)
    (processed / "preprocessing_report.json").write_text(
        json.dumps({"source_type": "synthetic"}), encoding="utf-8"
    )


def test_read_series_sorts_and_rejects_invalid_values(tmp_path):
    path = tmp_path / "hourly_energy.csv"
    pd.DataFrame({
        "timestamp": ["2024-01-01 01:00:00", "bad", "2024-01-01 00:00:00"],
        "active_power_kw": [2.0, 9.0, 1.0],
    }).to_csv(path, index=False)
    series = read_series(path)
    assert series.index.is_monotonic_increasing
    assert series.tolist() == [1.0, 2.0]


def test_read_series_missing_file_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess_data.py"):
        read_series(tmp_path / "missing.csv")


def test_stationarity_handles_constant_or_short_series():
    result = _stationarity_result(pd.Series([1.0, 1.0, 1.0]), "constant")
    assert result["adf"] is None
    assert result["kpss"] is None
    assert "Insufficient" in result["note"]


def test_run_analysis_writes_json_csv_and_plots(tmp_path):
    processed = tmp_path / "processed"
    output = tmp_path / "reports"
    _write_processed(processed)
    result = run_time_series_analysis(processed, output)

    assert result["source_label"] == "SYNTHETIC SIMULATED DATA"
    assert result["decomposition"]["hourly"]["period"] == 24
    assert result["decomposition"]["daily"]["period"] == 7
    assert result["acf_pacf"]["max_lag_days"] == 16
    assert len(result["stationarity"]) == 2
    assert (output / "time_series_analysis.json").exists()
    assert (output / "daily_acf_pacf.csv").exists()
    assert (output / "daily_acf_pacf.png").stat().st_size > 0
    for filename in (
        "hourly_decomposition.csv",
        "daily_decomposition.csv",
        "hourly_decomposition.png",
        "daily_decomposition.png",
    ):
        assert (output / filename).stat().st_size > 0
