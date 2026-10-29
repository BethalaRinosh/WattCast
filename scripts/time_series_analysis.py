#!/usr/bin/env python3
"""Run time-series decomposition, stationarity tests, and ACF/PACF analysis."""
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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

LOGGER = logging.getLogger("wattcast.time_series_analysis")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = ROOT / "backend" / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "backend" / "reports" / "time_series"
TARGET = "active_power_kw"


def read_series(path: Path) -> pd.Series:
    """Read a processed series and enforce a clean, ordered datetime index."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. Run scripts/preprocess_data.py first."
        )
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    missing = {"timestamp", TARGET}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["timestamp", TARGET])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    if frame.empty:
        raise ValueError(f"No valid observations found in {path}.")
    series = frame.set_index("timestamp")[TARGET].astype(float).sort_index()
    if len(series) < 3:
        raise ValueError(f"At least 3 valid observations are required in {path}.")
    return series


def _stationarity_result(series: pd.Series, label: str) -> dict[str, Any]:
    """Run ADF and KPSS; include interpretation based on the conventional 5% level."""
    values = np.asarray(series.dropna(), dtype=float)
    if len(values) < 10 or np.ptp(values) == 0:
        return {
            "series": label,
            "observations": int(len(values)),
            "adf": None,
            "kpss": None,
            "note": "Insufficient observations or constant values for reliable tests.",
        }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adf_stat, adf_p, adf_lags, adf_n, adf_critical, _ = adfuller(
            values, regression="c", autolag="AIC"
        )
        kpss_stat, kpss_p, kpss_lags, kpss_critical = kpss(
            values, regression="c", nlags="auto"
        )
    return {
        "series": label,
        "observations": int(len(values)),
        "adf": {
            "statistic": float(adf_stat),
            "p_value": float(adf_p),
            "used_lags": int(adf_lags),
            "n_observations": int(adf_n),
            "critical_values": {key: float(value) for key, value in adf_critical.items()},
            "null_hypothesis": "The series has a unit root (non-stationary).",
            "interpretation_5_percent": (
                "Reject unit-root null; evidence supports stationarity."
                if adf_p < 0.05 else
                "Do not reject unit-root null; evidence does not establish stationarity."
            ),
        },
        "kpss": {
            "statistic": float(kpss_stat),
            "p_value": float(kpss_p),
            "used_lags": int(kpss_lags),
            "critical_values": {key: float(value) for key, value in kpss_critical.items()},
            "null_hypothesis": "The series is level-stationary.",
            "interpretation_5_percent": (
                "Reject stationarity null; evidence suggests non-stationarity."
                if kpss_p < 0.05 else
                "Do not reject stationarity null; evidence is compatible with stationarity."
            ),
        },
        "warnings": [str(item.message) for item in caught],
        "note": "ADF and KPSS have opposite null hypotheses; interpret them together.",
    }


def _decompose(
    series: pd.Series, period: int, label: str, output_dir: Path, source_label: str
) -> dict[str, Any]:
    """Create additive decomposition components and save a four-panel chart."""
    if len(series) < 2 * period:
        raise ValueError(
            f"{label} decomposition needs at least {2 * period} observations "
            f"for period={period}; got {len(series)}."
        )
    result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    components = pd.DataFrame({
        "timestamp": series.index,
        "observed": result.observed.to_numpy(),
        "trend": result.trend.to_numpy(),
        "seasonal": result.seasonal.to_numpy(),
        "residual": result.resid.to_numpy(),
    })
    csv_name = f"{label}_decomposition.csv"
    components.to_csv(output_dir / csv_name, index=False)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for axis, column, title in zip(
        axes,
        ("observed", "trend", "seasonal", "residual"),
        ("Observed", "Trend", f"Seasonal component (period={period})", "Residual"),
    ):
        axis.plot(components["timestamp"], components[column], linewidth=0.65)
        axis.set_ylabel("kW")
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
    fig.suptitle(f"{label.title()} additive decomposition | {source_label}")
    plot_name = f"{label}_decomposition.png"
    fig.savefig(output_dir / plot_name, dpi=140)
    plt.close(fig)
    return {
        "period": period,
        "model": "additive",
        "components_csv": csv_name,
        "plot": plot_name,
        "residual_missing_values": int(components["residual"].isna().sum()),
        "interpretation_note": (
            "The decomposition separates observed values into estimated trend, "
            "repeating seasonal component and residual. It is descriptive, not a forecast."
        ),
    }


def _acf_pacf(series: pd.Series, output_dir: Path, source_label: str) -> dict[str, Any]:
    """Save ACF/PACF values and a diagnostic plot for the daily series."""
    clean = series.dropna().astype(float)
    max_lags = min(60, len(clean) // 2 - 1)
    if max_lags < 1:
        raise ValueError("Not enough daily observations for ACF/PACF analysis.")

    from statsmodels.tsa.stattools import acf, pacf

    acf_values = acf(clean.to_numpy(), nlags=max_lags, fft=True)
    pacf_values = pacf(clean.to_numpy(), nlags=max_lags, method="ywm")
    pd.DataFrame({
        "lag_days": np.arange(max_lags + 1),
        "acf": acf_values,
        "pacf": pacf_values,
    }).to_csv(output_dir / "daily_acf_pacf.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    plot_acf(clean, lags=max_lags, ax=axes[0], zero=True)
    axes[0].set_title("Daily series autocorrelation (ACF)")
    plot_pacf(clean, lags=max_lags, method="ywm", ax=axes[1], zero=True)
    axes[1].set_title("Daily series partial autocorrelation (PACF)")
    fig.suptitle(f"Daily ACF/PACF | {source_label}")
    fig.savefig(output_dir / "daily_acf_pacf.png", dpi=140)
    plt.close(fig)
    return {
        "series": "daily mean active power",
        "max_lag_days": int(max_lags),
        "values_csv": "daily_acf_pacf.csv",
        "plot": "daily_acf_pacf.png",
        "interpretation_note": (
            "ACF measures correlation with lagged values; PACF estimates the direct "
            "association at each lag after accounting for shorter lags. Spikes suggest "
            "candidate dependencies, not proof that a model will forecast well."
        ),
    }


def run_time_series_analysis(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run hourly/daily decomposition plus daily stationarity and ACF/PACF diagnostics."""
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    hourly = read_series(processed_dir / "hourly_energy.csv")
    daily = read_series(processed_dir / "daily_energy.csv")
    report_path = processed_dir / "preprocessing_report.json"
    source_type = "unknown"
    if report_path.exists():
        source_type = json.loads(report_path.read_text(encoding="utf-8")).get("source_type", "unknown")
    source_label = (
        "REAL UCI DATA" if source_type == "real"
        else "SYNTHETIC SIMULATED DATA" if source_type == "synthetic"
        else "SOURCE TYPE UNKNOWN"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "project": "WattCast",
        "source_type": source_type,
        "source_label": source_label,
        "target": "Mean active power (kW)",
        "decomposition": {
            "hourly": _decompose(hourly, 24, "hourly", output_dir, source_label),
            "daily": _decompose(daily, 7, "daily", output_dir, source_label),
        },
        "stationarity": [
            _stationarity_result(daily, "daily_level"),
            _stationarity_result(daily.diff().dropna(), "daily_first_difference"),
        ],
        "acf_pacf": _acf_pacf(daily, output_dir, source_label),
        "method_notes": [
            "Additive decomposition is used as a first diagnostic and assumes additive seasonal effects.",
            "Stationarity tests are run on daily values and their first difference.",
            "ADF null: unit root; KPSS null: level stationarity. Their results should be interpreted together.",
            "ACF/PACF are calculated on daily values through the maximum lag reported in this file.",
            "Diagnostics guide later model selection but do not establish out-of-sample forecast accuracy.",
        ],
    }
    (output_dir / "time_series_analysis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Time-series diagnostics complete for source=%s; outputs saved to %s",
        source_type, output_dir
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    summary = run_time_series_analysis(args.processed_dir, args.output_dir)
    for item in summary["stationarity"]:
        adf = item.get("adf")
        kpss_result = item.get("kpss")
        LOGGER.info(
            "%s: ADF p=%s; KPSS p=%s",
            item["series"],
            f"{adf['p_value']:.5g}" if adf else "n/a",
            f"{kpss_result['p_value']:.5g}" if kpss_result else "n/a",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
