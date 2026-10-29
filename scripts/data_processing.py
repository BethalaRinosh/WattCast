"""Load, inspect, clean and resample WattCast energy datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UCI_FILENAME = "household_power_consumption.txt"
SYNTHETIC_FILENAME = "synthetic_hourly_energy.csv"
UCI_POWER_COLUMN = "Global_active_power"
SYNTHETIC_POWER_COLUMN = "global_active_power_kw"


def discover_raw_dataset(raw_dir: Path) -> tuple[Path, str]:
    """Choose the available input file and return its explicit source type."""
    raw_dir = Path(raw_dir)
    real_path = raw_dir / UCI_FILENAME
    synthetic_path = raw_dir / SYNTHETIC_FILENAME
    if real_path.exists():
        return real_path, "real"
    if synthetic_path.exists():
        return synthetic_path, "synthetic"
    raise FileNotFoundError(
        f"No input dataset found in {raw_dir}. Run scripts/download_data.py first."
    )


def load_dataset(path: Path, source_type: str | None = None) -> pd.DataFrame:
    """Load UCI semicolon TXT or WattCast synthetic CSV into a common schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    if source_type is None:
        source_type = "synthetic" if path.name == SYNTHETIC_FILENAME else "real"

    if source_type == "real":
        frame = pd.read_csv(
            path,
            sep=";",
            na_values=["?", ""],
            low_memory=False,
            dtype={"Date": "string", "Time": "string"},
        )
        expected = {"Date", "Time", UCI_POWER_COLUMN}
        missing = expected.difference(frame.columns)
        if missing:
            raise ValueError(f"UCI dataset is missing expected columns: {sorted(missing)}")
        timestamps = pd.to_datetime(
            frame["Date"].str.strip() + " " + frame["Time"].str.strip(),
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        power = pd.to_numeric(frame[UCI_POWER_COLUMN], errors="coerce")
        result = pd.DataFrame({"active_power_kw": power.to_numpy()}, index=timestamps)
    elif source_type == "synthetic":
        frame = pd.read_csv(path, low_memory=False)
        if "timestamp" not in frame.columns or SYNTHETIC_POWER_COLUMN not in frame.columns:
            raise ValueError(
                "Synthetic CSV must contain 'timestamp' and "
                f"'{SYNTHETIC_POWER_COLUMN}' columns."
            )
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        power = pd.to_numeric(frame[SYNTHETIC_POWER_COLUMN], errors="coerce")
        result = pd.DataFrame({"active_power_kw": power.to_numpy()}, index=timestamps)
    else:
        raise ValueError("source_type must be either 'real' or 'synthetic'.")

    result.index.name = "timestamp"
    result["active_power_kw"] = result["active_power_kw"].replace([np.inf, -np.inf], np.nan)
    result = result.loc[~result.index.isna()].sort_index()
    result = result.loc[~result.index.duplicated(keep="first")]
    return result


def inspect_dataset(frame: pd.DataFrame, source_type: str) -> dict[str, Any]:
    """Summarise dimensions, types, date range, frequency and missing values."""
    if frame.empty:
        raise ValueError("The loaded dataset contains no valid timestamp rows.")
    differences = frame.index.to_series().diff().dropna()
    frequency = "irregular/unknown"
    if not differences.empty:
        common_delta = differences.mode().iloc[0]
        if common_delta == pd.Timedelta(hours=1):
            frequency = "hourly"
        elif common_delta == pd.Timedelta(minutes=1):
            frequency = "minutely"
        elif common_delta == pd.Timedelta(days=1):
            frequency = "daily"
    return {
        "source_type": source_type,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "start": frame.index.min().isoformat(),
        "end": frame.index.max().isoformat(),
        "inferred_frequency": frequency,
        "missing_values": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
    }


def _outlier_summary(series: pd.Series) -> dict[str, float | int | None]:
    """Count IQR outliers without deleting genuine demand peaks."""
    values = series.dropna()
    if values.empty:
        return {"count": 0, "lower_bound": None, "upper_bound": None}
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return {
        "count": int(((values < lower) | (values > upper)).sum()),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
    }


def _resample_power(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample mean power, interpolate short gaps, and drop long missing gaps."""
    resampled = frame["active_power_kw"].resample(rule).mean().to_frame()
    # Only bridge short gaps. Long gaps are dropped rather than fabricating a
    # long run of measurements. Time interpolation respects timestamp spacing.
    resampled["active_power_kw"] = resampled["active_power_kw"].interpolate(
        method="time", limit=3, limit_area="inside"
    )
    resampled = resampled.dropna()
    resampled.index.name = "timestamp"
    return resampled


def clean_and_resample(
    frame: pd.DataFrame, source_type: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Clean the common power series and produce hourly and daily mean-power data.

    Invalid and non-finite values are treated as missing. Non-positive values are
    retained if present in real data rather than silently deleted; outliers are
    reported with an IQR rule but not removed because peaks can be meaningful.
    """
    if frame.empty:
        raise ValueError("Cannot preprocess an empty dataset.")
    clean = frame.copy()
    clean.index = pd.to_datetime(clean.index, errors="coerce")
    clean = clean.loc[~clean.index.isna()].sort_index()
    clean = clean.loc[~clean.index.duplicated(keep="first")]
    clean["active_power_kw"] = pd.to_numeric(
        clean["active_power_kw"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    before = int(len(clean))
    missing_before = int(clean["active_power_kw"].isna().sum())
    outliers = _outlier_summary(clean["active_power_kw"])

    hourly = _resample_power(clean, "h")
    daily = _resample_power(clean, "D")
    if hourly.empty:
        raise ValueError("No hourly observations remain after cleaning.")
    if daily.empty:
        raise ValueError("No daily observations remain after cleaning.")

    report: dict[str, Any] = {
        "source_type": source_type,
        "rows_before_cleaning": before,
        "missing_power_before_resampling": missing_before,
        "iqr_outliers_detected": outliers,
        "hourly_rows": int(len(hourly)),
        "daily_rows": int(len(daily)),
        "hourly_missing_after_cleaning": int(hourly["active_power_kw"].isna().sum()),
        "daily_missing_after_cleaning": int(daily["active_power_kw"].isna().sum()),
        "hourly_start": hourly.index.min().isoformat(),
        "hourly_end": hourly.index.max().isoformat(),
        "daily_start": daily.index.min().isoformat(),
        "daily_end": daily.index.max().isoformat(),
        "aggregation": (
            "Mean active power in kW per hour/day. For hourly intervals, mean kW "
            "is numerically equivalent to approximate kWh consumed during that hour."
        ),
        "missing_value_policy": (
            "Resample with mean; interpolate at most three consecutive missing "
            "resampled bins using time interpolation; drop remaining missing bins."
        ),
        "outlier_policy": (
            "Count IQR outliers for inspection but retain them to avoid deleting "
            "potentially genuine demand peaks."
        ),
    }
    return hourly, daily, report


def write_processed_outputs(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    report: dict[str, Any],
    output_dir: Path,
) -> None:
    """Persist cleaned series and an inspection report for later pipeline steps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_dir / "hourly_energy.csv", index_label="timestamp")
    daily.to_csv(output_dir / "daily_energy.csv", index_label="timestamp")
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
