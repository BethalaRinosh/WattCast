#!/usr/bin/env python3
"""Create reproducible exploratory summaries and charts for WattCast."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger("wattcast.eda")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = ROOT / "backend" / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "backend" / "reports" / "eda"


def read_series(path: Path) -> pd.Series:
    """Read a processed WattCast CSV and validate its timestamp/target columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. Run scripts/preprocess_data.py first."
        )
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "active_power_kw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    frame["active_power_kw"] = pd.to_numeric(frame["active_power_kw"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "active_power_kw"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    if frame.empty:
        raise ValueError(f"No valid observations found in {path}.")
    return frame.set_index("timestamp")["active_power_kw"].astype(float).sort_index()


def _describe(series: pd.Series) -> dict[str, Any]:
    """Return JSON-safe descriptive statistics for a numeric time series."""
    stats = series.describe()
    return {
        "count": int(stats["count"]),
        "mean_kw": float(stats["mean"]),
        "std_kw": float(stats["std"]) if pd.notna(stats["std"]) else None,
        "min_kw": float(stats["min"]),
        "q25_kw": float(stats["25%"]),
        "median_kw": float(stats["50%"]),
        "q75_kw": float(stats["75%"]),
        "max_kw": float(stats["max"]),
        "start": series.index.min().isoformat(),
        "end": series.index.max().isoformat(),
    }


def build_eda_summary(
    hourly: pd.Series, daily: pd.Series, source_type: str
) -> dict[str, Any]:
    """Summarise distribution, intraday/weekly patterns and monthly means."""
    hourly_profile = hourly.groupby(hourly.index.hour).mean()
    weekday_profile = hourly.groupby(hourly.index.dayofweek).mean().reindex(range(7))
    monthly_profile = daily.groupby(daily.index.month).mean().reindex(range(1, 13))
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def profile_dict(profile: pd.Series, labels: list[str]) -> dict[str, float | None]:
        return {
            label: (float(profile.loc[index]) if index in profile.index and pd.notna(profile.loc[index]) else None)
            for index, label in enumerate(labels)
        }

    valid_weekdays = weekday_profile.dropna()
    valid_hours = hourly_profile.dropna()
    return {
        "project": "WattCast",
        "source_type": source_type,
        "source_label": (
            "SYNTHETIC SIMULATED DATA" if source_type == "synthetic"
            else "REAL UCI DATA" if source_type == "real"
            else "SOURCE TYPE UNKNOWN"
        ),
        "target": "Mean active power (kW)",
        "hourly": _describe(hourly),
        "daily": _describe(daily),
        "intraday_mean_power_kw": profile_dict(hourly_profile, [f"{hour:02d}:00" for hour in range(24)]),
        "weekday_mean_power_kw": profile_dict(weekday_profile, weekday_names),
        "monthly_mean_power_kw": {
            str(month): (float(monthly_profile.loc[month]) if pd.notna(monthly_profile.loc[month]) else None)
            for month in range(1, 13)
        },
        "peak_mean_hour": int(valid_hours.idxmax()) if not valid_hours.empty else None,
        "lowest_mean_hour": int(valid_hours.idxmin()) if not valid_hours.empty else None,
        "highest_mean_weekday": weekday_names[int(valid_weekdays.idxmax())] if not valid_weekdays.empty else None,
        "lowest_mean_weekday": weekday_names[int(valid_weekdays.idxmin())] if not valid_weekdays.empty else None,
        "interpretation_note": (
            "These are descriptive patterns, not causal explanations. "
            "If source_type is synthetic, all patterns describe simulated data only."
        ),
    }


def _save_line_plot(
    series: pd.Series, path: Path, title: str, ylabel: str, rolling_window: int | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.plot(series.index, series.values, linewidth=0.9, label="Observed series")
    if rolling_window:
        rolling = series.rolling(rolling_window, min_periods=rolling_window).mean()
        ax.plot(rolling.index, rolling.values, linewidth=1.8, label=f"{rolling_window}-day rolling mean")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_eda(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Generate EDA JSON, pattern tables, and four labelled charts."""
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    hourly = read_series(processed_dir / "hourly_energy.csv")
    daily = read_series(processed_dir / "daily_energy.csv")
    preprocessing_report_path = processed_dir / "preprocessing_report.json"
    source_type = "unknown"
    if preprocessing_report_path.exists():
        report = json.loads(preprocessing_report_path.read_text(encoding="utf-8"))
        source_type = report.get("source_type", "unknown")
    if source_type not in {"real", "synthetic"}:
        source_type = "unknown"

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_eda_summary(hourly, daily, source_type)
    summary["artifacts"] = {
        "summary": "eda_summary.json",
        "intraday_profile": "intraday_profile.csv",
        "weekday_profile": "weekday_profile.csv",
        "monthly_profile": "monthly_profile.csv",
        "hourly_series_plot": "hourly_series.png",
        "daily_series_plot": "daily_series.png",
        "intraday_profile_plot": "intraday_profile.png",
        "weekday_profile_plot": "weekday_profile.png",
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    hourly.groupby(hourly.index.hour).mean().rename("mean_active_power_kw").rename_axis("hour").to_csv(
        output_dir / "intraday_profile.csv"
    )
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_profile = hourly.groupby(hourly.index.dayofweek).mean().reindex(range(7))
    weekday_profile.rename("mean_active_power_kw").rename_axis("weekday_number").to_csv(
        output_dir / "weekday_profile.csv"
    )
    daily.groupby(daily.index.month).mean().reindex(range(1, 13)).rename(
        "mean_active_power_kw"
    ).rename_axis("month_number").to_csv(output_dir / "monthly_profile.csv")

    source_label = summary["source_label"]
    _save_line_plot(
        hourly, output_dir / "hourly_series.png",
        f"Hourly mean active power | {source_label}", "Mean active power (kW)"
    )
    _save_line_plot(
        daily, output_dir / "daily_series.png",
        f"Daily mean active power with 7-day rolling mean | {source_label}",
        "Mean active power (kW)", rolling_window=7
    )

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    intraday = hourly.groupby(hourly.index.hour).mean()
    ax.plot(intraday.index, intraday.values, marker="o", linewidth=1.6)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean active power (kW)")
    ax.set_title(f"Average intraday profile | {source_label}")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_dir / "intraday_profile.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    plot_weekday = weekday_profile.dropna()
    ax.bar([weekday_names[int(day)] for day in plot_weekday.index], plot_weekday.values)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Mean active power (kW)")
    ax.set_title(f"Average weekday profile | {source_label}")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_dir / "weekday_profile.png", dpi=140)
    plt.close(fig)

    LOGGER.info("EDA complete for source=%s; outputs saved to %s", source_type, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    summary = run_eda(args.processed_dir, args.output_dir)
    LOGGER.info("Source: %s", summary["source_label"])
    LOGGER.info("Hourly mean: %.3f kW", summary["hourly"]["mean_kw"])
    LOGGER.info("Peak average hour: %s", summary["peak_mean_hour"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
