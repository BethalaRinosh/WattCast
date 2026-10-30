"""Run Pearson and Spearman correlation analysis on household power measurements.

The real UCI source is re-read at minute resolution and resampled to hourly and
daily means. Synthetic input is explicitly marked as limited because it only
contains the active-power target; no artificial correlated features are added.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "backend" / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT / "backend" / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "backend" / "reports" / "correlation"

UCI_COLUMNS = {
    "Global_active_power": "active_power_kw",
    "Global_reactive_power": "reactive_power_kvar",
    "Voltage": "voltage_v",
    "Global_intensity": "global_intensity_a",
    "Sub_metering_1": "sub_metering_1_wh",
    "Sub_metering_2": "sub_metering_2_wh",
    "Sub_metering_3": "sub_metering_3_wh",
}
TARGET = "active_power_kw"
LOGGER = logging.getLogger("wattcast.correlation")


def load_real_measurements(path: Path) -> pd.DataFrame:
    """Load numeric UCI measurements while retaining all correlation variables."""
    frame = pd.read_csv(path, sep=";", na_values=["?", ""], low_memory=False,
                        dtype={"Date": "string", "Time": "string"})
    required = {"Date", "Time", *UCI_COLUMNS.keys()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"UCI dataset is missing required columns: {missing}")
    timestamps = pd.to_datetime(
        frame["Date"].str.strip() + " " + frame["Time"].str.strip(),
        format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    values = frame[list(UCI_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    values = values.rename(columns=UCI_COLUMNS)
    values.index = timestamps
    values.index.name = "timestamp"
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.loc[~values.index.isna()].sort_index()
    values = values.loc[~values.index.duplicated(keep="first")]
    return values


def resample_measurements(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate measurements using interval means; leave unavailable features null."""
    if frame.empty:
        raise ValueError("No valid timestamped measurements are available.")
    result = frame.resample(rule).mean()
    return result.dropna(subset=[TARGET])


def correlation_payload(frame: pd.DataFrame, series: str, source_type: str) -> dict[str, Any]:
    """Calculate pairwise Pearson/Spearman matrices and target correlation rankings."""
    numeric = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    available = [column for column in numeric.columns if numeric[column].notna().sum() >= 3
                 and numeric[column].nunique(dropna=True) > 1]
    if TARGET not in available:
        return {
            "series": series, "source_type": source_type, "available": False,
            "message": "Correlation analysis requires at least one non-constant numeric measurement alongside active power.",
            "variables": available, "observations": 0,
        }
    numeric = numeric[available]
    pearson = numeric.corr(method="pearson", min_periods=3)
    spearman = numeric.corr(method="spearman", min_periods=3)
    ranking = []
    for variable in available:
        if variable == TARGET:
            continue
        p = pearson.loc[TARGET, variable]
        s = spearman.loc[TARGET, variable]
        ranking.append({
            "variable": variable,
            "pearson": None if pd.isna(p) else float(p),
            "spearman": None if pd.isna(s) else float(s),
            "paired_observations": int(numeric[[TARGET, variable]].dropna().shape[0]),
        })
    ranking.sort(key=lambda item: abs(item["pearson"]) if item["pearson"] is not None else -1,
                 reverse=True)
    return {
        "series": series,
        "source_type": source_type,
        "available": True,
        "method": "Pairwise complete observations; interval means; no imputation across missing feature values.",
        "observations": int(len(numeric)),
        "variables": available,
        "pearson": pearson.astype(object).where(pd.notna(pearson), None).to_dict(),
        "spearman": spearman.astype(object).where(pd.notna(spearman), None).to_dict(),
        "target_ranking": ranking,
        "interpretation": "Correlation describes linear (Pearson) or monotonic rank (Spearman) association. It does not establish causation.",
    }


def _save_heatmap(matrix: dict[str, dict[str, float | None]], title: str, path: Path) -> None:
    frame = pd.DataFrame(matrix, dtype=float)
    if frame.empty:
        return
    width = max(8, 1.25 * len(frame.columns))
    height = max(6, 1.0 * len(frame.index))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(frame.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(frame.index)), frame.index)
    for row in range(frame.shape[0]):
        for col in range(frame.shape[1]):
            value = frame.iloc[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Correlation coefficient", fraction=0.04, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_analysis(raw_dir: Path = DEFAULT_RAW_DIR,
                 processed_dir: Path = DEFAULT_PROCESSED_DIR,
                 output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Create hourly/daily correlation reports and heatmaps."""
    raw_path = Path(raw_dir) / "household_power_consumption.txt"
    synthetic_path = Path(raw_dir) / "synthetic_hourly_energy.csv"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if raw_path.is_file():
        measurements = load_real_measurements(raw_path)
        source_type = "real"
        hourly = resample_measurements(measurements, "h")
        daily = resample_measurements(measurements, "D")
    elif synthetic_path.is_file():
        source_type = "synthetic"
        processed = pd.read_csv(Path(processed_dir) / "hourly_energy.csv")
        if "active_power_kw" not in processed:
            raise ValueError("Processed synthetic series is missing active_power_kw.")
        hourly = processed[["active_power_kw"]].copy()
        daily = pd.read_csv(Path(processed_dir) / "daily_energy.csv")[["active_power_kw"]].copy()
        # One-variable synthetic input cannot support cross-variable correlation.
    else:
        raise FileNotFoundError(
            f"No UCI or synthetic input found in {raw_dir}. Run scripts/download_data.py first."
        )

    results: dict[str, Any] = {
        "project": "WattCast", "source_type": source_type,
        "source_label": "REAL UCI DATA" if source_type == "real" else "SYNTHETIC SIMULATED DATA",
        "variables_note": "UCI measurements are preserved from the original source." if source_type == "real"
                          else "Synthetic input currently contains only active power; no extra variables were fabricated.",
        "series": {},
    }
    for name, frame in (("hourly", hourly), ("daily", daily)):
        payload = correlation_payload(frame, name, source_type)
        results["series"][name] = payload
        if payload["available"]:
            for method in ("pearson", "spearman"):
                _save_heatmap(payload[method],
                              f"{name.title()} {method.title()} correlation · {results['source_label']}",
                              output_dir / f"{name}_{method}_heatmap.png")
    (output_dir / "correlation_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved correlation reports to %s", output_dir.resolve())
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_analysis(args.raw_dir, args.processed_dir, args.output_dir)
    for name, item in result["series"].items():
        LOGGER.info("%s: available=%s, observations=%s, variables=%s",
                    name, item["available"], item["observations"], item["variables"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
