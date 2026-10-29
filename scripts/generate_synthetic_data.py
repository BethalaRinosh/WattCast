#!/usr/bin/env python3
"""Generate a reproducible, clearly labelled synthetic hourly energy dataset."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("wattcast.synthetic")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "backend" / "data" / "raw" / "synthetic_hourly_energy.csv"
)


def generate_synthetic_dataset(
    output_path: Path,
    years: int = 4,
    seed: int = 42,
    start: str = "2010-01-01 00:00:00",
) -> Path:
    """Write synthetic hourly power readings with trend and multiple seasonal cycles.

    The target column, global_active_power_kw, represents simulated average power
    in kW during each hour. This is not real measured data.
    """
    if years < 1 or years > 20:
        raise ValueError("years must be between 1 and 20.")

    rng = np.random.default_rng(seed)
    periods = years * 365 * 24
    timestamps = pd.date_range(start=start, periods=periods, freq="h")
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    elapsed_days = np.arange(periods) / 24.0

    # Smooth components imitate typical household demand; random noise prevents
    # the series from being perfectly periodic or unrealistically deterministic.
    daily_cycle = 0.42 * np.sin(2 * np.pi * (hour - 7) / 24)
    evening_peak = 0.55 * np.exp(-0.5 * ((hour - 19) / 2.2) ** 2)
    morning_peak = 0.22 * np.exp(-0.5 * ((hour - 7) / 1.8) ** 2)
    weekend_effect = np.where(weekday >= 5, 0.12, -0.02)
    yearly_cycle = 0.18 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    weekly_cycle = 0.08 * np.sin(2 * np.pi * np.arange(periods) / (24 * 7))
    gradual_trend = 0.00008 * elapsed_days
    noise = rng.normal(loc=0.0, scale=0.13, size=periods)

    power_kw = (
        1.15
        + daily_cycle
        + evening_peak
        + morning_peak
        + weekend_effect
        + yearly_cycle
        + weekly_cycle
        + gradual_trend
        + noise
    )
    power_kw = np.clip(power_kw, 0.08, None)
    reactive_power_kvar = np.clip(
        0.22 + 0.08 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.025, periods),
        0.02,
        None,
    )
    voltage = 240 + 3.0 * np.sin(2 * np.pi * np.arange(periods) / (24 * 30)) + rng.normal(0, 1.2, periods)
    intensity_amp = power_kw * 1000 / np.maximum(voltage, 1.0)
    sub_metering_1 = np.clip(35 + 25 * (hour >= 17) + rng.normal(0, 8, periods), 0, None)
    sub_metering_2 = np.clip(30 + 18 * ((hour >= 6) & (hour <= 9)) + rng.normal(0, 7, periods), 0, None)
    sub_metering_3 = np.clip(70 + 20 * ((hour <= 6) | (hour >= 20)) + rng.normal(0, 10, periods), 0, None)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "global_active_power_kw": np.round(power_kw, 4),
            "global_reactive_power_kvar": np.round(reactive_power_kvar, 4),
            "voltage_v": np.round(voltage, 2),
            "global_intensity_a": np.round(intensity_amp, 3),
            "sub_metering_1_wh": np.round(sub_metering_1, 2),
            "sub_metering_2_wh": np.round(sub_metering_2, 2),
            "sub_metering_3_wh": np.round(sub_metering_3, 2),
            "data_source": "synthetic",
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    LOGGER.info(
        "Generated %s hourly rows (%s to %s) with seed=%d; source=SYNTHETIC.",
        len(frame),
        timestamps.min(),
        timestamps.max(),
        seed,
    )
    return output_path


def write_metadata(
    path: Path,
    *,
    source_type: str,
    source_name: str,
    source_url: str | None,
    data_file: str,
    note: str,
    seed: int | None = None,
    years: int | None = None,
    generated_at_utc: str | None = None,
) -> Path:
    """Write machine-readable provenance so downstream code can identify the data."""
    metadata: dict[str, Any] = {
        "project": "WattCast",
        "source_type": source_type,
        "source_name": source_name,
        "source_url": source_url,
        "data_file": data_file,
        "note": note,
    }
    if seed is not None:
        metadata["random_seed"] = seed
    if years is not None:
        metadata["synthetic_years"] = years
    if generated_at_utc is not None:
        metadata["generated_at_utc"] = generated_at_utc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--years", type=int, default=4, help="Number of synthetic years (1–20).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    output_path = generate_synthetic_dataset(args.output, years=args.years, seed=args.seed)
    write_metadata(
        output_path.with_name("dataset_metadata.json"),
        source_type="synthetic",
        source_name="WattCast deterministic synthetic hourly energy data",
        source_url=None,
        data_file=output_path.name,
        note="Synthetic simulation only; not observed real-world measurements.",
        seed=args.seed,
        years=args.years,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
