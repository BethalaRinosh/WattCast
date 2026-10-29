#!/usr/bin/env python3
"""Create cleaned hourly/daily energy series and a dataset inspection report."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from .data_processing import (
        clean_and_resample,
        discover_raw_dataset,
        inspect_dataset,
        load_dataset,
        write_processed_outputs,
    )
except ImportError:  # Support direct execution as `python scripts/preprocess_data.py`.
    from data_processing import (
        clean_and_resample,
        discover_raw_dataset,
        inspect_dataset,
        load_dataset,
        write_processed_outputs,
    )

LOGGER = logging.getLogger("wattcast.preprocess")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "backend" / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "backend" / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional explicit input file path.")
    parser.add_argument(
        "--source-type",
        choices=("real", "synthetic"),
        help="Source type when --input is provided (auto-detected for default input).",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.input:
        input_path = args.input.resolve()
        source_type = args.source_type or (
            "synthetic" if input_path.name == "synthetic_hourly_energy.csv" else "real"
        )
    else:
        input_path, source_type = discover_raw_dataset(args.raw_dir)

    LOGGER.info("Loading %s dataset from %s", source_type, input_path)
    frame = load_dataset(input_path, source_type=source_type)
    inspection = inspect_dataset(frame, source_type)
    LOGGER.info("Input inspection: %s", json.dumps(inspection, indent=2))

    hourly, daily, report = clean_and_resample(frame, source_type)
    report["input_inspection"] = inspection
    write_processed_outputs(hourly, daily, report, args.output_dir)

    LOGGER.info("Hourly series: %s rows", len(hourly))
    LOGGER.info("Daily series: %s rows", len(daily))
    LOGGER.info("Saved processed outputs to %s", args.output_dir.resolve())
    LOGGER.info("Source type: %s", source_type.upper())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
