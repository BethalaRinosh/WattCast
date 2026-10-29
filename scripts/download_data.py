#!/usr/bin/env python3
"""Download the UCI household electricity dataset, with an explicit fallback.

Run from the repository root:
    python scripts/download_data.py
    python scripts/download_data.py --no-synthetic-fallback
    python scripts/download_data.py --force
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from generate_synthetic_data import generate_synthetic_dataset, write_metadata

LOGGER = logging.getLogger("wattcast.download")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "backend" / "data" / "raw"
UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
DATASET_FILENAME = "household_power_consumption.txt"
METADATA_FILENAME = "dataset_metadata.json"


def download_and_extract(output_dir: Path, force: bool = False) -> Path:
    """Download the official UCI ZIP and safely extract its data text file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / DATASET_FILENAME
    if destination.exists() and not force:
        LOGGER.info("Using existing real dataset: %s", destination)
        return destination

    zip_path = output_dir / "uci_household_power_consumption.zip"
    request = urllib.request.Request(
        UCI_DATASET_URL,
        headers={"User-Agent": "WattCast-Academic-Project/1.0"},
    )
    LOGGER.info("Downloading official UCI dataset...")
    with urllib.request.urlopen(request, timeout=45) as response, zip_path.open("wb") as target:
        shutil.copyfileobj(response, target)

    with zipfile.ZipFile(zip_path) as archive:
        members = [
            member for member in archive.namelist()
            if Path(member).name == DATASET_FILENAME and not member.endswith("/")
        ]
        if not members:
            raise ValueError(
                f"Expected {DATASET_FILENAME!r} was not found in the UCI ZIP."
            )
        # Extract only the expected file to avoid trusting archive paths.
        with archive.open(members[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)

    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("The downloaded dataset file is empty.")

    zip_path.unlink(missing_ok=True)
    write_metadata(
        output_dir / METADATA_FILENAME,
        source_type="real",
        source_name="UCI Individual Household Electric Power Consumption",
        source_url=UCI_DATASET_URL,
        data_file=destination.name,
        note=(
            "Original semicolon-delimited, one-minute readings from one household. "
            "Cleaning and hourly/daily resampling are handled in a later pipeline step."
        ),
    )
    LOGGER.info("Saved real UCI data to %s", destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download WattCast's public dataset or generate a labelled fallback."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for raw data (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing real dataset if a fresh download succeeds.",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Exit with an error if the UCI download fails instead of generating fallback data.",
    )
    parser.add_argument(
        "--synthetic-years",
        type=int,
        default=4,
        help="Number of years for the synthetic hourly fallback (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic data (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the download workflow and clearly report the data source used."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    output_dir = args.output_dir.resolve()

    existing_real = output_dir / DATASET_FILENAME
    if existing_real.exists() and not args.force:
        LOGGER.info("Real dataset already exists; skipping download.")
        return 0

    try:
        download_and_extract(output_dir, force=args.force)
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, TimeoutError) as exc:
        LOGGER.error("UCI download failed: %s", exc)
        if args.no_synthetic_fallback:
            LOGGER.error("Synthetic fallback is disabled; no fallback file was generated.")
            return 1

        LOGGER.warning(
            "FALLBACK ACTIVE: generating SYNTHETIC data. This is simulated data, "
            "not real household measurements."
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        synthetic_path = output_dir / "synthetic_hourly_energy.csv"
        generate_synthetic_dataset(
            output_path=synthetic_path,
            years=args.synthetic_years,
            seed=args.seed,
        )
        write_metadata(
            output_dir / METADATA_FILENAME,
            source_type="synthetic",
            source_name="WattCast deterministic synthetic hourly energy data",
            source_url=None,
            data_file=synthetic_path.name,
            note=(
                "Synthetic fallback generated because the UCI download failed. "
                "Do not describe these values as observed real-world measurements."
            ),
            seed=args.seed,
            years=args.synthetic_years,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        LOGGER.warning("Synthetic fallback saved to %s", synthetic_path)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
