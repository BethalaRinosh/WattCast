"""Offline tests for reproducible data acquisition and source provenance."""
from __future__ import annotations

import io
import json
import zipfile

import pandas as pd

from scripts import download_data
from scripts.generate_synthetic_data import generate_synthetic_dataset


def _zip_bytes(filename: str, payload: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, payload)
        archive.writestr("unrelated/readme.txt", "ignore this file")
    return buffer.getvalue()


class _FakeResponse(io.BytesIO):
    """BytesIO response that supports urllib's context-manager usage."""


def test_synthetic_generator_is_reproducible_and_labelled(tmp_path):
    first = generate_synthetic_dataset(tmp_path / "first.csv", years=1, seed=17)
    second = generate_synthetic_dataset(tmp_path / "second.csv", years=1, seed=17)
    first_frame = pd.read_csv(first)
    second_frame = pd.read_csv(second)

    pd.testing.assert_frame_equal(first_frame, second_frame)
    assert len(first_frame) == 365 * 24
    assert set(first_frame["data_source"]) == {"synthetic"}


def test_downloader_extracts_only_expected_dataset_and_writes_metadata(tmp_path, monkeypatch):
    payload = "Date;Time;Global_active_power\n01/01/2007;00:00:00;1.500\n"
    archive_bytes = _zip_bytes("nested/household_power_consumption.txt", payload)
    monkeypatch.setattr(
        download_data.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _FakeResponse(archive_bytes),
    )

    destination = download_data.download_and_extract(tmp_path)

    assert destination.read_text(encoding="utf-8") == payload
    assert not (tmp_path / "unrelated" / "readme.txt").exists()
    assert not (tmp_path / "uci_household_power_consumption.zip").exists()
    metadata = json.loads((tmp_path / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_type"] == "real"
    assert metadata["data_file"] == "household_power_consumption.txt"


def test_failed_forced_refresh_preserves_existing_real_file(tmp_path, monkeypatch):
    real_path = tmp_path / download_data.DATASET_FILENAME
    real_path.write_text("previous valid real dataset", encoding="utf-8")
    monkeypatch.setattr(
        download_data.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _FakeResponse(b"not a zip archive"),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["download_data.py", "--output-dir", str(tmp_path), "--force"],
    )

    assert download_data.main() == 0
    assert real_path.read_text(encoding="utf-8") == "previous valid real dataset"
    assert not (tmp_path / "synthetic_hourly_energy.csv").exists()
    metadata = json.loads((tmp_path / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_type"] == "real"


def test_failed_download_uses_explicitly_labelled_synthetic_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download_data.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "download_data.py",
            "--output-dir",
            str(tmp_path),
            "--synthetic-years",
            "1",
            "--seed",
            "7",
        ],
    )

    assert download_data.main() == 0
    synthetic_path = tmp_path / "synthetic_hourly_energy.csv"
    assert synthetic_path.exists()
    metadata = json.loads((tmp_path / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_type"] == "synthetic"
    assert metadata["random_seed"] == 7
    assert metadata["data_file"] == synthetic_path.name
    assert not (tmp_path / download_data.DATASET_FILENAME).exists()
