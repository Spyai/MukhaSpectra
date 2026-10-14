"""Unit tests for input helpers: extension sniffing, neutral names, calibration serialisation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.models import CalibrationRecord
from app.services.inputs import (
    guess_extension,
    image_calibration_from_record,
    image_calibration_to_npz,
    mime_type_for,
    neutral_file_name,
)
from app.services.records import new_pseudonymous_id
from mukha_pipeline.schema.models import CaptureKind
from mukha_pipeline.sites import OralSite
from mukha_pipeline.types import ImageCalibration


@pytest.mark.parametrize(
    ("filename", "data", "expected"),
    [
        ("photo.JPG", b"", ".jpg"),
        ("scan.tiff", b"", ".tiff"),
        (None, b"\x89PNG\r\n\x1a\n....", ".png"),
        ("no_extension", b"\xff\xd8\xff\xe0....", ".jpg"),
        ("x.exe", b"II*\x00....", ".tif"),
        (None, b"MM\x00*....", ".tif"),
        (None, b"\x00\x00\x00\x18ftypheic....", ".heic"),
        (None, b"RIFF\x00\x00\x00\x00WEBPVP8 ", ".webp"),
        (None, b"BM......", ".bmp"),
        ("x.sh", b"#!/bin/sh", ".bin"),
    ],
)
def test_guess_image_extension(filename, data, expected):
    assert guess_extension(filename, data, spectral=False) == expected


@pytest.mark.parametrize(
    ("filename", "data", "expected"),
    [
        ("probe.CSV", b"{", ".csv"),
        ("probe.json", b"a,b", ".json"),
        (None, b'  {"format": 1}', ".json"),
        ("probe.txt", b"reading_id,kind", ".csv"),
    ],
)
def test_guess_spectral_extension(filename, data, expected):
    assert guess_extension(filename, data, spectral=True) == expected


def test_mime_types_and_neutral_names():
    assert mime_type_for(".png") == "image/png"
    assert mime_type_for(".json") == "application/json"
    assert mime_type_for(".bin") == "application/octet-stream"
    assert neutral_file_name(OralSite.TONGUE_DORSUM, CaptureKind.AUTOFLUORESCENCE, ".jpg") == (
        "tongue_dorsum-af.jpg"
    )
    assert neutral_file_name(None, CaptureKind.SPECTRAL, ".csv") == "spectral.csv"


def test_pseudonymous_ids_are_random_and_labelled():
    ids = {new_pseudonymous_id() for _ in range(50)}
    assert len(ids) == 50 and all(i.startswith("P-") and len(i) == 14 for i in ids)
    assert new_pseudonymous_id(synthetic=True).startswith("SYN-")


def test_image_calibration_without_flat_field_round_trips():
    now = datetime(2026, 10, 14, tzinfo=UTC)
    cal = ImageCalibration(
        calibration_id="c1",
        device_id="scout-1",
        created_at=now,
        expires_at=now + timedelta(hours=24),
        flat_field=None,
        color_matrix=np.eye(3) * 0.9,
        af_scale=2.0,
    )
    blob = image_calibration_to_npz(cal)
    record = CalibrationRecord(
        id="r1",
        device_id="scout-1",
        kind="image",
        created_at=now.replace(tzinfo=None),
        expires_at=None,
        synthetic=False,
        data={"calibration_id": "c1", "af_scale": 2.0},
    )
    back = image_calibration_from_record(record, blob)
    assert back.flat_field is None
    assert np.allclose(back.color_matrix, cal.color_matrix)
    assert back.af_scale == 2.0 and back.calibration_id == "c1"
    assert back.created_at == now and back.expires_at is None
