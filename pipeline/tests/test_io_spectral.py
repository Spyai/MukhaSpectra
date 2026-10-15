"""Tests for mukha_pipeline.io.spectral (spectral file format v1, CSV and JSON)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from wlspec_helpers import CSV_HEADER, make_calibration, make_reading, spectral_csv, spectral_json

from mukha_pipeline.io.spectral import (
    SPECTRAL_FORMAT_V1,
    SpectralParseError,
    parse_spectral_file,
)
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import AS7341_CHANNELS, CaptureKind, LedMode, SourceAdapter
from mukha_pipeline.sites import OralSite

L, R = OralSite.BUCCAL_MUCOSA_LEFT, OralSite.BUCCAL_MUCOSA_RIGHT


@pytest.fixture
def pair(rng):
    lesion = make_reading(rng, "s1", L, "lesion", 2.0, n=3, reference_reading_id="s2")
    lesion.lesion_ref = "r1"
    ref = make_reading(rng, "s2", R, "reference", 1.0, n=3)
    return lesion, ref


def _error(data: bytes, filename: str = "spectrum.csv", **kw) -> str:
    with pytest.raises(SpectralParseError) as info:
        parse_spectral_file(data, filename, **kw)
    message = str(info.value)
    assert not find_positioning_violations(message)
    return message


def _csv(header: list[str], *rows: list[object]) -> bytes:
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    return ("\n".join(lines) + "\n").encode()


BASE = {
    "reading_id": "s1",
    "site": "buccal_mucosa_left",
    "role": "lesion",
    "kind": "lit",
    "repeat": 0,
    **{c: 1000 for c in AS7341_CHANNELS},
}


def _row(**over: object) -> list[object]:
    rec = {**BASE, **over}
    return [rec.get(c, "") for c in CSV_HEADER]


# --------------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------------


def test_csv_roundtrip_with_calibration(pair, fixed_now):
    lesion, ref = pair
    cal = make_calibration(fixed_now)
    data = spectral_csv([lesion, ref], cal)
    parsed = parse_spectral_file(data, "probe.csv", now=fixed_now, synthetic=True)

    assert [r.reading_id for r in parsed.readings] == ["s1", "s2"]
    s1, s2 = parsed.readings
    assert s1.site == L and s1.role == "lesion" and s2.role == "reference"
    assert s1.reference_reading_id == "s2" and s1.lesion_ref == "r1"
    assert len(s1.lit) == 3 and len(s1.dark) == 2
    assert s1.lit[0].gain == 64 and s1.lit[0].integration_ms == 50
    assert s1.lit[1].channels["F5_555"] == pytest.approx(lesion.lit[1].channels["F5_555"], abs=1e-3)
    assert set(s1.lit[0].channels) == set(AS7341_CHANNELS)

    meta = s1.meta
    assert meta.kind == CaptureKind.SPECTRAL and meta.led_mode == LedMode.WHITE
    assert meta.source_adapter == SourceAdapter.UPLOAD
    assert meta.oral_site == L and meta.device_id == "confirm-01"
    assert meta.synthetic is True and meta.mime_type == "text/csv"
    assert meta.file_name == "probe.csv" and len(meta.file_sha256 or "") == 64
    assert meta.exposure_time_ms == 50
    assert meta.extra["reference_pairing"] == "explicit" and meta.extra["n_lit"] == 3
    assert s1.meta.capture_id != s2.meta.capture_id

    c = parsed.calibration
    assert c is not None and c.synthetic
    assert meta.calibration_id == c.calibration_id
    assert c.device_id == "confirm-01"
    assert c.created_at == fixed_now and c.expires_at == fixed_now + timedelta(hours=24)
    assert c.gain == 64 and c.integration_ms == 50
    assert c.white["F5_555"] == pytest.approx(cal.white["F5_555"], abs=1e-3)
    assert c.dark["F5_555"] == pytest.approx(200.0)


def test_csv_auto_pairs_contralateral_reference(pair, fixed_now):
    lesion, ref = pair
    parsed = parse_spectral_file(spectral_csv([lesion, ref], explicit_refs=False), "p.csv")
    s1 = parsed.readings[0]
    assert s1.reference_reading_id == "s2"
    assert s1.meta.extra["reference_pairing"] == "auto_contralateral"
    assert any("paired automatically" in n for n in parsed.notes)
    assert parsed.calibration is None
    assert s1.meta.calibration_id is None


def test_auto_pairing_requires_exactly_one_candidate(rng):
    lesion = make_reading(rng, "s1", L, "lesion", 2.0, n=3)
    refs = [make_reading(rng, f"r{i}", R, "reference", 1.0, n=3) for i in range(2)]
    parsed = parse_spectral_file(spectral_csv([lesion, *refs], explicit_refs=False), "p.csv")
    assert parsed.readings[0].reference_reading_id is None
    assert parsed.readings[0].meta.extra["reference_pairing"] == "none"

    # Analogous pairing (upper vs lower lip mucosa).
    up = make_reading(rng, "u", OralSite.UPPER_LIP_MUCOSA, "lesion", 2.0, n=3)
    low = make_reading(rng, "d", OralSite.LOWER_LIP_MUCOSA, "reference", 1.0, n=3)
    parsed = parse_spectral_file(spectral_csv([up, low]), "p.csv")
    assert parsed.readings[0].reference_reading_id == "d"
    assert parsed.readings[0].meta.extra["reference_pairing"] == "auto_analogous"

    # Unpaired site (tongue dorsum has no paired site).
    td = make_reading(rng, "t", OralSite.TONGUE_DORSUM, "lesion", 2.0, n=3)
    parsed = parse_spectral_file(spectral_csv([td, low]), "p.csv")
    assert parsed.readings[0].reference_reading_id is None


def test_csv_is_lenient_about_layout(fixed_now):
    header = [h.upper() if h.startswith("F") else h for h in CSV_HEADER] + ["temperature_c"]
    rows = [
        "# exported by Confirm firmware 0.1.0",
        ";".join(header),
        "",
        ";".join(
            str(v)
            for v in _row(
                kind="DARK", site="Buccal-Mucosa-Left", role="LESION", gain="64x", repeat=""
            )
        )
        + ";31.5",
        ";".join(str(v) for v in _row(repeat=1)) + ";31.5",
        ";".join(str(v) for v in _row(repeat="0.0", CLEAR="", NIR="")) + ";31.5",
        ";;;;;;;;",
    ]
    data = "﻿" + "\r\n".join(rows) + "\r\n"
    parsed = parse_spectral_file(data.encode("utf-8"), "export.txt", device_id="probe-9")
    reading = parsed.readings[0]
    assert reading.site == L and reading.role == "lesion"
    assert len(reading.dark) == 1 and reading.dark[0].gain == 64
    assert len(reading.lit) == 2
    assert "CLEAR" not in reading.lit[0].channels  # repeat 0 sorts first
    assert reading.meta.device_id == "probe-9"
    assert reading.meta.extra["gain"] is None  # lit frames have no gain
    assert any("temperature_c" in n for n in parsed.notes)


def test_csv_sniffed_without_extension_and_calibration_only(fixed_now):
    cal = make_calibration(fixed_now)
    data = spectral_csv([], cal)
    parsed = parse_spectral_file(data, "upload", now=fixed_now, device_id="confirm-07")
    assert parsed.readings == []
    assert parsed.calibration is not None and parsed.calibration.device_id == "confirm-07"
    parsed = parse_spectral_file(data, "upload", now=datetime(2026, 1, 1, 8, 0))
    assert parsed.calibration is not None
    assert parsed.calibration.device_id == "unknown"
    assert parsed.calibration.created_at.tzinfo is not None


def test_device_id_from_rows_wins(pair, fixed_now):
    lesion, ref = pair
    cal = make_calibration(fixed_now)
    parsed = parse_spectral_file(spectral_csv([lesion, ref], cal), "p.csv", device_id="other")
    assert parsed.readings[0].meta.device_id == "confirm-01"
    assert parsed.calibration is not None and parsed.calibration.device_id == "confirm-01"


def test_default_clock_is_used_without_now(pair):
    parsed = parse_spectral_file(
        spectral_csv(list(pair), make_calibration(datetime.now(UTC))), "p.csv"
    )
    assert parsed.calibration is not None
    assert abs((parsed.calibration.created_at - datetime.now(UTC)).total_seconds()) < 60


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", "empty"),
        (b"   \n  ", "empty"),
        (b"\xff\xfe\x00bad", "not UTF-8"),
        (b"# only a comment\n", "empty"),
        (_csv(["reading_id", "kind", "F1_415"]), "missing required column(s): F2_445"),
        (_csv(["reading_id", "kind", "F1_415", "F1_415"]), "lists the column 'F1_415' twice"),
        (_csv(CSV_HEADER, _row(kind="glow")), "Row 2: unknown kind 'glow'"),
        (_csv(CSV_HEADER, _row(kind="")), "Row 2: unknown kind '(empty)'"),
        (_csv(CSV_HEADER, _row(), _row(site="cheek")), "Row 3: unknown site 'cheek'"),
        (_csv(CSV_HEADER, _row(site="")), "Row 2: 'site' is empty"),
        (_csv(CSV_HEADER, _row(role="normal")), "Row 2: unknown role 'normal'"),
        (_csv(CSV_HEADER, _row(F5_555="abc")), "Row 2: F5_555 must be a number (got 'abc')"),
        (_csv(CSV_HEADER, _row(F5_555="nan")), "Row 2: F5_555 must be a finite number"),
        (_csv(CSV_HEADER, _row(F3_480="")), "Row 2: channel F3_480 is missing"),
        (_csv(CSV_HEADER, _row(NIR="x")), "Row 2: NIR must be a number"),
        (_csv(CSV_HEADER, _row(gain="0")), "Row 2: gain must be greater than 0"),
        (_csv(CSV_HEADER, _row(repeat="1.5")), "Row 2: repeat must be a whole number"),
        (_csv(CSV_HEADER, _row(repeat="-1")), "Row 2: repeat must be a whole number"),
        (_csv(CSV_HEADER, _row(reading_id="")), "Row 2: reading_id is empty"),
        (_csv(CSV_HEADER, _row(reading_id="calibration")), "reserved for white and white_dark"),
        (
            _csv(CSV_HEADER, _row(), _row(kind="dark", site="buccal_mucosa_right")),
            "Row 3: reading 's1' has rows with different sites",
        ),
        (
            _csv(CSV_HEADER, _row(), _row(kind="dark", role="reference")),
            "Row 3: reading 's1' has rows with different roles",
        ),
        (
            _csv(CSV_HEADER, _row(lesion_ref="a"), _row(kind="dark", lesion_ref="b")),
            "different lesion_ref values ('a' and 'b')",
        ),
        (_csv(CSV_HEADER, _row()), "Reading 's1' has no dark rows"),
        (_csv(CSV_HEADER, _row(kind="dark")), "Reading 's1' has no lit rows"),
        (
            _csv(CSV_HEADER, _row(reference_reading_id="s9"), _row(kind="dark")),
            "names reference reading 's9', which is not in the file",
        ),
        (
            _csv(
                CSV_HEADER,
                _row(reference_reading_id="s3"),
                _row(kind="dark"),
                _row(reading_id="s3"),
                _row(reading_id="s3", kind="dark"),
            ),
            "'s3' is a lesion reading",
        ),
        (
            _csv(CSV_HEADER, _row(reading_id="calibration", kind="white", site="", role="")),
            "has no white_dark rows",
        ),
        (
            _csv(CSV_HEADER, _row(reading_id="calibration", kind="white_dark", site="", role="")),
            "has no white rows",
        ),
        (
            _csv(
                CSV_HEADER,
                _row(kind="white", device_id="a"),
                _row(kind="white_dark", device_id="b"),
            ),
            "white-reference rows name different devices: a, b",
        ),
        (
            _csv(CSV_HEADER) + b"s1,buccal_mucosa_left,lesion,lit,0" + b",1" * 15 + b",extra\n",
            "Row 2: has more values than the header",
        ),
    ],
)
def test_csv_errors(data: bytes, expected: str):
    message = _error(data)
    assert expected in message, message


def test_csv_header_only_has_no_readings():
    assert "contains no readings" in _error(_csv(CSV_HEADER))


def test_file_size_limit(monkeypatch):
    from mukha_pipeline.io import spectral as spectral_mod

    monkeypatch.setattr(spectral_mod, "MAX_SPECTRAL_FILE_BYTES", 10)
    assert "too large" in _error(b"x" * 11)


def test_reference_reading_with_reference_id_is_noted(rng):
    lesion = make_reading(rng, "s1", L, "lesion", 2.0, n=3, reference_reading_id="s2")
    ref = make_reading(rng, "s2", R, "reference", 1.0, n=3, reference_reading_id="s1")
    parsed = parse_spectral_file(spectral_csv([lesion, ref]), "p.csv")
    assert parsed.readings[1].reference_reading_id is None
    assert any("was ignored" in n for n in parsed.notes)


def test_captured_at_column(rng):
    header = [*CSV_HEADER, "captured_at"]
    row_lit = [*_row(), "2026-10-14T09:00:00+05:30"]
    row_dark = [*_row(kind="dark"), ""]
    parsed = parse_spectral_file(_csv(header, row_lit, row_dark), "p.csv")
    assert parsed.readings[0].meta.captured_at == datetime.fromisoformat(
        "2026-10-14T09:00:00+05:30"
    )
    bad = _csv(header, [*_row(), "yesterday"], row_dark)
    assert "captured_at must be an ISO 8601 date-time (got 'yesterday')" in _error(bad)
    conflict = _csv(header, row_lit, [*_row(kind="dark"), "2026-10-15T09:00:00"])
    assert "Row 3: reading 's1' has rows with different captured_at values" in _error(conflict)


# --------------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------------


def test_json_roundtrip(pair, fixed_now):
    lesion, ref = pair
    cal = make_calibration(fixed_now)
    data = spectral_json([lesion, ref], cal, extra={"synthetic": True})
    parsed = parse_spectral_file(data, "probe.json", now=fixed_now)
    s1, s2 = parsed.readings
    assert s1.reference_reading_id == "s2" and s1.lesion_ref == "r1"
    assert s1.meta.firmware_version == "0.1.0" and s1.meta.device_id == "confirm-01"
    assert s1.meta.mime_type == "application/json"
    assert s1.meta.synthetic is True  # the file can add the flag
    assert len(s1.lit) == 3 and len(s1.dark) == 2
    assert s1.lit[0].gain == 64
    c = parsed.calibration
    assert c is not None and c.synthetic and c.gain == 64 and c.integration_ms == 50
    assert s1.meta.calibration_id == c.calibration_id
    assert c.white["F7_630"] == pytest.approx(cal.white["F7_630"], abs=1e-3)
    assert s2.role == "reference"


def test_json_sniffed_and_frame_settings_override(rng):
    lesion = make_reading(rng, "s1", L, "lesion", 2.0, n=2)
    doc = json.loads(spectral_json([lesion], device_id=None, firmware_version=None))
    doc["readings"][0]["lit"][1]["gain"] = 128
    doc["readings"][0]["lit"][1]["f5_555"] = doc["readings"][0]["lit"][1].pop("F5_555")
    doc["captured_at"] = "2026-10-14T09:00:00Z"
    parsed = parse_spectral_file(json.dumps(doc).encode(), "upload.bin", device_id="arg-dev")
    r = parsed.readings[0]
    assert [f.gain for f in r.lit] == [64, 128]
    assert r.meta.extra["gain"] is None  # mixed gains are not summarised
    assert r.meta.device_id == "arg-dev"
    assert r.meta.captured_at == datetime(2026, 10, 14, 9, 0, tzinfo=UTC)


def _json_doc(**over: object) -> dict:
    frame = {c: 100 for c in AS7341_CHANNELS[:8]}
    doc: dict = {
        "format": SPECTRAL_FORMAT_V1,
        "readings": [
            {
                "reading_id": "s1",
                "site": "buccal_mucosa_left",
                "role": "lesion",
                "dark": [dict(frame)],
                "lit": [dict(frame)],
            }
        ],
    }
    doc.update(over)
    return doc


def _reading(**over: object) -> dict:
    r = _json_doc()["readings"][0]
    r.update(over)
    return r


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        ("{not json", "not valid JSON (line 1, column 2)"),
        ([1, 2], "must be an object"),
        ({"readings": []}, '"format": "mukhaspectra.spectral.v1"'),
        (_json_doc(format="mukhaspectra.spectral.v2"), "Unsupported spectral file format"),
        (_json_doc(readings={}), "'readings' must be a list"),
        (_json_doc(readings=[]), "contains no readings"),
        (_json_doc(readings=["x"]), "Reading 1 must be an object"),
        (_json_doc(readings=[_reading(reading_id="")]), "Reading 1: reading_id is empty"),
        (_json_doc(readings=[_reading(reading_id="calibration")]), "reserved for the white"),
        (_json_doc(readings=[_reading(), _reading()]), "reading_id 's1' is used twice"),
        (_json_doc(readings=[_reading(site="palate")]), "Reading 1 ('s1'): unknown site 'palate'"),
        (_json_doc(readings=[_reading(role=None)]), "unknown role '(empty)'"),
        (_json_doc(readings=[_reading(lit=[])]), "'lit' must be a non-empty list"),
        (_json_doc(readings=[_reading(dark=None)]), "'dark' must be a non-empty list"),
        (_json_doc(readings=[_reading(lit=[5])]), "lit frame 1 must be an object"),
        (
            _json_doc(readings=[_reading(lit=[{"F1_415": 1}])]),
            "Reading 1 ('s1'), lit frame 1: channel F2_445 is missing",
        ),
        (
            _json_doc(
                readings=[_reading(lit=[{**{c: 1 for c in AS7341_CHANNELS[:8]}, "F4_515": True}])]
            ),
            "F4_515 must be a number (got True)",
        ),
        (
            _json_doc(
                readings=[_reading(lit=[{**{c: 1 for c in AS7341_CHANNELS[:8]}, "F4_515": [1]}])]
            ),
            "F4_515 must be a number",
        ),
        (_json_doc(readings=[_reading(gain=-1)]), "Reading 1 ('s1'): gain must be greater than 0"),
        (_json_doc(white_reference=[1]), "'white_reference' must be an object"),
        (_json_doc(white_reference={"white": 3}), "white_reference.white must be a list"),
        (
            _json_doc(white_reference={"white": [{"F1_415": 1}], "white_dark": []}),
            "white_reference.white frame 1: channel F2_445 is missing",
        ),
        (_json_doc(captured_at="soon"), "The file: captured_at must be an ISO 8601"),
    ],
)
def test_json_errors(doc: object, expected: str):
    data = doc.encode() if isinstance(doc, str) else json.dumps(doc).encode()
    message = _error(data, "probe.json")
    assert expected in message, message


def test_json_calibration_only(fixed_now):
    cal = make_calibration(fixed_now)
    data = spectral_json([], cal, device_id="confirm-02")
    parsed = parse_spectral_file(data, "cal.json", now=fixed_now)
    assert parsed.readings == []
    assert parsed.calibration is not None and parsed.calibration.device_id == "confirm-02"
    assert set(parsed.calibration.white) == set(AS7341_CHANNELS)
    assert np.isfinite(list(parsed.calibration.white.values())).all()


def test_all_notes_and_messages_are_positioning_clean(pair, fixed_now):
    lesion, ref = pair
    parsed = parse_spectral_file(spectral_csv([lesion, ref], explicit_refs=False), "p.csv")
    for note in parsed.notes:
        assert not find_positioning_violations(note)
