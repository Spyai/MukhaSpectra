"""Parser for Confirm-probe spectral files (format v1, CSV or JSON).

The format is shared by uploads, the synthetic generator and the firmware, and is specified in
``docs/pipeline_contracts.md`` and ``docs/device_protocol.md``. Parsing is strict about anything
that would silently corrupt a measurement (unknown site or role, missing channels, non-numeric
counts) and reports problems with the row or reading that caused them, so the person uploading
can fix the file. Rows sharing a ``reading_id`` form one reading: N lit frames + dark frames.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import PurePath
from typing import Any, Literal

import numpy as np

from mukha_pipeline.schema.models import (
    AS7341_CHANNELS,
    AS7341_SPECTRAL_CHANNELS,
    CaptureKind,
    CaptureMetadata,
    LedMode,
    SourceAdapter,
)
from mukha_pipeline.sites import OralSite
from mukha_pipeline.spectral.pairing import PairingEntry, auto_pair
from mukha_pipeline.types import SpectralCalibration, SpectralFrame, SpectralInput

SPECTRAL_FORMAT_V1 = "mukhaspectra.spectral.v1"
CALIBRATION_READING_ID = "calibration"
CALIBRATION_VALID_HOURS = 24
MAX_SPECTRAL_FILE_BYTES = 5 * 1024 * 1024
OPTIONAL_CHANNELS: tuple[str, ...] = tuple(
    c for c in AS7341_CHANNELS if c not in AS7341_SPECTRAL_CHANNELS
)

Kind = Literal["lit", "dark", "white", "white_dark"]
KINDS: tuple[str, ...] = ("lit", "dark", "white", "white_dark")
ROLES: tuple[str, ...] = ("lesion", "reference")

_TEXT_COLUMNS = (
    "reading_id",
    "site",
    "role",
    "kind",
    "repeat",
    "gain",
    "integration_ms",
    "reference_reading_id",
    "lesion_ref",
    "device_id",
    "captured_at",
)
_KNOWN_COLUMNS = {c.lower(): c for c in (*_TEXT_COLUMNS, *AS7341_CHANNELS)}
_REQUIRED_COLUMNS = ("reading_id", "kind", *AS7341_SPECTRAL_CHANNELS)


class SpectralParseError(ValueError):
    """The spectral file is invalid. ``str(err)`` is a plain message that names the bad row."""


@dataclass
class ParsedSpectral:
    readings: list[SpectralInput]
    calibration: SpectralCalibration | None
    notes: list[str] = field(default_factory=list)  # e.g. automatic pairings, ignored columns


@dataclass
class _Row:
    """One frame, independent of the file format."""

    where: str  # "Row 5" (CSV) or "Reading 2 ('s1'), lit frame 3" (JSON)
    reading_id: str
    kind: Kind
    channels: dict[str, float]
    site: OralSite | None = None
    role: str | None = None
    repeat: int | None = None
    gain: float | None = None
    integration_ms: float | None = None
    reference_reading_id: str | None = None
    lesion_ref: str | None = None
    device_id: str | None = None
    captured_at: datetime | None = None
    order: int = 0


@dataclass
class _FileInfo:
    device_id: str | None = None
    firmware_version: str | None = None
    synthetic: bool = False
    captured_at: datetime | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------


def parse_spectral_file(
    data: bytes,
    filename: str,
    *,
    source_adapter: SourceAdapter = SourceAdapter.UPLOAD,
    device_id: str | None = None,
    synthetic: bool = False,
    now: datetime | None = None,
) -> ParsedSpectral:
    """Parse a v1 spectral CSV or JSON file into readings and an optional white calibration.

    ``device_id`` is used when the file does not name the probe. ``synthetic`` marks every
    reading and the calibration as synthetic (a file can add the flag, never remove it).
    """
    if not data:
        raise SpectralParseError("The spectral file is empty.")
    if len(data) > MAX_SPECTRAL_FILE_BYTES:
        raise SpectralParseError(
            f"The spectral file is too large ({len(data) / 1e6:.1f} MB); the limit is "
            f"{MAX_SPECTRAL_FILE_BYTES / 1e6:.0f} MB."
        )
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as err:
        raise SpectralParseError("The spectral file is not UTF-8 text.") from err
    if not text.strip():
        raise SpectralParseError("The spectral file is empty.")

    is_json = _looks_like_json(text, filename)
    rows, info = _rows_from_json(text) if is_json else _rows_from_csv(text)
    info.synthetic = info.synthetic or synthetic
    clock = _as_utc(now) if now is not None else datetime.now(UTC)
    row_devices = {r.device_id for r in rows if r.device_id}
    single_row_device = next(iter(row_devices)) if len(row_devices) == 1 else None
    probe_id = info.device_id or single_row_device or device_id

    calibration = _build_calibration(
        [r for r in rows if r.kind in ("white", "white_dark")], probe_id, info.synthetic, clock
    )
    readings, notes = _build_readings(
        [r for r in rows if r.kind in ("lit", "dark")],
        info=info,
        filename=filename,
        sha256=hashlib.sha256(data).hexdigest(),
        mime_type="application/json" if is_json else "text/csv",
        source_adapter=SourceAdapter(source_adapter),
        default_device_id=probe_id,
        calibration_id=calibration.calibration_id if calibration else None,
    )
    if not readings and calibration is None:
        raise SpectralParseError("The spectral file contains no readings.")
    return ParsedSpectral(readings=readings, calibration=calibration, notes=info.notes + notes)


def _looks_like_json(text: str, filename: str | None) -> bool:
    suffix = PurePath(filename or "").suffix.lower()
    if suffix == ".json":
        return True
    if suffix in (".csv", ".tsv", ".txt"):
        return False
    return text.lstrip().startswith("{")


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


# --------------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------------


def _rows_from_csv(text: str) -> tuple[list[_Row], _FileInfo]:
    info = _FileInfo()
    lines = text.splitlines()
    header_line = next((ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")), "")
    delimiter = max((",", ";", "\t"), key=header_line.count)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)

    header: list[str | None] | None = None
    rows: list[_Row] = []
    for raw in reader:
        if not raw or not any(cell.strip() for cell in raw) or raw[0].lstrip().startswith("#"):
            continue
        if header is None:
            header = _csv_header(raw, info)
            continue
        where = f"Row {reader.line_num}"
        record = {
            name: raw[i].strip()
            for i, name in enumerate(header)
            if name is not None and i < len(raw)
        }
        if len(raw) > len(header) and any(c.strip() for c in raw[len(header) :]):
            raise SpectralParseError(f"{where}: has more values than the header has columns.")
        rows.append(_csv_row(record, where, len(rows)))
    if header is None:
        raise SpectralParseError("The spectral file is empty.")
    return rows, info


def _csv_header(raw: list[str], info: _FileInfo) -> list[str | None]:
    names: list[str | None] = []
    unknown: list[str] = []
    for cell in raw:
        key = cell.strip()
        canonical = _KNOWN_COLUMNS.get(key.lower())
        if canonical is None:
            if key:
                unknown.append(key)
            names.append(None)
            continue
        if canonical in names:
            raise SpectralParseError(f"The CSV header lists the column '{canonical}' twice.")
        names.append(canonical)
    missing = [c for c in _REQUIRED_COLUMNS if c not in names]
    if missing:
        raise SpectralParseError(
            "The CSV header is missing required column(s): " + ", ".join(missing) + "."
        )
    if unknown:
        info.notes.append("Ignored unknown column(s): " + ", ".join(unknown) + ".")
    return names


def _csv_row(rec: Mapping[str, str], where: str, order: int) -> _Row:
    kind = _kind(rec.get("kind"), where)
    reading_id = _text(rec.get("reading_id"))
    if kind in ("white", "white_dark"):
        reading_id = reading_id or CALIBRATION_READING_ID
    elif not reading_id:
        raise SpectralParseError(f"{where}: reading_id is empty.")
    channels = _channels(rec, where)
    return _Row(
        where=where,
        reading_id=reading_id,
        kind=kind,
        channels=channels,
        site=_site(rec.get("site"), where) if kind in ("lit", "dark") else None,
        role=_role(rec.get("role"), where) if kind in ("lit", "dark") else None,
        repeat=_repeat(rec.get("repeat"), where),
        gain=_optional_number(rec.get("gain"), where, "gain"),
        integration_ms=_optional_number(rec.get("integration_ms"), where, "integration_ms"),
        reference_reading_id=_text(rec.get("reference_reading_id")),
        lesion_ref=_text(rec.get("lesion_ref")),
        device_id=_text(rec.get("device_id")),
        captured_at=_datetime(rec.get("captured_at"), where),
        order=order,
    )


# --------------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------------


def _rows_from_json(text: str) -> tuple[list[_Row], _FileInfo]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as err:
        raise SpectralParseError(
            f"The spectral file is not valid JSON (line {err.lineno}, column {err.colno}): "
            f"{err.msg}."
        ) from err
    if not isinstance(doc, dict):
        raise SpectralParseError("The spectral JSON must be an object with a 'readings' list.")
    fmt = doc.get("format")
    if fmt != SPECTRAL_FORMAT_V1:
        if fmt is None:
            raise SpectralParseError(
                f'The spectral JSON must contain "format": "{SPECTRAL_FORMAT_V1}".'
            )
        raise SpectralParseError(
            f"Unsupported spectral file format '{fmt}'. This version reads '{SPECTRAL_FORMAT_V1}'."
        )
    info = _FileInfo(
        device_id=_text(doc.get("device_id")),
        firmware_version=_text(doc.get("firmware_version")),
        synthetic=doc.get("synthetic") is True,
        captured_at=_datetime(doc.get("captured_at"), "The file"),
    )
    rows: list[_Row] = []
    rows.extend(_json_white_rows(doc.get("white_reference"), len(rows)))
    readings = doc.get("readings", [])
    if not isinstance(readings, list):
        raise SpectralParseError("'readings' must be a list.")
    seen: set[str] = set()
    for idx, reading in enumerate(readings, start=1):
        new = _json_reading_rows(reading, idx, len(rows))
        rid = new[0].reading_id
        if rid in seen:
            raise SpectralParseError(f"Reading {idx}: reading_id '{rid}' is used twice.")
        seen.add(rid)
        rows.extend(new)
    return rows, info


def _json_white_rows(block: Any, order: int) -> list[_Row]:
    if block is None:
        return []
    if not isinstance(block, dict):
        raise SpectralParseError(
            "'white_reference' must be an object with 'white' and 'white_dark' lists."
        )
    gain = _optional_number(block.get("gain"), "white_reference", "gain")
    integ = _optional_number(block.get("integration_ms"), "white_reference", "integration_ms")
    rows: list[_Row] = []
    for kind in ("white", "white_dark"):
        frames = block.get(kind, [])
        if not isinstance(frames, list):
            raise SpectralParseError(f"white_reference.{kind} must be a list of readings.")
        for j, frame in enumerate(frames, start=1):
            where = f"white_reference.{kind} frame {j}"
            rows.append(
                _json_frame(
                    frame,
                    where,
                    CALIBRATION_READING_ID,
                    kind,
                    j - 1,
                    gain,
                    integ,
                    order + len(rows),
                )
            )
    return rows


def _json_reading_rows(reading: Any, idx: int, order: int) -> list[_Row]:
    if not isinstance(reading, dict):
        raise SpectralParseError(f"Reading {idx} must be an object.")
    rid = _text(reading.get("reading_id"))
    if not rid:
        raise SpectralParseError(f"Reading {idx}: reading_id is empty.")
    if rid == CALIBRATION_READING_ID:
        raise SpectralParseError(
            f"Reading {idx}: reading_id 'calibration' is reserved for the white reference."
        )
    where = f"Reading {idx} ('{rid}')"
    site = _site(reading.get("site"), where)
    role = _role(reading.get("role"), where)
    gain = _optional_number(reading.get("gain"), where, "gain")
    integ = _optional_number(reading.get("integration_ms"), where, "integration_ms")
    common = {
        "site": site,
        "role": role,
        "reference_reading_id": _text(reading.get("reference_reading_id")),
        "lesion_ref": _text(reading.get("lesion_ref")),
        "device_id": _text(reading.get("device_id")),
        "captured_at": _datetime(reading.get("captured_at"), where),
    }
    rows: list[_Row] = []
    for kind in ("dark", "lit"):
        frames = reading.get(kind)
        if not isinstance(frames, list) or not frames:
            raise SpectralParseError(f"{where}: '{kind}' must be a non-empty list of frames.")
        for j, frame in enumerate(frames, start=1):
            row = _json_frame(
                frame,
                f"{where}, {kind} frame {j}",
                rid,
                kind,
                j - 1,
                gain,
                integ,
                order + len(rows),
            )
            for key, value in common.items():
                setattr(row, key, value)
            rows.append(row)
    return rows


def _json_frame(
    frame: Any,
    where: str,
    reading_id: str,
    kind: Kind,
    repeat: int,
    gain: float | None,
    integration_ms: float | None,
    order: int,
) -> _Row:
    if not isinstance(frame, dict):
        raise SpectralParseError(f"{where} must be an object of channel counts.")
    rec = {_KNOWN_COLUMNS.get(str(k).strip().lower(), str(k)): v for k, v in frame.items()}
    return _Row(
        where=where,
        reading_id=reading_id,
        kind=kind,
        channels=_channels(rec, where),
        repeat=repeat,
        gain=_optional_number(rec.get("gain"), where, "gain") or gain,
        integration_ms=_optional_number(rec.get("integration_ms"), where, "integration_ms")
        or integration_ms,
        order=order,
    )


# --------------------------------------------------------------------------------------------
# Field parsing
# --------------------------------------------------------------------------------------------


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _kind(value: Any, where: str) -> Kind:
    text = (_text(value) or "").lower()
    if text not in KINDS:
        shown = _text(value) or "(empty)"
        raise SpectralParseError(
            f"{where}: unknown kind '{shown}'. Use one of: {', '.join(KINDS)}."
        )
    return text  # type: ignore[return-value]


def _site(value: Any, where: str) -> OralSite:
    text = _text(value)
    if not text:
        raise SpectralParseError(f"{where}: 'site' is empty; every reading needs an oral site.")
    key = text.lower().replace("-", "_").replace(" ", "_")
    try:
        return OralSite(key)
    except ValueError:
        raise SpectralParseError(
            f"{where}: unknown site '{text}'. Use one of: {', '.join(s.value for s in OralSite)}."
        ) from None


def _role(value: Any, where: str) -> str:
    text = (_text(value) or "").lower()
    if text not in ROLES:
        shown = _text(value) or "(empty)"
        raise SpectralParseError(f"{where}: unknown role '{shown}'. Use 'lesion' or 'reference'.")
    return text


def _number(value: Any, where: str, column: str) -> float:
    if isinstance(value, bool) or value is None:
        raise SpectralParseError(f"{where}: {column} must be a number (got {value!r}).")
    try:
        out = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise SpectralParseError(f"{where}: {column} must be a number (got {value!r}).") from None
    if not math.isfinite(out):
        raise SpectralParseError(f"{where}: {column} must be a finite number (got {value!r}).")
    return out


def _optional_number(value: Any, where: str, column: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str) and column == "gain":
        value = value.strip().rstrip("xX")  # AS7341 gain is often written "64x"
    out = _number(value, where, column)
    if out <= 0:
        raise SpectralParseError(f"{where}: {column} must be greater than 0 (got {value!r}).")
    return out


def _repeat(value: Any, where: str) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    out = _number(value, where, "repeat")
    if out < 0 or not float(out).is_integer():
        raise SpectralParseError(f"{where}: repeat must be a whole number from 0 (got {value!r}).")
    return int(out)


def _channels(rec: Mapping[str, Any], where: str) -> dict[str, float]:
    channels: dict[str, float] = {}
    for name in AS7341_SPECTRAL_CHANNELS:
        value = rec.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise SpectralParseError(f"{where}: channel {name} is missing.")
        channels[name] = _number(value, where, name)
    for name in OPTIONAL_CHANNELS:
        value = rec.get(name)
        if value is not None and not (isinstance(value, str) and not value.strip()):
            channels[name] = _number(value, where, name)
    return channels


def _datetime(value: Any, where: str) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise SpectralParseError(
            f"{where}: captured_at must be an ISO 8601 date-time (got '{text}')."
        ) from None


# --------------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------------


@dataclass
class _Group:
    reading_id: str
    site: OralSite
    role: str
    rows: list[_Row] = field(default_factory=list)
    reference_reading_id: str | None = None
    lesion_ref: str | None = None
    device_id: str | None = None
    captured_at: datetime | None = None


def _merge(group: _Group, attr: str, row: _Row, label: str) -> None:
    new = getattr(row, attr)
    old = getattr(group, attr)
    if new is None or new == old:
        return
    if old is None:
        setattr(group, attr, new)
        return
    raise SpectralParseError(
        f"{row.where}: reading '{group.reading_id}' has rows with different {label} "
        f"('{old}' and '{new}')."
    )


def _group_rows(rows: Iterable[_Row]) -> list[_Group]:
    groups: dict[str, _Group] = {}
    for row in rows:
        if row.reading_id == CALIBRATION_READING_ID:
            raise SpectralParseError(
                f"{row.where}: reading_id 'calibration' is reserved for white and white_dark rows."
            )
        assert row.site is not None and row.role is not None  # guaranteed by row parsing
        group = groups.get(row.reading_id)
        if group is None:
            group = groups[row.reading_id] = _Group(row.reading_id, row.site, row.role)
        if row.site != group.site:
            raise SpectralParseError(
                f"{row.where}: reading '{group.reading_id}' has rows with different sites "
                f"('{group.site.value}' and '{row.site.value}')."
            )
        if row.role != group.role:
            raise SpectralParseError(
                f"{row.where}: reading '{group.reading_id}' has rows with different roles "
                f"('{group.role}' and '{row.role}')."
            )
        _merge(group, "reference_reading_id", row, "reference_reading_id values")
        _merge(group, "lesion_ref", row, "lesion_ref values")
        _merge(group, "device_id", row, "device_id values")
        _merge(group, "captured_at", row, "captured_at values")
        group.rows.append(row)
    return list(groups.values())


def _frames(rows: Iterable[_Row], kind: str) -> list[SpectralFrame]:
    selected = sorted(
        (r for r in rows if r.kind == kind),
        key=lambda r: (r.repeat if r.repeat is not None else math.inf, r.order),
    )
    return [SpectralFrame(dict(r.channels), r.gain, r.integration_ms) for r in selected]


def _uniform(values: Iterable[float | None]) -> float | None:
    distinct = {v for v in values if v is not None}
    return distinct.pop() if len(distinct) == 1 else None


def _validate_references(groups: list[_Group], notes: list[str]) -> None:
    by_id = {g.reading_id: g for g in groups}
    for g in groups:
        ref = g.reference_reading_id
        if ref is None:
            continue
        if g.role == "reference":
            notes.append(
                f"Reading '{g.reading_id}' is a reference reading; its reference_reading_id "
                f"'{ref}' was ignored."
            )
            g.reference_reading_id = None
            continue
        target = by_id.get(ref)
        if target is None:
            raise SpectralParseError(
                f"Reading '{g.reading_id}' names reference reading '{ref}', which is not in "
                "the file."
            )
        if target.role != "reference":
            raise SpectralParseError(
                f"Reading '{g.reading_id}' names '{ref}' as its reference, but '{ref}' is a "
                "lesion reading."
            )


def _build_readings(
    rows: list[_Row],
    *,
    info: _FileInfo,
    filename: str,
    sha256: str,
    mime_type: str,
    source_adapter: SourceAdapter,
    default_device_id: str | None,
    calibration_id: str | None,
) -> tuple[list[SpectralInput], list[str]]:
    groups = _group_rows(rows)
    notes: list[str] = []
    for g in groups:
        kinds = {r.kind for r in g.rows}
        for needed, why in (("lit", "with the LED on"), ("dark", "with the LED off")):
            if needed not in kinds:
                raise SpectralParseError(
                    f"Reading '{g.reading_id}' has no {needed} rows; the probe takes at least "
                    f"one reading {why}."
                )
    _validate_references(groups, notes)
    pairs = auto_pair(
        PairingEntry(g.reading_id, g.site, g.role, g.reference_reading_id) for g in groups
    )

    readings: list[SpectralInput] = []
    for g in groups:
        pairing = "explicit" if g.reference_reading_id else "none"
        if g.reading_id in pairs:
            chosen = pairs[g.reading_id]
            g.reference_reading_id = chosen.reference_reading_id
            pairing = f"auto_{chosen.pairing}"
            notes.append(
                f"Lesion reading '{g.reading_id}' was paired automatically with the "
                f"{chosen.pairing} reference reading '{chosen.reference_reading_id}'."
            )
        lit, dark = _frames(g.rows, "lit"), _frames(g.rows, "dark")
        gain = _uniform(f.gain for f in lit)
        integration = _uniform(f.integration_ms for f in lit)
        meta = CaptureMetadata(
            capture_id=str(uuid.uuid4()),
            kind=CaptureKind.SPECTRAL,
            oral_site=g.site,
            source_adapter=source_adapter,
            device_id=g.device_id or default_device_id,
            firmware_version=info.firmware_version,
            led_mode=LedMode.WHITE,
            exposure_time_ms=integration,
            captured_at=g.captured_at or info.captured_at,
            calibration_id=calibration_id,
            file_name=filename,
            file_sha256=sha256,
            mime_type=mime_type,
            synthetic=info.synthetic,
            extra={
                "reading_id": g.reading_id,
                "role": g.role,
                "format": SPECTRAL_FORMAT_V1,
                "n_lit": len(lit),
                "n_dark": len(dark),
                "gain": gain,
                "integration_ms": integration,
                "reference_pairing": pairing,
            },
        )
        readings.append(
            SpectralInput(
                reading_id=g.reading_id,
                site=g.site,
                role="lesion" if g.role == "lesion" else "reference",
                lit=lit,
                dark=dark,
                meta=meta,
                reference_reading_id=g.reference_reading_id,
                lesion_ref=g.lesion_ref,
            )
        )
    return readings, notes


def _mean_channels(rows: list[_Row]) -> dict[str, float]:
    names = [c for c in AS7341_CHANNELS if all(c in r.channels for r in rows)]
    return {c: float(np.mean([r.channels[c] for r in rows])) for c in names}


def _build_calibration(
    rows: list[_Row], device_id: str | None, synthetic: bool, now: datetime
) -> SpectralCalibration | None:
    """White-standard rows -> SpectralCalibration, so reflectance = (S - D) / (W - Dw)."""
    if not rows:
        return None
    white = [r for r in rows if r.kind == "white"]
    white_dark = [r for r in rows if r.kind == "white_dark"]
    if not white or not white_dark:
        missing = "white" if not white else "white_dark"
        raise SpectralParseError(
            f"The white-reference calibration has no {missing} rows; it needs both a reading of "
            "the white standard (white) and one with the LED off (white_dark)."
        )
    devices = {r.device_id for r in rows if r.device_id}
    if len(devices) > 1:
        raise SpectralParseError(
            "The white-reference rows name different devices: " + ", ".join(sorted(devices)) + "."
        )
    return SpectralCalibration(
        calibration_id=f"spcal-{uuid.uuid4().hex[:12]}",
        device_id=(devices.pop() if devices else None) or device_id or "unknown",
        created_at=now,
        expires_at=now + timedelta(hours=CALIBRATION_VALID_HOURS),
        dark=_mean_channels(white_dark),
        white=_mean_channels(white),
        gain=_uniform(r.gain for r in white),
        integration_ms=_uniform(r.integration_ms for r in white),
        synthetic=synthetic,
    )
