"""Stored inputs: raw capture bytes, calibration records, and rebuilding a ScreeningInput.

Raw input bytes are kept exactly as received under ``screenings/{id}/inputs/{capture_id}{ext}``
so a screening can be re-analysed later with a newer pipeline or config. Stored file names are
neutral (``<site>-<kind><ext>``): uploaded file names often contain the patient's name.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CalibrationRecord, Capture, Screening, as_utc, new_id
from app.services.pipeline_bridge import PipelineBridge
from app.storage import Storage, join_key
from mukha_pipeline.schema.models import (
    CaptureKind,
    CaptureMetadata,
    ConsentInfo,
    DataProvenance,
    Diascopy,
    PatientInfo,
    RiskFactors,
    ScreeningContext,
    Sex,
    SiteRole,
    SourceAdapter,
    VisualFlag,
)
from mukha_pipeline.sites import OralSite

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mukha_pipeline.types import (
        ImageCalibration,
        ImageCapture,
        ScreeningInput,
        SpectralCalibration,
    )

IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".dng": "image/x-adobe-dng",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
SPECTRAL_MIME_TYPES: dict[str, str] = {".csv": "text/csv", ".json": "application/json"}

KIND_SLUG: dict[CaptureKind, str] = {
    CaptureKind.WHITE_LIGHT: "wl",
    CaptureKind.AUTOFLUORESCENCE: "af",
    CaptureKind.DARK_FRAME: "dark",
    CaptureKind.SPECTRAL: "spectral",
}


@dataclass(frozen=True)
class FilePart:
    """One uploaded file, read into memory."""

    field: str
    filename: str | None
    content_type: str | None
    data: bytes


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sniff_image_extension(data: bytes) -> str | None:
    head = data[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif"
    if head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"mif1", b"msf1", b"heif"):
        return ".heic"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    if head.startswith(b"BM"):
        return ".bmp"
    return None


def guess_extension(filename: str | None, data: bytes, *, spectral: bool) -> str:
    """Pick a safe extension from the uploaded name or, failing that, the content."""
    allowed = SPECTRAL_MIME_TYPES if spectral else IMAGE_MIME_TYPES
    ext = Path(filename or "").suffix.lower()
    if ext in allowed:
        return ext
    if spectral:
        return ".json" if data.lstrip()[:1] in (b"{", b"[") else ".csv"
    return _sniff_image_extension(data) or ".bin"


def mime_type_for(ext: str) -> str:
    return IMAGE_MIME_TYPES.get(ext) or SPECTRAL_MIME_TYPES.get(ext) or "application/octet-stream"


def neutral_file_name(site: OralSite | str | None, kind: CaptureKind, ext: str) -> str:
    stem = KIND_SLUG[kind] if site is None else f"{OralSite(site).value}-{KIND_SLUG[kind]}"
    return f"{stem}{ext}"


# --------------------------------------------------------------------------------------------
# Calibration (de)serialisation. ImageCalibration holds numpy arrays, stored as .npz (no pickle).
# --------------------------------------------------------------------------------------------


def image_calibration_to_npz(cal: ImageCalibration) -> bytes:
    arrays: dict[str, np.ndarray] = {"color_matrix": np.asarray(cal.color_matrix, np.float64)}
    if cal.flat_field is not None:
        arrays["flat_field"] = np.asarray(cal.flat_field, np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)  # type: ignore[arg-type]
    return buf.getvalue()


def image_calibration_from_record(record: CalibrationRecord, blob: bytes) -> ImageCalibration:
    from mukha_pipeline.types import ImageCalibration

    with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
        color_matrix = np.array(npz["color_matrix"])
        flat_field = np.array(npz["flat_field"]) if "flat_field" in npz.files else None
    return ImageCalibration(
        calibration_id=str(record.data.get("calibration_id") or record.id),
        device_id=record.device_id,
        created_at=as_utc(record.created_at),  # type: ignore[arg-type]
        expires_at=as_utc(record.expires_at),  # type: ignore[arg-type]
        flat_field=flat_field,
        color_matrix=color_matrix,
        af_scale=float(record.data.get("af_scale", 1.0)),
        synthetic=record.synthetic,
    )


def spectral_calibration_from_record(record: CalibrationRecord) -> SpectralCalibration:
    from mukha_pipeline.types import SpectralCalibration

    data = record.data
    return SpectralCalibration(
        calibration_id=str(data.get("calibration_id") or record.id),
        device_id=record.device_id,
        created_at=as_utc(record.created_at),  # type: ignore[arg-type]
        expires_at=as_utc(record.expires_at),  # type: ignore[arg-type]
        dark={k: float(v) for k, v in data.get("dark", {}).items()},
        white={k: float(v) for k, v in data.get("white", {}).items()},
        gain=data.get("gain"),
        integration_ms=data.get("integration_ms"),
        synthetic=record.synthetic,
    )


class StoredInputsError(RuntimeError):
    """Stored captures could not be read back for re-analysis."""


class CaptureStore:
    """Writes captures/calibrations to storage + DB and rebuilds ScreeningInputs from them."""

    def __init__(self, storage: Storage, bridge: PipelineBridge) -> None:
        self.storage = storage
        self.bridge = bridge

    # -- writing -------------------------------------------------------------------------------

    def add_capture(
        self,
        session: Session,
        screening_id: str,
        *,
        site: OralSite | None,
        kind: CaptureKind,
        data: bytes,
        ext: str,
        meta: dict[str, Any],
    ) -> Capture:
        capture_id = new_id()
        capture = Capture(
            id=capture_id,
            screening_id=screening_id,
            site=site.value if site is not None else None,
            kind=kind.value,
            storage_key=join_key("screenings", screening_id, "inputs", f"{capture_id}{ext}"),
            sha256=sha256_hex(data),
            size_bytes=len(data),
            mime_type=mime_type_for(ext),
            file_name=neutral_file_name(site, kind, ext),
            meta=meta,
        )
        self.storage.put(capture.storage_key, data)
        session.add(capture)
        return capture

    def add_image_calibration(
        self, session: Session, cal: ImageCalibration, *, synthetic: bool
    ) -> CalibrationRecord:
        record_id = new_id()
        key = join_key("calibrations", f"{record_id}.npz")
        self.storage.put(key, image_calibration_to_npz(cal))
        record = CalibrationRecord(
            id=record_id,
            device_id=cal.device_id,
            kind="image",
            created_at=cal.created_at,
            expires_at=cal.expires_at,
            synthetic=bool(synthetic or cal.synthetic),
            data={"calibration_id": cal.calibration_id, "af_scale": float(cal.af_scale)},
            storage_key=key,
        )
        session.add(record)
        return record

    def add_spectral_calibration(
        self, session: Session, cal: SpectralCalibration, *, synthetic: bool
    ) -> CalibrationRecord:
        record = CalibrationRecord(
            id=new_id(),
            device_id=cal.device_id,
            kind="spectral",
            created_at=cal.created_at,
            expires_at=cal.expires_at,
            synthetic=bool(synthetic or cal.synthetic),
            data={
                "calibration_id": cal.calibration_id,
                "dark": {k: float(v) for k, v in cal.dark.items()},
                "white": {k: float(v) for k, v in cal.white.items()},
                "gain": cal.gain,
                "integration_ms": cal.integration_ms,
            },
        )
        session.add(record)
        return record

    # -- rebuilding ----------------------------------------------------------------------------

    def _image_capture(self, capture: Capture) -> ImageCapture:
        from mukha_pipeline.types import ImageCapture

        data = self.storage.get(capture.storage_key)
        image = self.bridge.load_image(data, capture.file_name)
        meta = CaptureMetadata.model_validate(capture.meta["capture_metadata"])
        return ImageCapture(image=image, meta=meta)

    def rebuild_input(self, session: Session, screening: Screening) -> ScreeningInput:
        """Recreate the ScreeningInput from stored bytes, metadata and calibration records."""
        from mukha_pipeline.types import ScreeningInput, SiteInput

        captures = {c.id: c for c in screening.captures}
        clock: datetime | None = as_utc(screening.captured_at)
        try:
            sites = []
            for entry in screening.site_inputs:
                refs = entry["captures"]
                sites.append(
                    SiteInput(
                        site=OralSite(entry["site"]),
                        wl=self._image_capture(captures[refs["wl"]]),
                        af=self._image_capture(captures[refs["af"]]),
                        dark=self._image_capture(captures[refs["dark"]])
                        if refs.get("dark")
                        else None,
                        role=SiteRole(entry.get("role", SiteRole.EXAMINED)),
                        visual_flag=VisualFlag.model_validate(entry.get("visual_flag") or {}),
                        diascopy=Diascopy(entry.get("diascopy", Diascopy.NOT_DONE)),
                        px_per_mm=entry.get("px_per_mm"),
                    )
                )

            spectral: list[Any] = []
            spectral_cals: dict[str, Any] = {}
            for capture in screening.captures:
                if capture.kind != CaptureKind.SPECTRAL.value:
                    continue
                parsed = self.bridge.parse_spectral(
                    self.storage.get(capture.storage_key),
                    capture.file_name or "spectral.csv",
                    source_adapter=SourceAdapter(capture.meta.get("source_adapter", "upload")),
                    device_id=capture.meta.get("device_id"),
                    synthetic=screening.synthetic,
                    now=clock,
                )
                spectral.extend(parsed.readings)
                if parsed.calibration is not None:
                    spectral_cals[parsed.calibration.device_id] = parsed.calibration

            image_cals: dict[str, Any] = {}
            if screening.calibration_ids:
                records = session.scalars(
                    select(CalibrationRecord).where(
                        CalibrationRecord.id.in_(screening.calibration_ids)
                    )
                ).all()
                for record in records:
                    if record.kind == "image" and record.storage_key:
                        blob = self.storage.get(record.storage_key)
                        image_cals[record.device_id] = image_calibration_from_record(record, blob)
                    elif record.kind == "spectral":
                        spectral_cals.setdefault(
                            record.device_id, spectral_calibration_from_record(record)
                        )
        except Exception as exc:  # noqa: BLE001 - any failure here means unreadable inputs
            raise StoredInputsError(str(exc)) from exc

        patient = screening.patient
        return ScreeningInput(
            screening_id=screening.id,
            patient=PatientInfo(
                pseudonymous_id=screening.pseudonymous_id,
                age_years=patient.age_years,
                sex=Sex(patient.sex),
            ),
            consent=ConsentInfo.model_validate(screening.consent),
            risk_factors=RiskFactors.model_validate(screening.risk_factors),
            context=ScreeningContext.model_validate(screening.context),
            sites=sites,
            spectral=spectral,
            image_calibrations=image_cals,
            spectral_calibrations=spectral_cals,
            data_provenance=DataProvenance(screening.data_provenance),
            now=clock,
        )
