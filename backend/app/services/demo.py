"""SYNTHETIC demo patients (low / moderate / high).

The generated inputs are stored like real captures (PNG images, a spectral v1 JSON file and
calibration records) so a demo screening can be re-analysed exactly like an upload. The images
analysed in version 1 are the 8-bit PNG round trip of the generator output, so version 1 and a
later re-analysis see identical pixels. Everything is labelled synthetic: provenance, file
metadata, the patient key (``SYN-...``) and the display name ``SYNTHETIC PATIENT <KIND>``.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from app.errors import InvalidInputError, ServiceUnavailableError
from app.models import AnalysisResult, as_utc, new_id, utcnow
from app.services.analysis import AnalysisService
from app.services.audit import AuditAction, record_event
from app.services.inputs import CaptureStore, neutral_file_name, sha256_hex
from app.services.pipeline_bridge import PipelineBridge
from app.services.records import create_screening_record, new_pseudonymous_id, site_input_entry
from app.storage import Storage
from mukha_pipeline.schema.models import (
    CaptureKind,
    DataProvenance,
    ScreeningResult,
    SourceAdapter,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mukha_pipeline.types import SpectralInput

DEMO_KINDS: tuple[str, ...] = ("low", "moderate", "high")
SPECTRAL_FORMAT = "mukhaspectra.spectral.v1"
_IMAGE_SLOTS = (
    ("wl", CaptureKind.WHITE_LIGHT),
    ("af", CaptureKind.AUTOFLUORESCENCE),
    ("dark", CaptureKind.DARK_FRAME),
)


def demo_identity_name(kind: str) -> str:
    return f"SYNTHETIC PATIENT {kind.upper()}"


def encode_png_rgb(image: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Encode a float [0, 1] (or uint8) image as 8-bit PNG.

    Returns the PNG bytes and the float32 RGB image exactly as it will decode again.
    """
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(np.nan_to_num(arr.astype(np.float64)) * 255.0), 0, 255)
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr), mode="RGB").save(buf, format="PNG")
    return buf.getvalue(), arr.astype(np.float32) / np.float32(255.0)


def _frame(frame: Any) -> dict[str, float]:
    out = {str(k): float(v) for k, v in frame.channels.items()}
    if frame.gain is not None:
        out["gain"] = float(frame.gain)
    if frame.integration_ms is not None:
        out["integration_ms"] = float(frame.integration_ms)
    return out


def spectral_inputs_to_files(readings: list[SpectralInput]) -> list[tuple[str | None, bytes]]:
    """Serialise readings as spectral file format v1 (JSON), one file per device.

    Calibrations are stored separately as CalibrationRecords, so there is no
    ``white_reference`` block. Per-frame gain and integration time are kept.
    """
    groups: dict[str | None, list[SpectralInput]] = {}
    for reading in readings:
        groups.setdefault(reading.meta.device_id, []).append(reading)
    files: list[tuple[str | None, bytes]] = []
    for device_id, items in groups.items():
        doc: dict[str, Any] = {"format": SPECTRAL_FORMAT}
        if device_id:
            doc["device_id"] = device_id
        firmware = next((r.meta.firmware_version for r in items if r.meta.firmware_version), None)
        if firmware:
            doc["firmware_version"] = firmware
        if any(r.meta.synthetic for r in items):
            doc["synthetic"] = True
        entries = []
        for r in items:
            entry: dict[str, Any] = {
                "reading_id": r.reading_id,
                "site": r.site.value,
                "role": r.role,
                "reference_reading_id": r.reference_reading_id,
                "lesion_ref": r.lesion_ref,
            }
            if r.meta.captured_at is not None:
                entry["captured_at"] = r.meta.captured_at.isoformat()
            entry["dark"] = [_frame(f) for f in r.dark]
            entry["lit"] = [_frame(f) for f in r.lit]
            entries.append(entry)
        doc["readings"] = entries
        files.append((device_id, json.dumps(doc, indent=1).encode("utf-8")))
    return files


class DemoService:
    def __init__(
        self,
        storage: Storage,
        bridge: PipelineBridge,
        capture_store: CaptureStore,
        analysis: AnalysisService,
    ) -> None:
        self.storage = storage
        self.bridge = bridge
        self.capture_store = capture_store
        self.analysis = analysis

    def create(self, session: Session, kind: str) -> tuple[str, AnalysisResult, ScreeningResult]:
        if kind not in DEMO_KINDS:
            raise InvalidInputError(
                f"Unknown demo patient '{kind}'. Use one of: {', '.join(DEMO_KINDS)}."
            )
        with self.analysis.analysis_slot():
            return self._create(session, kind)

    def _create(self, session: Session, kind: str) -> tuple[str, AnalysisResult, ScreeningResult]:
        try:
            inp = self.bridge.make_demo_patient(kind)
        except ImportError as exc:
            raise ServiceUnavailableError(
                "The synthetic data generator is not available on this server."
            ) from exc

        now = utcnow()
        screening_id = new_id()
        inp.screening_id = screening_id
        inp.patient = inp.patient.model_copy(
            update={"pseudonymous_id": new_pseudonymous_id(synthetic=True)}
        )
        inp.data_provenance = DataProvenance.SYNTHETIC
        inp.now = as_utc(inp.now) or now

        try:
            site_entries: list[dict[str, Any]] = []
            for site_input in inp.sites:
                refs: dict[str, str | None] = {}
                for slot, capture_kind in _IMAGE_SLOTS:
                    capture = getattr(site_input, slot)
                    if capture is None:
                        refs[slot] = None
                        continue
                    png, decoded = encode_png_rgb(capture.image)
                    capture.image = decoded  # analyse exactly what is stored
                    capture.meta = capture.meta.model_copy(
                        update={
                            "file_name": neutral_file_name(site_input.site, capture_kind, ".png"),
                            "file_sha256": sha256_hex(png),
                            "mime_type": "image/png",
                            "synthetic": True,
                        }
                    )
                    row = self.capture_store.add_capture(
                        session,
                        screening_id,
                        site=site_input.site,
                        kind=capture_kind,
                        data=png,
                        ext=".png",
                        meta={
                            "capture_metadata": capture.meta.model_dump(mode="json"),
                            "source_adapter": SourceAdapter.SYNTHETIC.value,
                            "demo_kind": kind,
                        },
                        synthetic=True,
                    )
                    refs[slot] = row.id
                site_entries.append(site_input_entry(site_input, refs))

            for device_id, blob in spectral_inputs_to_files(list(inp.spectral)):
                self.capture_store.add_capture(
                    session,
                    screening_id,
                    site=None,
                    kind=CaptureKind.SPECTRAL,
                    data=blob,
                    ext=".json",
                    meta={
                        "source_adapter": SourceAdapter.SYNTHETIC.value,
                        "device_id": device_id,
                        "reading_ids": [
                            r.reading_id for r in inp.spectral if r.meta.device_id == device_id
                        ],
                        "demo_kind": kind,
                    },
                    synthetic=True,
                )

            calibration_ids = [
                self.capture_store.add_image_calibration(session, cal, synthetic=True).id
                for cal in inp.image_calibrations.values()
            ] + [
                self.capture_store.add_spectral_calibration(session, cal, synthetic=True).id
                for cal in inp.spectral_calibrations.values()
            ]

            screening = create_screening_record(
                session,
                inp=inp,
                source="demo",
                site_inputs=site_entries,
                calibration_ids=calibration_ids,
                identity_name=demo_identity_name(kind),
                identity_external_id=None,
                created_at=now,
            )
            record_event(
                session,
                AuditAction.SCREENING_CREATED,
                "screening",
                screening_id,
                {
                    "source": "demo",
                    "demo_kind": kind,
                    "sites": len(site_entries),
                    "spectral_readings": len(inp.spectral),
                    "data_provenance": DataProvenance.SYNTHETIC.value,
                },
            )
            session.commit()
        except Exception:
            session.rollback()
            self.storage.delete_prefix(f"screenings/{screening_id}/")
            raise

        row, result = self.analysis.run(session, screening, inp, reanalysis=False)
        return screening_id, row, result
