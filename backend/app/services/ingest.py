"""Upload ingestion: manifest validation, consent gate, decoding, storage, first analysis.

Order matters for privacy and integrity:
1. Consent is checked first. Without ``consent_recorded: true`` the request is rejected before
   anything is decoded or stored.
2. The manifest is validated and every referenced file must be present (router + here).
3. Every capture is decoded and every spectral file parsed *before* the first write, so a bad
   image never leaves a half-stored screening behind.
4. Raw bytes, rows and the audit event are written in one transaction; on failure the stored
   files are removed again.
5. The pipeline runs. If it fails, the captures stay stored and the screening can be re-analysed.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.errors import InvalidInputError, ServiceUnavailableError
from app.models import AnalysisResult, as_utc, new_id, utcnow
from app.schemas import ManifestSite, UploadManifest
from app.services.analysis import AnalysisService
from app.services.audit import AuditAction, record_event
from app.services.inputs import CaptureStore, FilePart, guess_extension, neutral_file_name
from app.services.pipeline_bridge import PipelineBridge
from app.services.records import (
    create_screening_record,
    new_pseudonymous_id,
    site_input_entry,
)
from app.storage import Storage
from mukha_pipeline.schema.models import (
    CaptureKind,
    DataProvenance,
    PatientInfo,
    ScreeningResult,
    SourceAdapter,
)
from mukha_pipeline.sites import site_label

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mukha_pipeline.types import SiteInput

logger = logging.getLogger("app.ingest")

CONSENT_REQUIRED = "Consent is required before any data is stored."
MANIFEST_MISSING = (
    "The upload must include a 'manifest' form field with the screening details as JSON."
)
PIPELINE_UNAVAILABLE = "The analysis pipeline is not available on this server."

IMAGE_SLOTS: tuple[tuple[str, CaptureKind], ...] = (
    ("wl", CaptureKind.WHITE_LIGHT),
    ("af", CaptureKind.AUTOFLUORESCENCE),
    ("dark", CaptureKind.DARK_FRAME),
)
_MAX_REPORTED_ERRORS = 8


def format_validation_error(exc: ValidationError, *, prefix: str = "manifest") -> str:
    parts: list[str] = []
    for err in exc.errors()[:_MAX_REPORTED_ERRORS]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
        parts.append(f"{prefix}.{loc}: {msg}" if loc else f"{prefix}: {msg}")
    extra = exc.error_count() - len(parts)
    if extra > 0:
        parts.append(f"(and {extra} more)")
    return "Invalid manifest. " + "; ".join(parts)


class IngestService:
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

    # -- validation (no side effects) ---------------------------------------------------------

    @staticmethod
    def parse_manifest(raw: str | bytes | None) -> UploadManifest:
        """Parse and validate the manifest. Consent is checked before anything else."""
        if raw is None or (isinstance(raw, str | bytes) and not raw.strip()):
            raise InvalidInputError(MANIFEST_MISSING)
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise InvalidInputError(f"The manifest is not valid JSON ({exc}).") from exc
        if not isinstance(data, dict):
            raise InvalidInputError("The manifest must be a JSON object.")
        consent = data.get("consent")
        if not (isinstance(consent, dict) and consent.get("consent_recorded") is True):
            raise InvalidInputError(CONSENT_REQUIRED)
        try:
            return UploadManifest.model_validate(data)
        except ValidationError as exc:
            raise InvalidInputError(format_validation_error(exc)) from exc

    # -- ingestion -----------------------------------------------------------------------------

    def _decode_site(
        self,
        adapter: Any,
        entry: ManifestSite,
        files: dict[str, FilePart],
        *,
        synthetic: bool,
        error_types: tuple[type[BaseException], ...],
    ) -> tuple[SiteInput, dict[str, FilePart], dict[str, str]]:
        names = {"wl": entry.wl, "af": entry.af, "dark": entry.dark}
        parts = {slot: files[name] for slot, name in names.items() if name}
        exts = {
            slot: guess_extension(part.filename, part.data, spectral=False)
            for slot, part in parts.items()
        }
        kinds = dict(IMAGE_SLOTS)
        filenames = {
            slot: neutral_file_name(entry.site, kinds[slot], ext) for slot, ext in exts.items()
        }
        try:
            site_input = adapter.site_input(
                entry.site,
                parts["wl"].data,
                parts["af"].data,
                parts["dark"].data if "dark" in parts else None,
                filenames=filenames,
                role=entry.role,
                visual_flag=entry.visual_flag,
                diascopy=entry.diascopy,
                px_per_mm=entry.px_per_mm,
                device_id=entry.device_id,
                firmware_version=entry.firmware_version,
                synthetic=synthetic,
            )
        except ImportError as exc:
            raise ServiceUnavailableError(PIPELINE_UNAVAILABLE) from exc
        except error_types as exc:
            raise InvalidInputError(
                f"Could not read the images for {site_label(entry.site)}: {exc}"
            ) from exc
        return site_input, parts, exts

    def create_screening(
        self, session: Session, manifest: UploadManifest, files: dict[str, FilePart]
    ) -> tuple[str, AnalysisResult, ScreeningResult]:
        """Decode, store and analyse one uploaded screening. Returns (id, row, result)."""
        from mukha_pipeline.types import ScreeningInput

        missing = [name for name, _ in manifest.file_fields() if name not in files]
        if missing:  # the router checks this too; keep the service safe on its own
            raise InvalidInputError(f"File field '{missing[0]}' is missing from the upload.")

        now = utcnow()
        synthetic = manifest.data_provenance == DataProvenance.SYNTHETIC
        clock = as_utc(manifest.context.captured_at) or now
        error_types = self.bridge.input_error_types()
        try:
            adapter = self.bridge.upload_adapter_factory()
        except ImportError as exc:
            raise ServiceUnavailableError(PIPELINE_UNAVAILABLE) from exc

        # 1) decode everything first: nothing is written if any input is unreadable
        decoded = [
            (
                entry,
                *self._decode_site(
                    adapter, entry, files, synthetic=synthetic, error_types=error_types
                ),
            )
            for entry in manifest.sites
        ]
        spectral_part: FilePart | None = None
        spectral_ext = ".csv"
        readings: list[Any] = []
        spectral_cals: dict[str, Any] = {}
        if manifest.spectral_file:
            spectral_part = files[manifest.spectral_file]
            spectral_ext = guess_extension(
                spectral_part.filename, spectral_part.data, spectral=True
            )
            try:
                parsed = self.bridge.parse_spectral(
                    spectral_part.data,
                    neutral_file_name(None, CaptureKind.SPECTRAL, spectral_ext),
                    source_adapter=SourceAdapter.UPLOAD,
                    device_id=manifest.spectral_device_id,
                    synthetic=synthetic,
                    now=clock,
                )
            except ImportError as exc:
                raise ServiceUnavailableError(PIPELINE_UNAVAILABLE) from exc
            except error_types as exc:
                raise InvalidInputError(f"Could not read the spectral file: {exc}") from exc
            readings = list(parsed.readings)
            if parsed.calibration is not None:
                spectral_cals[parsed.calibration.device_id] = parsed.calibration

        # 2) the pipeline input
        context = manifest.context
        device_ids = context.device_ids or sorted(
            {e.device_id for e in manifest.sites if e.device_id}
            | ({manifest.spectral_device_id} if manifest.spectral_device_id else set())
        )
        firmware = context.firmware_versions or sorted(
            {e.firmware_version for e in manifest.sites if e.firmware_version}
        )
        context = context.model_copy(
            update={"device_ids": device_ids, "firmware_versions": firmware}
        )
        screening_id = new_id()
        inp = ScreeningInput(
            screening_id=screening_id,
            patient=PatientInfo(
                pseudonymous_id=new_pseudonymous_id(synthetic=synthetic),
                age_years=manifest.patient.age_years,
                sex=manifest.patient.sex,
            ),
            consent=manifest.consent,
            risk_factors=manifest.risk_factors,
            context=context,
            sites=[site_input for _entry, site_input, _parts, _exts in decoded],
            spectral=readings,
            spectral_calibrations=spectral_cals,
            data_provenance=manifest.data_provenance,
            now=clock,
        )

        # 3) store raw inputs + rows in one transaction
        try:
            site_entries: list[dict[str, Any]] = []
            n_captures = 0
            for entry, site_input, parts, exts in decoded:
                refs: dict[str, str | None] = {}
                for slot, kind in IMAGE_SLOTS:
                    part = parts.get(slot)
                    if part is None:
                        refs[slot] = None
                        continue
                    capture = self.capture_store.add_capture(
                        session,
                        screening_id,
                        site=entry.site,
                        kind=kind,
                        data=part.data,
                        ext=exts[slot],
                        meta={
                            "capture_metadata": getattr(site_input, slot).meta.model_dump(
                                mode="json"
                            ),
                            "source_adapter": SourceAdapter.UPLOAD.value,
                        },
                    )
                    refs[slot] = capture.id
                    n_captures += 1
                site_entries.append(site_input_entry(site_input, refs))
            if spectral_part is not None:
                self.capture_store.add_capture(
                    session,
                    screening_id,
                    site=None,
                    kind=CaptureKind.SPECTRAL,
                    data=spectral_part.data,
                    ext=spectral_ext,
                    meta={
                        "source_adapter": SourceAdapter.UPLOAD.value,
                        "device_id": manifest.spectral_device_id,
                        "reading_ids": [r.reading_id for r in readings],
                    },
                )
                n_captures += 1
            screening = create_screening_record(
                session,
                inp=inp,
                source="upload",
                site_inputs=site_entries,
                calibration_ids=[],
                identity_name=manifest.patient.name,
                identity_external_id=manifest.patient.external_id,
                created_at=now,
            )
            record_event(
                session,
                AuditAction.SCREENING_CREATED,
                "screening",
                screening_id,
                {
                    "source": "upload",
                    "sites": len(site_entries),
                    "captures": n_captures,
                    "spectral_readings": len(readings),
                    "data_provenance": manifest.data_provenance.value,
                    "consent_method": manifest.consent.method,
                    "research_use_opt_in": manifest.consent.research_use_opt_in,
                },
            )
            session.commit()
        except Exception:
            session.rollback()
            self.storage.delete_prefix(f"screenings/{screening_id}/")
            raise

        # 4) analyse (failures keep the stored captures; see AnalysisFailedError)
        row, result = self.analysis.run(session, screening, inp, reanalysis=False)
        return screening_id, row, result
