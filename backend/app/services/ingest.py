"""Upload ingestion: manifest validation, consent gate, decoding, storage, first analysis.

Order matters for privacy and integrity:
1. Consent is checked first. Without ``consent_recorded: true`` and a real consent method
   (checkbox, signature or thumbprint) the request is rejected before anything is decoded or
   stored. The server stamps ``recorded_at`` when the client did not send it.
2. The manifest is validated (JSON without NaN / Infinity, field lengths matching the database)
   and every referenced file must be present (router + here).
3. Every capture is decoded (at most 2048 px on the longest side, within a per-request pixel
   budget) and every spectral file parsed *before* the first write, so a bad image never leaves a
   half-stored screening behind. Captures carrying a SYNTHETIC label are marked synthetic even if
   the manifest says "real".
4. Identifying metadata (GPS, owner, serial numbers, XMP, text chunks) is stripped from stored
   JPEG / PNG files; raw bytes, rows and the audit event are written in one transaction; on
   failure the stored files are removed again.
5. The pipeline runs. If it fails, the captures stay stored and the screening can be re-analysed.

Times sent without a UTC offset are camp-local (``settings.default_timezone``) and are stored in
UTC. Only ``settings.max_concurrent_analyses`` uploads / analyses run at once (503 otherwise).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.errors import InvalidInputError, PayloadTooLargeError, ServiceUnavailableError
from app.models import AnalysisResult, new_id, utcnow
from app.schemas import CONSENT_METHOD_REQUIRED, CONSENT_METHODS, ManifestSite, UploadManifest
from app.services.analysis import AnalysisService
from app.services.audit import AuditAction, record_event
from app.services.inputs import CaptureStore, FilePart, guess_extension, neutral_file_name
from app.services.pipeline_bridge import PipelineBridge
from app.services.records import (
    create_screening_record,
    new_pseudonymous_id,
    site_input_entry,
)
from app.settings import Settings
from app.storage import Storage
from mukha_pipeline.schema.models import (
    CaptureKind,
    ConsentInfo,
    DataProvenance,
    PatientInfo,
    ScreeningContext,
    ScreeningResult,
    SourceAdapter,
)
from mukha_pipeline.sites import site_label

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mukha_pipeline.types import ImageCapture, SiteInput

logger = logging.getLogger("app.ingest")

CONSENT_REQUIRED = "Consent is required before any data is stored."
MANIFEST_MISSING = (
    "The upload must include a 'manifest' form field with the screening details as JSON."
)
MANIFEST_NOT_FINITE = "The manifest is not valid JSON: numbers must be finite (no NaN or Infinity)."
MANIFEST_TOO_DEEP = "The manifest is nested too deeply."
PIPELINE_UNAVAILABLE = "The analysis pipeline is not available on this server."

__all__ = [
    "CONSENT_METHOD_REQUIRED",
    "CONSENT_REQUIRED",
    "MANIFEST_MISSING",
    "IngestService",
    "local_to_utc",
]

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


def local_to_utc(value: datetime | None, tz: tzinfo) -> datetime | None:
    """A naive time is camp-local (``tz``); every stored time is timezone-aware UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=tz)
    return value.astimezone(UTC)


class _NonFiniteError(ValueError):
    pass


def _reject_constant(token: str) -> Any:
    raise _NonFiniteError(token)


def _finite_float(text: str) -> float:
    value = float(text)
    if value in (float("inf"), float("-inf")):  # e.g. 1e999
        raise _NonFiniteError(text)
    return value


def _check_consent(consent: Any) -> None:
    """Consent gate (also re-checked inside the service, not only by the router)."""
    if isinstance(consent, dict):
        recorded, method = consent.get("consent_recorded"), consent.get("method")
    else:
        recorded = getattr(consent, "consent_recorded", None)
        method = getattr(consent, "method", None)
    if recorded is not True:
        raise InvalidInputError(CONSENT_REQUIRED)
    if method not in CONSENT_METHODS:
        raise InvalidInputError(CONSENT_METHOD_REQUIRED)


def _captures(site_input: SiteInput) -> list[ImageCapture]:
    return [c for c in (site_input.wl, site_input.af, site_input.dark) if c is not None]


def effective_provenance(
    declared: DataProvenance, any_synthetic: bool, all_synthetic: bool
) -> DataProvenance:
    """Provenance after looking at the files: SYNTHETIC labels are never ignored."""
    if declared == DataProvenance.SYNTHETIC or not any_synthetic:
        return declared
    if all_synthetic and declared == DataProvenance.REAL:
        return DataProvenance.SYNTHETIC
    return DataProvenance.MIXED


class IngestService:
    def __init__(
        self,
        storage: Storage,
        bridge: PipelineBridge,
        capture_store: CaptureStore,
        analysis: AnalysisService,
        settings: Settings | None = None,
    ) -> None:
        self.storage = storage
        self.bridge = bridge
        self.capture_store = capture_store
        self.analysis = analysis
        self.settings = settings or analysis.settings

    # -- validation (no side effects) ---------------------------------------------------------

    @staticmethod
    def parse_manifest(raw: str | bytes | None) -> UploadManifest:
        """Parse and validate the manifest. Consent is checked before anything else."""
        if raw is None or (isinstance(raw, str | bytes) and not raw.strip()):
            raise InvalidInputError(MANIFEST_MISSING)
        try:
            data = json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)
        except _NonFiniteError as exc:
            raise InvalidInputError(MANIFEST_NOT_FINITE) from exc
        except RecursionError as exc:
            raise InvalidInputError(MANIFEST_TOO_DEEP) from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise InvalidInputError(f"The manifest is not valid JSON ({exc}).") from exc
        if not isinstance(data, dict):
            raise InvalidInputError("The manifest must be a JSON object.")
        _check_consent(data.get("consent"))
        try:
            return UploadManifest.model_validate(data)
        except ValidationError as exc:
            raise InvalidInputError(format_validation_error(exc)) from exc
        except RecursionError as exc:
            raise InvalidInputError(MANIFEST_TOO_DEEP) from exc

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
                # The uploaded names are only checked for the SYNTHETIC prefix, never stored.
                source_filenames={slot: part.filename for slot, part in parts.items()},
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

    def _budget_message(self, used_pixels: int) -> str:
        limit = self.settings.max_decoded_megapixels
        return (
            f"The images in this upload are too large to analyse together "
            f"({used_pixels / 1e6:.0f} MP after resizing; the limit is {limit} MP). "
            "Upload fewer sites per screening or smaller images."
        )

    def create_screening(
        self, session: Session, manifest: UploadManifest, files: dict[str, FilePart]
    ) -> tuple[str, AnalysisResult, ScreeningResult]:
        """Decode, store and analyse one uploaded screening. Returns (id, row, result)."""
        _check_consent(manifest.consent)  # defence in depth: the router checked it too
        missing = [name for name, _ in manifest.file_fields() if name not in files]
        if missing:  # the router checks this too; keep the service safe on its own
            raise InvalidInputError(f"File field '{missing[0]}' is missing from the upload.")
        with self.analysis.analysis_slot():
            return self._create(session, manifest, files)

    def _create(
        self, session: Session, manifest: UploadManifest, files: dict[str, FilePart]
    ) -> tuple[str, AnalysisResult, ScreeningResult]:
        from mukha_pipeline.types import ScreeningInput

        now = utcnow()
        tz = self.settings.tz
        declared = DataProvenance(manifest.data_provenance)
        captured_at = local_to_utc(manifest.context.captured_at, tz)
        clock = captured_at or now
        consent_time = local_to_utc(manifest.consent.recorded_at, tz)
        consent = ConsentInfo.model_validate(
            {**manifest.consent.model_dump(), "recorded_at": consent_time or now}
        )
        error_types = self.bridge.input_error_types()
        try:
            adapter = self.bridge.upload_adapter_factory()
        except ImportError as exc:
            raise ServiceUnavailableError(PIPELINE_UNAVAILABLE) from exc

        # 1) decode everything first: nothing is written if any input is unreadable or the
        #    upload is over the decoded-pixel budget
        decoded = []
        pixels = 0
        for entry in manifest.sites:
            site_input, parts, exts = self._decode_site(
                adapter,
                entry,
                files,
                synthetic=declared == DataProvenance.SYNTHETIC,
                error_types=error_types,
            )
            for capture in _captures(site_input):
                pixels += int(capture.image.shape[0]) * int(capture.image.shape[1])
                when = capture.meta.captured_at
                if when is not None and when.tzinfo is None:  # EXIF time is camp-local
                    capture.meta = capture.meta.model_copy(
                        update={"captured_at": local_to_utc(when, tz)}
                    )
            if pixels > self.settings.max_decoded_pixels:
                raise PayloadTooLargeError(self._budget_message(pixels))
            decoded.append((entry, site_input, parts, exts))

        flags = [c.meta.synthetic for _e, s, _p, _x in decoded for c in _captures(s)]
        provenance = effective_provenance(declared, any(flags), bool(flags) and all(flags))
        synthetic = provenance == DataProvenance.SYNTHETIC
        if provenance != declared:
            logger.warning(
                "Upload declared as %s contains SYNTHETIC-labelled captures; stored as %s",
                declared.value,
                provenance.value,
            )

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
        context_in = manifest.context
        device_ids = list(context_in.device_ids) or sorted(
            {e.device_id for e in manifest.sites if e.device_id}
            | ({manifest.spectral_device_id} if manifest.spectral_device_id else set())
        )
        firmware = list(context_in.firmware_versions) or sorted(
            {e.firmware_version for e in manifest.sites if e.firmware_version}
        )
        context = ScreeningContext.model_validate(
            {
                **context_in.model_dump(),
                "device_ids": device_ids,
                "firmware_versions": firmware,
                "captured_at": captured_at,
            }
        )
        screening_id = new_id()
        inp = ScreeningInput(
            screening_id=screening_id,
            patient=PatientInfo(
                pseudonymous_id=new_pseudonymous_id(synthetic=synthetic),
                age_years=manifest.patient.age_years,
                sex=manifest.patient.sex,
            ),
            consent=consent,
            risk_factors=manifest.risk_factors,
            context=context,
            sites=[site_input for _entry, site_input, _parts, _exts in decoded],
            spectral=readings,
            spectral_calibrations=spectral_cals,
            data_provenance=provenance,
            now=clock,
        )

        # 3) store sanitised inputs + rows in one transaction
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
                    meta = getattr(site_input, slot).meta
                    orientation = meta.extra.get("exif_orientation")
                    clean = self.bridge.sanitize_image(
                        part.data,
                        orientation=int(orientation) if isinstance(orientation, int) else None,
                    )
                    capture = self.capture_store.add_capture(
                        session,
                        screening_id,
                        site=entry.site,
                        kind=kind,
                        data=clean.data,
                        ext=exts[slot],
                        meta={
                            "capture_metadata": meta.model_dump(mode="json"),
                            "source_adapter": SourceAdapter.UPLOAD.value,
                            # The original upload's sha256 is capture_metadata.file_sha256.
                            "metadata_sanitised": bool(clean.sanitised),
                            "metadata_removed": list(clean.removed),
                        },
                        synthetic=meta.synthetic,
                    )
                    refs[slot] = capture.id
                    n_captures += 1
                # Store the scale the user gave (original pixels); re-analysis converts it to
                # the decoded pixels again.
                site_entries.append(site_input_entry(site_input, refs, px_per_mm=entry.px_per_mm))
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
                    synthetic=synthetic,
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
                synthetic=synthetic or any(flags),
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
                    "data_provenance": provenance.value,
                    "declared_provenance": declared.value,
                    "synthetic_captures": sum(flags),
                    "consent_method": consent.method,
                    "consent_recorded_at_source": "client" if consent_time else "server",
                    "research_use_opt_in": consent.research_use_opt_in,
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
