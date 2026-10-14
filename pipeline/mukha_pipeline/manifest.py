"""Build a ``ScreeningInput`` from an upload manifest (docs/architecture.md §6) and its files.

The same manifest format is used by the API upload endpoint, the sample data and the CLI, so a
folder of captures plus ``manifest.json`` can be analysed anywhere.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mukha_pipeline.capture import UploadAdapter
from mukha_pipeline.io import parse_spectral_file
from mukha_pipeline.schema.models import (
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
from mukha_pipeline.types import ScreeningInput, SiteInput

CONSENT_REQUIRED = "Consent is required before any data is stored."

FileReader = Callable[[str], tuple[bytes, str] | None]
"""Return (bytes, original file name) for a manifest file field, or None if absent."""


class ManifestError(ValueError):
    """The manifest is invalid; the message is plain language for the uploader."""


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ManifestPatient(_M):
    name: str | None = Field(default=None, max_length=200)
    external_id: str | None = Field(default=None, max_length=100)
    age_years: int | None = Field(default=None, ge=0, le=120)
    sex: Sex = Sex.UNDISCLOSED


class ManifestSite(_M):
    site: OralSite
    role: SiteRole = SiteRole.EXAMINED
    wl: str
    af: str
    dark: str | None = None
    visual_flag: VisualFlag = Field(default_factory=VisualFlag)
    diascopy: Diascopy = Diascopy.NOT_DONE
    px_per_mm: float | None = Field(default=None, gt=0)
    device_id: str | None = None
    firmware_version: str | None = None


class UploadManifest(_M):
    patient: ManifestPatient = Field(default_factory=ManifestPatient)
    consent: ConsentInfo
    risk_factors: RiskFactors = Field(default_factory=RiskFactors)
    context: ScreeningContext = Field(default_factory=ScreeningContext)
    data_provenance: DataProvenance = DataProvenance.REAL
    sites: list[ManifestSite] = Field(min_length=1)
    spectral_file: str | None = None


@dataclass
class ManifestInput:
    screening_input: ScreeningInput
    manifest: UploadManifest


def parse_manifest(data: Mapping[str, Any]) -> UploadManifest:
    """Validate a manifest; consent is checked first so nothing is processed without it."""
    consent = data.get("consent") if isinstance(data, Mapping) else None
    if not isinstance(consent, Mapping) or consent.get("consent_recorded") is not True:
        raise ManifestError(CONSENT_REQUIRED)
    try:
        manifest = UploadManifest.model_validate(data)
    except ValidationError as err:
        first = err.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise ManifestError(f"Invalid manifest at '{where}': {first.get('msg')}") from err
    sites = [s.site for s in manifest.sites]
    if len(sites) != len(set(sites)):
        raise ManifestError("Each oral site can appear only once in a screening.")
    return manifest


def build_screening_input(
    data: Mapping[str, Any],
    read_file: FileReader,
    *,
    screening_id: str | None = None,
    now: datetime | None = None,
    source_adapter: SourceAdapter = SourceAdapter.UPLOAD,
) -> ManifestInput:
    """Decode every referenced file and assemble the pipeline input."""
    manifest = parse_manifest(data)
    now = now or datetime.now(UTC)
    synthetic = manifest.data_provenance == DataProvenance.SYNTHETIC
    adapter = UploadAdapter()

    def need(field_name: str) -> tuple[bytes, str]:
        got = read_file(field_name)
        if got is None:
            raise ManifestError(f"File '{field_name}' is listed in the manifest but missing.")
        return got

    sites: list[SiteInput] = []
    for ms in manifest.sites:
        wl_bytes, wl_name = need(ms.wl)
        af_bytes, af_name = need(ms.af)
        dark_bytes, dark_name = need(ms.dark) if ms.dark else (None, None)
        sites.append(
            adapter.site_input(
                ms.site,
                wl_bytes,
                af_bytes,
                dark_bytes,
                filenames={"wl": wl_name, "af": af_name, "dark": dark_name},
                role=ms.role,
                visual_flag=ms.visual_flag,
                diascopy=ms.diascopy,
                px_per_mm=ms.px_per_mm,
                device_id=ms.device_id,
                firmware_version=ms.firmware_version,
                synthetic=synthetic,
            )
        )

    spectral = []
    spectral_cals = {}
    if manifest.spectral_file:
        spec_bytes, spec_name = need(manifest.spectral_file)
        parsed = parse_spectral_file(
            spec_bytes, spec_name, source_adapter=source_adapter, synthetic=synthetic, now=now
        )
        spectral = parsed.readings
        if parsed.calibration is not None:
            spectral_cals[parsed.calibration.device_id] = parsed.calibration

    context = manifest.context.model_copy(
        update={
            "device_ids": sorted({d for s in manifest.sites if (d := s.device_id)}),
            "firmware_versions": sorted({f for s in manifest.sites if (f := s.firmware_version)}),
        }
    )
    sid = screening_id or f"scr-{uuid.uuid4()}"
    inp = ScreeningInput(
        screening_id=sid,
        patient=PatientInfo(
            pseudonymous_id=f"P-{uuid.uuid5(uuid.NAMESPACE_URL, sid).hex[:12].upper()}",
            age_years=manifest.patient.age_years,
            sex=manifest.patient.sex,
        ),
        consent=manifest.consent,
        risk_factors=manifest.risk_factors,
        context=context,
        sites=sites,
        spectral=spectral,
        spectral_calibrations=spectral_cals,
        data_provenance=manifest.data_provenance,
        now=now,
    )
    return ManifestInput(screening_input=inp, manifest=manifest)
