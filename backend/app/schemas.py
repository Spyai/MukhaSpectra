"""API request/response models.

The manifest reuses the pipeline's ScreeningResult building blocks (consent, risk factors,
context, visual flag, enums) so the upload contract cannot drift from the result contract.
Identity fields (``patient.name``, ``patient.external_id``) are the only PII the API accepts; they
are stored in the separate ``patient_identities`` table and never appear in any response below.

Every free-text field has a length limit that matches its database column, so oversized input is
a 422 with the field's name rather than a database error. NaN / Infinity are rejected.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from mukha_pipeline.schema.models import (
    ConfigRef,
    ConsentInfo,
    DataProvenance,
    Diascopy,
    ModelInfo,
    RiskFactors,
    RiskLevel,
    ScreeningContext,
    ScreeningResult,
    Sex,
    SiteRole,
    VisualFlag,
)
from mukha_pipeline.sites import OralSite

_FIELD_NAME_PATTERN = r"^[A-Za-z0-9_.\-]{1,100}$"
CONSENT_METHODS: tuple[str, ...] = ("checkbox", "signature", "thumbprint")
CONSENT_METHOD_REQUIRED = (
    "Recorded consent needs its method: 'checkbox', 'signature' or 'thumbprint'."
)
MAX_LIST_ITEMS = 32

DeviceId = Annotated[str, StringConstraints(max_length=100)]
FirmwareVersion = Annotated[str, StringConstraints(max_length=50)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


# --------------------------------------------------------------------------------------------
# Upload manifest
# --------------------------------------------------------------------------------------------


class ManifestPatient(_Strict):
    name: str | None = Field(default=None, max_length=200, description="PII; stored separately.")
    external_id: str | None = Field(
        default=None, max_length=100, description="Hospital/programme id; PII, stored separately."
    )
    age_years: int | None = Field(default=None, ge=0, le=120)
    sex: Sex = Sex.UNDISCLOSED

    @field_validator("name", "external_id")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ManifestConsent(ConsentInfo):
    """``ConsentInfo`` with upload rules: recorded consent must say how it was given."""

    consent_id: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def _method_when_recorded(self) -> ManifestConsent:
        if self.consent_recorded and self.method not in CONSENT_METHODS:
            raise ValueError(CONSENT_METHOD_REQUIRED)
        return self


class ManifestContext(ScreeningContext):
    """``ScreeningContext`` with the length limits of the ``screenings`` columns.

    A ``captured_at`` without a UTC offset is camp-local time (settings.default_timezone); it is
    stored converted to UTC.
    """

    screening_site_name: str | None = Field(
        default=None, max_length=200, description="Camp or clinic name."
    )
    camp_id: str | None = Field(default=None, max_length=100)
    health_worker_id: str | None = Field(default=None, max_length=100)
    health_worker_name: str | None = Field(default=None, max_length=200)
    device_ids: list[DeviceId] = Field(default_factory=list, max_length=MAX_LIST_ITEMS)
    firmware_versions: list[FirmwareVersion] = Field(
        default_factory=list, max_length=MAX_LIST_ITEMS
    )


class ManifestSite(_Strict):
    site: OralSite
    role: SiteRole = SiteRole.EXAMINED
    wl: str = Field(pattern=_FIELD_NAME_PATTERN, description="Multipart field of the WL image.")
    af: str = Field(pattern=_FIELD_NAME_PATTERN, description="Multipart field of the AF image.")
    dark: str | None = Field(
        default=None, pattern=_FIELD_NAME_PATTERN, description="Optional dark-frame field."
    )
    visual_flag: VisualFlag = Field(default_factory=VisualFlag)
    diascopy: Diascopy = Diascopy.NOT_DONE
    px_per_mm: float | None = Field(default=None, gt=0, le=10_000)
    device_id: str | None = Field(default=None, max_length=100)
    firmware_version: str | None = Field(default=None, max_length=50)


class UploadManifest(_Strict):
    patient: ManifestPatient = Field(default_factory=ManifestPatient)
    consent: ManifestConsent
    risk_factors: RiskFactors = Field(default_factory=RiskFactors)
    context: ManifestContext = Field(default_factory=ManifestContext)
    data_provenance: DataProvenance = DataProvenance.REAL
    sites: list[ManifestSite] = Field(min_length=1, max_length=len(OralSite))
    spectral_file: str | None = Field(default=None, pattern=_FIELD_NAME_PATTERN)
    spectral_device_id: str | None = Field(
        default=None, max_length=100, description="Probe id when the spectral file has none."
    )

    @model_validator(mode="after")
    def _check_sites_and_fields(self) -> UploadManifest:
        seen: set[OralSite] = set()
        for entry in self.sites:
            if entry.site in seen:
                raise ValueError(f"Site '{entry.site.value}' is listed more than once.")
            seen.add(entry.site)
        if not any(entry.role == SiteRole.EXAMINED for entry in self.sites):
            raise ValueError("At least one site must have role 'examined'.")
        for name, _location in self.file_fields():
            if name == "manifest":
                raise ValueError("'manifest' cannot be used as a file field name.")
        return self

    def file_fields(self) -> list[tuple[str, str]]:
        """(multipart field name, manifest location) for every referenced file."""
        refs: list[tuple[str, str]] = []
        for i, entry in enumerate(self.sites):
            refs.append((entry.wl, f"sites[{i}].wl"))
            refs.append((entry.af, f"sites[{i}].af"))
            if entry.dark:
                refs.append((entry.dark, f"sites[{i}].dark"))
        if self.spectral_file:
            refs.append((self.spectral_file, "spectral_file"))
        return refs


# --------------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------------


class ResultVersion(BaseModel):
    version: int
    result_id: str
    created_at: datetime
    overall_risk_level: RiskLevel
    pipeline_version: str
    rule_config_version: str
    rule_config_sha256: str
    synthetic: bool
    supersedes_result_id: str | None = None


class ScreeningCreated(BaseModel):
    screening_id: str
    result_version: int
    result: ScreeningResult


class ScreeningSummary(BaseModel):
    """List view of a screening. Pseudonymous: no name or external id."""

    screening_id: str
    created_at: datetime
    status: str
    source: str
    data_provenance: DataProvenance
    synthetic: bool
    pseudonymous_id: str
    age_years: int | None = None
    sex: Sex = Sex.UNDISCLOSED
    screening_site_name: str | None = None
    camp_id: str | None = None
    health_worker_id: str | None = None
    site_count: int
    latest_result_version: int | None = None
    latest_result_id: str | None = None
    overall_risk_level: RiskLevel | None = None
    lesion_count: int | None = None
    review_status: str | None = None


class ScreeningDetail(BaseModel):
    screening_id: str
    created_at: datetime
    status: str
    latest: ScreeningResult | None
    versions: list[ResultVersion]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    notice: str


class ConfigInfo(ConfigRef):
    status: str | None = None


class VersionResponse(BaseModel):
    service: str
    api_version: str
    software: str = Field(description="Backend version + build id, as recorded in results.")
    build: str = Field(description="Build identifier (git commit or MUKHA_BUILD_ID).")
    pipeline: str
    schema_version: str
    configs: dict[str, ConfigInfo]
    models: list[ModelInfo]
    prototype_banner: str
    disclaimer: str
    authentication: Literal["none", "api_key"] = Field(
        description="'none': no authentication is configured on this server."
    )
    access_notice: str


class ErasureResponse(BaseModel):
    """What an erasure request removed (database rows, stored files, cached reports)."""

    erased_screening_ids: list[str]
    patient_record_erased: bool = Field(
        description="True when the patient row and the identity (name, external id) were "
        "removed; False when another screening still uses them."
    )
    stored_objects_deleted: int


class SiteShape(BaseModel):
    cx: float
    cy: float
    rx: float
    ry: float


class SiteCatalogueEntry(BaseModel):
    site: OralSite
    label: str
    short_label: str
    paired_site: OralSite | None
    pairing: str | None
    keratinized: bool
    shape: SiteShape


class SitesResponse(BaseModel):
    viewBox: str = Field(description="SVG viewBox of the site-map canvas.")  # noqa: N815
    width: float
    height: float
    orientation: str
    sites: list[SiteCatalogueEntry]


class ErrorResponse(BaseModel):
    detail: str
    screening_id: str | None = Field(
        default=None, description="Set when analysis failed after the captures were stored."
    )
