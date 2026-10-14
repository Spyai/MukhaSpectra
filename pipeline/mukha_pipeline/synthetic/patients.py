"""Three SYNTHETIC demo patients (low / moderate / high) for demos, tests and the PDF report.

Each patient is a complete :class:`ScreeningInput`: consent, risk questionnaire, WL + AF + dark
captures per site, Confirm-probe readings and daily calibrations, all generated
deterministically from a seed. Every capture is ``synthetic=True`` with source adapter
SYNTHETIC, and the screening's provenance is SYNTHETIC, so results and reports carry the
synthetic notice. The "expected" levels describe which triage path each case was designed to
exercise; they are design intent, not measured performance, and must never be quoted as such.

Clinical sketches (typical presentations in Indian screening camps):

* low: 34-year-old woman, no tobacco, areca nut or alcohol; healthy buccal mucosa and lateral
  tongue on both sides; a small plaque (porphyrin) spot at the gingival margin; one pair of
  normal reference probe readings (calibrated).
* moderate: 46-year-old man chewing khaini for 12 years with a sharp molar cusp; homogeneous
  white patch on the left buccal mucosa with only mild, sub-threshold fluorescence dimming
  (common in homogeneous leukoplakia); contralateral buccal mucosa as reference; probe on the
  patch vs the right cheek WITHOUT a white-reference calibration (relative-signal path). The
  health worker's white-patch observation sets the MODERATE floor.
* high: 58-year-old man, gutka 20 years, bidi 15 years, daily alcohol; red patch with white
  specks and an irregular border on the right lateral tongue, a sore for more than 2 weeks,
  marked fluorescence loss; probe calibrated against a white reference standard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, get_args

import numpy as np

from mukha_pipeline.schema.models import (
    AlcoholUse,
    CaptureKind,
    CaptureMetadata,
    ConsentInfo,
    DataProvenance,
    Irritants,
    Language,
    LedMode,
    PatientInfo,
    RiskFactors,
    ScreeningContext,
    Sex,
    SiteRole,
    SmokelessTobaccoUse,
    SmokingUse,
    SourceAdapter,
    Symptoms,
    VisualAppearance,
    VisualFlag,
)
from mukha_pipeline.sites import OralSite
from mukha_pipeline.synthetic.af import generate_af_from_wl, generate_dark_frame
from mukha_pipeline.synthetic.lesions import LesionSpec
from mukha_pipeline.synthetic.spectra import (
    SYNTHETIC_FIRMWARE,
    SYNTHETIC_PROBE_ID,
    generate_spectral_reading,
    generate_white_reference,
    make_spectral_calibration,
)
from mukha_pipeline.synthetic.wl import DEFAULT_SIZE, generate_wl_scene
from mukha_pipeline.types import (
    ImageCalibration,
    ImageCapture,
    ScreeningInput,
    SiteInput,
    SpectralFrame,
)

DemoKind = Literal["low", "moderate", "high"]
DEMO_KINDS: tuple[str, ...] = get_args(DemoKind)
EXPECTED_LEVEL: dict[str, str] = {"low": "LOW", "moderate": "MODERATE", "high": "HIGH"}

SYNTHETIC_SCOUT_ID = "SYN-SCOUT-01"
DEMO_CAMP_NAME = "SYNTHETIC DEMO CAMP"
DEMO_CAMP_ID = "SYN-CAMP-01"
DEMO_HEALTH_WORKER_ID = "SYN-HW-01"
DEMO_CAPTURED_AT = datetime(2026, 10, 14, 9, 30, tzinfo=UTC)

# Scout capture settings, locked for the session so AF intensities are comparable across sites.
WL_EXPOSURE_MS, WL_ISO = 8.0, 100.0
AF_EXPOSURE_MS, AF_ISO = 33.0, 400.0  # the dark frame uses the AF exposure
WHITE_BALANCE = "locked:5000K"
# Dark pedestal of exactly 3/255: same physics as any level, and it keeps the 8-bit dark PNGs
# tiny because the noise rounds to a constant.
DEMO_DARK_LEVEL = 3.0 / 255.0
_SITE_STRIDE_S = 40  # seconds between sites in the capture timeline


@dataclass(frozen=True)
class DemoSite:
    site: OralSite
    role: SiteRole = SiteRole.EXAMINED
    lesion: LesionSpec | None = None
    visual_flag: VisualFlag = field(default_factory=VisualFlag)
    porphyrin_spots: int = 0


@dataclass(frozen=True)
class DemoSpectral:
    reading_id: str
    site: OralSite
    role: Literal["lesion", "reference"]
    hb: float
    reference_reading_id: str | None = None


@dataclass(frozen=True)
class DemoPatientSpec:
    kind: str
    pseudonymous_id: str
    display_name: str
    age_years: int
    sex: Sex
    risk_factors: RiskFactors
    sites: tuple[DemoSite, ...]
    spectral: tuple[DemoSpectral, ...]
    spectral_calibrated: bool
    image_calibrated: bool = True
    teeth_stain: float = 0.0


@dataclass
class DemoBundle:
    """A demo patient plus the raw white-reference frames (needed for the spectrum CSV)."""

    spec: DemoPatientSpec
    screening: ScreeningInput
    white_reference: tuple[list[SpectralFrame], list[SpectralFrame]] | None = None


# --------------------------------------------------------------------------------------------
# Case definitions
# --------------------------------------------------------------------------------------------

_S = OralSite


def _low() -> DemoPatientSpec:
    pid = "SYN-LOW-001"
    return DemoPatientSpec(
        kind="low",
        pseudonymous_id=pid,
        display_name="SYNTHETIC PATIENT LOW",
        age_years=34,
        sex=Sex.FEMALE,
        risk_factors=RiskFactors(),
        sites=(
            DemoSite(_S.BUCCAL_MUCOSA_LEFT),
            DemoSite(_S.BUCCAL_MUCOSA_RIGHT, porphyrin_spots=1),
            DemoSite(_S.LATERAL_TONGUE_LEFT),
            DemoSite(_S.LATERAL_TONGUE_RIGHT),
        ),
        spectral=(
            DemoSpectral(f"{pid}-S1", _S.BUCCAL_MUCOSA_LEFT, "reference", 0.24),
            DemoSpectral(f"{pid}-S2", _S.BUCCAL_MUCOSA_RIGHT, "reference", 0.26),
        ),
        spectral_calibrated=True,
    )


def _moderate() -> DemoPatientSpec:
    pid = "SYN-MOD-001"
    patch = LesionSpec(
        kind="white_patch",
        center=(0.50, 0.42),
        radii=(0.14, 0.095),
        irregularity=0.3,
        intensity=0.8,
        # Mild, sub-threshold dimming (relative AF ~0.87). Homogeneous leukoplakia often keeps
        # its fluorescence; with triage rules v1 any detected FVL here (+ the visual white
        # patch, khaini and age >= 40) already scores HIGH, which is not this case's purpose.
        fvl_depth=0.15,
        angle_deg=-8.0,
    )
    flag = VisualFlag(
        lesion_seen=True,
        appearance=[VisualAppearance.WHITE_PATCH],
        description=(
            "SYNTHETIC case: homogeneous white patch on the left buccal mucosa next to a sharp "
            "molar cusp; does not wipe off."
        ),
    )
    return DemoPatientSpec(
        kind="moderate",
        pseudonymous_id=pid,
        display_name="SYNTHETIC PATIENT MODERATE",
        age_years=46,
        sex=Sex.MALE,
        risk_factors=RiskFactors(
            smokeless_tobacco=SmokelessTobaccoUse(uses=True, types=["khaini"], years=12, per_day=6),
            irritants=Irritants(sharp_tooth=True),
        ),
        sites=(
            DemoSite(_S.BUCCAL_MUCOSA_LEFT, lesion=patch, visual_flag=flag),
            DemoSite(_S.BUCCAL_MUCOSA_RIGHT, role=SiteRole.REFERENCE_ONLY, porphyrin_spots=1),
            DemoSite(_S.LATERAL_TONGUE_LEFT),
            DemoSite(_S.LATERAL_TONGUE_RIGHT),
        ),
        spectral=(
            DemoSpectral(f"{pid}-S1", _S.BUCCAL_MUCOSA_LEFT, "lesion", 0.36, f"{pid}-S2"),
            DemoSpectral(f"{pid}-S2", _S.BUCCAL_MUCOSA_RIGHT, "reference", 0.25),
        ),
        spectral_calibrated=False,
        teeth_stain=0.35,
    )


def _high() -> DemoPatientSpec:
    pid = "SYN-HIGH-001"
    patch = LesionSpec(
        kind="mixed_red_white",
        center=(0.45, 0.45),
        radii=(0.15, 0.09),
        irregularity=0.75,
        intensity=0.85,
        fvl_depth=0.6,
        angle_deg=6.0,
    )
    flag = VisualFlag(
        lesion_seen=True,
        appearance=[VisualAppearance.RED_PATCH, VisualAppearance.MIXED_RED_WHITE],
        description=(
            "SYNTHETIC case: red patch with white specks and an irregular border on the right "
            "lateral tongue; patient reports a sore there for more than 2 weeks."
        ),
    )
    return DemoPatientSpec(
        kind="high",
        pseudonymous_id=pid,
        display_name="SYNTHETIC PATIENT HIGH",
        age_years=58,
        sex=Sex.MALE,
        risk_factors=RiskFactors(
            smokeless_tobacco=SmokelessTobaccoUse(uses=True, types=["gutka"], years=20, per_day=8),
            smoking=SmokingUse(uses=True, types=["bidi"], years=15, per_day=12),
            alcohol=AlcoholUse(uses=True, frequency="daily"),
            symptoms=Symptoms(non_healing_ulcer_over_2_weeks=True, burning_sensation=True),
        ),
        sites=(
            DemoSite(_S.LATERAL_TONGUE_RIGHT, lesion=patch, visual_flag=flag),
            DemoSite(_S.LATERAL_TONGUE_LEFT, role=SiteRole.REFERENCE_ONLY, porphyrin_spots=1),
            DemoSite(_S.BUCCAL_MUCOSA_LEFT, porphyrin_spots=2),
            DemoSite(_S.BUCCAL_MUCOSA_RIGHT, porphyrin_spots=2),
        ),
        spectral=(
            DemoSpectral(f"{pid}-S1", _S.LATERAL_TONGUE_RIGHT, "lesion", 0.60, f"{pid}-S2"),
            DemoSpectral(f"{pid}-S2", _S.LATERAL_TONGUE_LEFT, "reference", 0.25),
        ),
        spectral_calibrated=True,
        teeth_stain=0.6,
    )


_SPECS = {"low": _low, "moderate": _moderate, "high": _high}


def demo_patient_spec(kind: str) -> DemoPatientSpec:
    """The case definition behind a demo patient."""
    if kind not in _SPECS:
        raise ValueError(f"unknown demo patient {kind!r}; expected one of {DEMO_KINDS}")
    return _SPECS[kind]()


# --------------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------------


def _meta(
    *,
    capture_id: str,
    kind: CaptureKind,
    site: OralSite,
    led: LedMode,
    exposure_ms: float,
    iso: float,
    captured_at: datetime,
    calibration_id: str | None,
    size: tuple[int, int],
    file_name: str,
    mime_type: str,
) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=capture_id,
        kind=kind,
        oral_site=site,
        source_adapter=SourceAdapter.SYNTHETIC,
        device_id=SYNTHETIC_SCOUT_ID,
        firmware_version=SYNTHETIC_FIRMWARE,
        led_mode=led,
        exposure_time_ms=exposure_ms,
        iso=iso,
        white_balance=WHITE_BALANCE,
        focus="locked",
        captured_at=captured_at,
        calibration_id=calibration_id,
        width_px=size[1],
        height_px=size[0],
        file_name=file_name,
        mime_type=mime_type,
        synthetic=True,
        extra={"generator": "mukha_pipeline.synthetic"},
    )


def capture_file_name(site: OralSite | str, kind: Literal["wl", "af", "dark"]) -> str:
    """File name used in sample_data and in capture metadata: SYNTHETIC_<site>_<kind>.<ext>."""
    ext = "png" if kind == "dark" else "jpg"
    return f"SYNTHETIC_{OralSite(site).value}_{kind}.{ext}"


def _site_input(
    spec: DemoPatientSpec,
    demo: DemoSite,
    rng: np.random.Generator,
    *,
    size: tuple[int, int],
    t0: datetime,
    calibration_id: str | None,
) -> SiteInput:
    scene = generate_wl_scene(
        rng, size, site=demo.site, lesion=demo.lesion, teeth_stain=spec.teeth_stain
    )
    af = generate_af_from_wl(
        scene.wl,
        rng,
        tissue_mask=scene.tissue_mask,
        teeth_mask=scene.teeth_mask,
        fvl_mask=scene.lesion_mask if demo.lesion else None,
        fvl_depth=demo.lesion.fvl_depth if demo.lesion else 0.0,
        porphyrin_spots=demo.porphyrin_spots,
        specular_mask=scene.specular_mask,
        dark_level=DEMO_DARK_LEVEL,
    )
    dark = generate_dark_frame(size, rng, level=DEMO_DARK_LEVEL)
    base_id = f"{spec.pseudonymous_id}-{demo.site.value}"
    # Synced capture order on the Scout: dark, then WL, then AF, a second apart.
    dark_meta = _meta(
        capture_id=f"{base_id}-dark",
        kind=CaptureKind.DARK_FRAME,
        led=LedMode.OFF,
        exposure_ms=AF_EXPOSURE_MS,
        iso=AF_ISO,
        captured_at=t0,
        calibration_id=None,
        file_name=capture_file_name(demo.site, "dark"),
        mime_type="image/png",
        site=demo.site,
        size=size,
    )
    wl_meta = _meta(
        capture_id=f"{base_id}-wl",
        kind=CaptureKind.WHITE_LIGHT,
        led=LedMode.WHITE,
        exposure_ms=WL_EXPOSURE_MS,
        iso=WL_ISO,
        captured_at=t0 + timedelta(seconds=1),
        calibration_id=calibration_id,
        file_name=capture_file_name(demo.site, "wl"),
        mime_type="image/jpeg",
        site=demo.site,
        size=size,
    )
    af_meta = _meta(
        capture_id=f"{base_id}-af",
        kind=CaptureKind.AUTOFLUORESCENCE,
        led=LedMode.VIOLET,
        exposure_ms=AF_EXPOSURE_MS,
        iso=AF_ISO,
        captured_at=t0 + timedelta(seconds=2),
        calibration_id=calibration_id,
        file_name=capture_file_name(demo.site, "af"),
        mime_type="image/jpeg",
        site=demo.site,
        size=size,
    )
    return SiteInput(
        site=demo.site,
        wl=ImageCapture(scene.wl, wl_meta),
        af=ImageCapture(af, af_meta),
        dark=ImageCapture(dark, dark_meta),
        role=demo.role,
        visual_flag=demo.visual_flag.model_copy(deep=True),
    )


def _image_calibration(created_at: datetime, size: tuple[int, int]) -> ImageCalibration:
    """SYNTHETIC daily Scout calibration: neutral flat field and colour, unit AF scale."""
    return ImageCalibration(
        calibration_id=f"SYN-IMGCAL-{SYNTHETIC_SCOUT_ID}-{created_at:%Y%m%dT%H%M}",
        device_id=SYNTHETIC_SCOUT_ID,
        created_at=created_at,
        expires_at=created_at + timedelta(hours=24),
        flat_field=np.ones(size, np.float32),
        color_matrix=np.eye(3, dtype=np.float64),
        af_scale=1.0,
        synthetic=True,
    )


def make_demo_bundle(
    kind: str,
    seed: int = 0,
    *,
    now: datetime | None = None,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> DemoBundle:
    """Build a demo patient and keep its raw white-reference frames (if calibrated).

    ``now`` moves the whole capture timeline (default: a fixed date, for reproducibility);
    calibrations are created 30 minutes before the first capture and stay valid for 24 h.
    """
    spec = demo_patient_spec(kind)
    kind_idx = DEMO_KINDS.index(kind)
    captured_at = now or DEMO_CAPTURED_AT
    cal_created = captured_at - timedelta(minutes=30)
    image_cal = _image_calibration(cal_created, size) if spec.image_calibrated else None
    sites = [
        _site_input(
            spec,
            demo,
            np.random.default_rng([seed, kind_idx, 1, i]),
            size=size,
            t0=captured_at + timedelta(seconds=_SITE_STRIDE_S * i),
            calibration_id=image_cal.calibration_id if image_cal else None,
        )
        for i, demo in enumerate(spec.sites)
    ]
    spec_rng = np.random.default_rng([seed, kind_idx, 2])
    white_ref = generate_white_reference(spec_rng) if spec.spectral_calibrated else None
    spectral_cal = (
        make_spectral_calibration(*white_ref, created_at=cal_created) if white_ref else None
    )
    probe_time = captured_at + timedelta(seconds=_SITE_STRIDE_S * len(sites) + 60)
    readings = [
        generate_spectral_reading(
            spec_rng,
            hb=r.hb,
            reading_id=r.reading_id,
            site=r.site,
            role=r.role,
            reference_reading_id=r.reference_reading_id,
            captured_at=probe_time + timedelta(seconds=20 * j),
            calibration_id=spectral_cal.calibration_id if spectral_cal else None,
        )
        for j, r in enumerate(spec.spectral)
    ]
    device_ids = [SYNTHETIC_SCOUT_ID] + ([SYNTHETIC_PROBE_ID] if readings else [])
    screening = ScreeningInput(
        screening_id=f"SYN-SCREENING-{kind.upper()}-001",
        patient=PatientInfo(
            pseudonymous_id=spec.pseudonymous_id, age_years=spec.age_years, sex=spec.sex
        ),
        consent=ConsentInfo(
            consent_recorded=True,
            method="checkbox",
            language=Language.EN,
            research_use_opt_in=False,
            consent_id=f"SYN-CONSENT-{kind.upper()}-001",
            recorded_at=captured_at - timedelta(minutes=10),
        ),
        risk_factors=spec.risk_factors.model_copy(deep=True),
        context=ScreeningContext(
            screening_site_name=DEMO_CAMP_NAME,
            camp_id=DEMO_CAMP_ID,
            health_worker_id=DEMO_HEALTH_WORKER_ID,
            device_ids=device_ids,
            firmware_versions=[SYNTHETIC_FIRMWARE],
            captured_at=captured_at,
            language=Language.EN,
        ),
        sites=sites,
        spectral=readings,
        image_calibrations={SYNTHETIC_SCOUT_ID: image_cal} if image_cal else {},
        spectral_calibrations={SYNTHETIC_PROBE_ID: spectral_cal} if spectral_cal else {},
        data_provenance=DataProvenance.SYNTHETIC,
        now=probe_time + timedelta(minutes=5),
    )
    return DemoBundle(spec=spec, screening=screening, white_reference=white_ref)


def make_demo_patient(
    kind: str,
    seed: int = 0,
    *,
    now: datetime | None = None,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> ScreeningInput:
    """A SYNTHETIC demo patient ("low" | "moderate" | "high") as a ScreeningInput."""
    return make_demo_bundle(kind, seed, now=now, size=size).screening
