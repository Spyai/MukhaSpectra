"""SYNTHETIC demo patients as complete ScreeningInputs."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from synthetic_helpers import SMALL, assert_positioning_clean

from mukha_pipeline.schema.models import (
    CaptureKind,
    DataProvenance,
    LedMode,
    SiteRole,
    SourceAdapter,
    VisualAppearance,
)
from mukha_pipeline.sites import OralSite, reference_site_for
from mukha_pipeline.synthetic import (
    DEMO_KINDS,
    EXPECTED_LEVEL,
    demo_patient_spec,
    make_demo_bundle,
    make_demo_patient,
)
from mukha_pipeline.types import ScreeningInput


@pytest.fixture(scope="module")
def patients() -> dict[str, ScreeningInput]:
    return {kind: make_demo_patient(kind, size=SMALL) for kind in DEMO_KINDS}


def _captures(inp: ScreeningInput):
    for s in inp.sites:
        yield s.wl
        yield s.af
        if s.dark is not None:
            yield s.dark


@pytest.mark.parametrize("kind", DEMO_KINDS)
def test_demo_patient_is_synthetic_everywhere(patients, kind: str) -> None:
    inp = patients[kind]
    assert isinstance(inp, ScreeningInput)
    assert inp.data_provenance is DataProvenance.SYNTHETIC
    assert inp.patient.pseudonymous_id.startswith("SYN-")
    assert inp.screening_id.startswith("SYN-")
    assert inp.context.screening_site_name == "SYNTHETIC DEMO CAMP"
    assert inp.context.health_worker_id == "SYN-HW-01"
    for cap in _captures(inp):
        m = cap.meta
        assert m.synthetic is True and m.source_adapter is SourceAdapter.SYNTHETIC
        assert m.device_id == "SYN-SCOUT-01" and m.firmware_version == "synthetic"
        assert m.file_name and m.file_name.startswith("SYNTHETIC_")
        assert m.captured_at is not None and m.led_mode is not None
    for r in inp.spectral:
        assert r.meta.synthetic is True and r.meta.source_adapter is SourceAdapter.SYNTHETIC
        assert r.meta.device_id == "SYN-CONFIRM-01"
    for cal in inp.image_calibrations.values():
        assert cal.synthetic is True
    for cal in inp.spectral_calibrations.values():
        assert cal.synthetic is True


@pytest.mark.parametrize("kind", DEMO_KINDS)
def test_each_site_has_wl_af_and_dark_with_locked_settings(patients, kind: str) -> None:
    inp = patients[kind]
    assert len(inp.sites) == 4
    af_settings = set()
    for s in inp.sites:
        assert s.wl.image.shape == (*SMALL, 3) and s.af.image.shape == (*SMALL, 3)
        assert s.dark is not None and s.dark.image.shape == (*SMALL, 3)
        assert s.wl.meta.kind is CaptureKind.WHITE_LIGHT and s.wl.meta.led_mode is LedMode.WHITE
        assert s.af.meta.kind is CaptureKind.AUTOFLUORESCENCE
        assert s.af.meta.led_mode is LedMode.VIOLET
        assert s.dark.meta.kind is CaptureKind.DARK_FRAME and s.dark.meta.led_mode is LedMode.OFF
        af_settings.add((s.af.meta.exposure_time_ms, s.af.meta.iso))
        assert (s.dark.meta.exposure_time_ms, s.dark.meta.iso) == (
            s.af.meta.exposure_time_ms,
            s.af.meta.iso,
        ), "dark frame uses the AF exposure"
        for img in (s.wl.image, s.af.image, s.dark.image):
            assert img.dtype == np.float32 and img.min() >= 0.0 and img.max() <= 1.0
    assert len(af_settings) == 1, "AF exposure identical across sites"
    assert None not in next(iter(af_settings))


@pytest.mark.parametrize("kind", DEMO_KINDS)
def test_consent_and_calibrations_are_valid_at_now(patients, kind: str) -> None:
    inp = patients[kind]
    assert inp.consent.consent_recorded is True and inp.consent.method == "checkbox"
    assert inp.now is not None
    cal = inp.image_calibrations["SYN-SCOUT-01"]
    assert cal.created_at <= inp.now < cal.expires_at
    assert cal.flat_field is not None and cal.flat_field.shape == SMALL
    assert np.allclose(cal.flat_field, 1.0) and np.allclose(cal.color_matrix, np.eye(3))
    assert cal.af_flat_field is not None and np.allclose(cal.af_flat_field, 1.0)
    assert cal.af_scale == 1.0
    for s in inp.sites:
        assert s.wl.meta.calibration_id == cal.calibration_id
    for spec_cal in inp.spectral_calibrations.values():
        assert spec_cal.created_at <= inp.now < spec_cal.expires_at


def test_low_case(patients) -> None:
    inp = patients["low"]
    assert (inp.patient.age_years, inp.patient.sex.value) == (34, "female")
    rf = inp.risk_factors
    assert not (rf.smokeless_tobacco.uses or rf.smoking.uses or rf.areca_nut.uses)
    assert not any(s.visual_flag.lesion_seen for s in inp.sites)
    assert {s.site for s in inp.sites} == {
        OralSite.BUCCAL_MUCOSA_LEFT,
        OralSite.BUCCAL_MUCOSA_RIGHT,
        OralSite.LATERAL_TONGUE_LEFT,
        OralSite.LATERAL_TONGUE_RIGHT,
    }
    assert [r.role for r in inp.spectral] == ["reference", "reference"]
    assert {r.site for r in inp.spectral} == {
        OralSite.BUCCAL_MUCOSA_LEFT,
        OralSite.BUCCAL_MUCOSA_RIGHT,
    }
    assert EXPECTED_LEVEL["low"] == "LOW"


def test_moderate_case(patients) -> None:
    inp = patients["moderate"]
    assert (inp.patient.age_years, inp.patient.sex.value) == (46, "male")
    rf = inp.risk_factors
    assert rf.smokeless_tobacco.uses and rf.smokeless_tobacco.types == ["khaini"]
    assert rf.smokeless_tobacco.years == 12 and rf.irritants.sharp_tooth
    lesion_site = inp.sites[0]
    assert lesion_site.site is OralSite.BUCCAL_MUCOSA_LEFT
    assert lesion_site.visual_flag.lesion_seen
    assert lesion_site.visual_flag.appearance == [VisualAppearance.WHITE_PATCH]
    ref = next(s for s in inp.sites if s.site is OralSite.BUCCAL_MUCOSA_RIGHT)
    assert ref.role is SiteRole.REFERENCE_ONLY
    assert inp.spectral_calibrations == {}, "relative-signal path"
    lesion, reference = inp.spectral
    assert lesion.role == "lesion" and lesion.reference_reading_id == reference.reading_id
    assert reference_site_for(lesion.site)[0] == reference.site
    spec = demo_patient_spec("moderate")
    assert spec.sites[0].lesion is not None and spec.sites[0].lesion.kind == "white_patch"
    assert spec.spectral[0].hb == pytest.approx(0.36)


def test_high_case(patients) -> None:
    inp = patients["high"]
    assert (inp.patient.age_years, inp.patient.sex.value) == (58, "male")
    rf = inp.risk_factors
    assert rf.smokeless_tobacco.types == ["gutka"] and rf.smokeless_tobacco.years == 20
    assert rf.smoking.types == ["bidi"] and rf.smoking.years == 15
    assert rf.alcohol.frequency == "daily"
    assert rf.symptoms.non_healing_ulcer_over_2_weeks
    lesion_site = inp.sites[0]
    assert lesion_site.site is OralSite.LATERAL_TONGUE_RIGHT
    assert VisualAppearance.RED_PATCH in lesion_site.visual_flag.appearance
    ref = next(s for s in inp.sites if s.site is OralSite.LATERAL_TONGUE_LEFT)
    assert ref.role is SiteRole.REFERENCE_ONLY
    assert "SYN-CONFIRM-01" in inp.spectral_calibrations
    lesion, reference = inp.spectral
    assert lesion.reference_reading_id == reference.reading_id
    spec = demo_patient_spec("high")
    assert spec.sites[0].lesion is not None and spec.sites[0].lesion.fvl_depth == 0.6
    assert spec.spectral[0].hb == pytest.approx(0.6) and spec.spectral[1].hb == 0.25


def test_lesion_sites_show_fluorescence_loss(patients) -> None:
    high = patients["high"].sites[0]
    ref = patients["high"].sites[1]
    assert np.median(high.af.image[..., 1]) < np.median(ref.af.image[..., 1])


def test_demo_patient_is_deterministic() -> None:
    a = make_demo_patient("moderate", seed=3, size=SMALL)
    b = make_demo_patient("moderate", seed=3, size=SMALL)
    c = make_demo_patient("moderate", seed=4, size=SMALL)
    assert np.array_equal(a.sites[0].wl.image, b.sites[0].wl.image)
    assert np.array_equal(a.sites[0].af.image, b.sites[0].af.image)
    assert a.spectral[0].lit[0].channels == b.spectral[0].lit[0].channels
    assert not np.array_equal(a.sites[0].wl.image, c.sites[0].wl.image)


def test_now_moves_the_whole_timeline() -> None:
    now = datetime(2030, 1, 2, 3, 4, tzinfo=UTC)
    bundle = make_demo_bundle("high", size=SMALL, now=now)
    inp = bundle.screening
    assert inp.context.captured_at == now
    assert all(s.wl.meta.captured_at and s.wl.meta.captured_at >= now for s in inp.sites)
    cal = inp.image_calibrations["SYN-SCOUT-01"]
    assert inp.now is not None and cal.created_at < inp.now < cal.expires_at
    assert bundle.white_reference is not None and len(bundle.white_reference[0]) == 5


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown demo patient"):
        make_demo_patient("extreme")


@pytest.mark.parametrize("kind", DEMO_KINDS)
def test_descriptions_are_positioning_clean(kind: str) -> None:
    spec = demo_patient_spec(kind)
    for site in spec.sites:
        if site.visual_flag.description:
            assert "SYNTHETIC" in site.visual_flag.description
            assert_positioning_clean(site.visual_flag.description, kind)
    assert_positioning_clean(spec.display_name, kind)
    assert spec.display_name.startswith("SYNTHETIC PATIENT")
