"""End-to-end pipeline: ScreeningInput → ScreeningResult + artifacts (all inputs SYNTHETIC)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest
from wlspec_helpers import MODEL_CARD, build_tiny_onnx

from mukha_pipeline.config import PipelineConfig
from mukha_pipeline.orchestrator import analyze_screening
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.preprocess.correction import linear_to_srgb, srgb_to_linear
from mukha_pipeline.preprocess.segmentation import ClassicalSegmenter
from mukha_pipeline.schema.export import SCHEMA_PATH
from mukha_pipeline.schema.models import (
    CalibrationStatus,
    DataProvenance,
    ModelComponent,
    ModelStatus,
    QCSeverity,
    RiskFactors,
    RiskLevel,
    SiteRole,
    VisualAppearance,
    VisualFlag,
)
from mukha_pipeline.sites import OralSite
from mukha_pipeline.synthetic import make_demo_patient, stamp_mask
from mukha_pipeline.types import Masks, ScreeningInput, SiteInput
from mukha_pipeline.wl import load_lesion_classifier

NOW = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
SIZE = (480, 640)  # smaller than the demo default to keep the suite fast
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def demo(kind: str) -> ScreeningInput:
    return make_demo_patient(kind, seed=0, now=NOW, size=SIZE)


@pytest.fixture(scope="module")
def outputs():
    return {k: analyze_screening(demo(k), strict=True) for k in ("low", "moderate", "high")}


@pytest.mark.parametrize(
    ("kind", "level"),
    [("low", RiskLevel.LOW), ("moderate", RiskLevel.MODERATE), ("high", RiskLevel.HIGH)],
)
def test_demo_patients_reach_expected_levels(outputs, kind, level):
    result = outputs[kind].result
    assert result.triage.overall_risk_level == level
    assert result.triage.requires_clinician_review is True
    assert result.triage.review_status == "pending_review"
    assert result.triage.explanation


@pytest.mark.parametrize("kind", ["low", "moderate", "high"])
def test_result_validates_against_json_schema(outputs, kind):
    payload = json.loads(outputs[kind].result.model_dump_json())
    jsonschema.Draft202012Validator(SCHEMA).validate(payload)


@pytest.mark.parametrize("kind", ["low", "moderate", "high"])
def test_result_text_respects_positioning(outputs, kind):
    assert find_positioning_violations(outputs[kind].result.model_dump_json()) == []


def test_high_case_finds_fvl_against_contralateral_reference(outputs):
    result = outputs["high"].result
    tongue = next(s for s in result.sites if s.site.value == "lateral_tongue_right")
    assert tongue.analysis_status == "analysed"
    assert tongue.af is not None and tongue.af.regions
    assert tongue.af.reference.type.value == "contralateral"
    assert tongue.af.regions[0].mean_relative_af < tongue.af.fvl_threshold
    assert tongue.site_risk_level == RiskLevel.HIGH
    assert result.triage.highest_risk_lesion_id is not None
    lesion = next(
        lz for lz in result.lesions if lz.lesion_id == result.triage.highest_risk_lesion_id
    )
    assert sum(f.points for f in lesion.contributing_factors) == pytest.approx(lesion.score)
    assert any(f.code.startswith("af.") for f in lesion.contributing_factors)


def test_every_referenced_artifact_exists(outputs):
    for out in outputs.values():
        for site in out.result.sites:
            keys = [v for v in site.artifacts.model_dump().values() if v]
            assert keys, site.site
            for key in keys:
                assert key in out.artifacts, key
            for capture in site.captures:
                if capture.artifact:
                    assert capture.artifact in out.artifacts


def test_artifacts_decode(outputs):
    arts = outputs["high"].artifacts
    heat = next(k for k in arts if k.endswith("heatmap_overlay.jpg"))
    img = cv2.imdecode(np.frombuffer(arts[heat], np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[2] == 3
    grid_key = next(k for k in arts if k.endswith("relative_af_grid.json"))
    grid = json.loads(arts[grid_key])
    assert grid["width"] > 0 and grid["height"] > 0 and len(grid["values"]) == grid["height"]


def test_synthetic_is_labelled_everywhere(outputs):
    result = outputs["low"].result
    assert result.synthetic is True
    assert result.data_provenance == DataProvenance.SYNTHETIC
    assert result.notices.synthetic_notice
    assert "synthetic_data" in {lim.code for lim in result.limitations}


def test_versions_and_models_are_recorded(outputs):
    result = outputs["low"].result
    cfg = PipelineConfig.default()
    assert result.versions.rule_config.sha256 == cfg.triage.ref.sha256
    assert result.versions.qc_config.sha256 == cfg.qc.ref.sha256
    assert result.versions.analysis_config.sha256 == cfg.analysis.ref.sha256
    components = {m.component: m for m in result.versions.models}
    assert set(components) == set(ModelComponent)
    assert components[ModelComponent.WL_LESION_CLASSIFIER].status == ModelStatus.MODEL_UNAVAILABLE
    assert all(m.validated is False for m in result.versions.models)
    assert result.processing.stages and result.processing.duration_ms > 0


def test_calibrated_demo_is_calibrated(outputs):
    assert outputs["high"].result.calibration_status == CalibrationStatus.CALIBRATED


def test_missing_calibration_is_reported():
    inp = demo("low")
    inp.image_calibrations = {}
    inp.spectral_calibrations = {}
    result = analyze_screening(inp, strict=True).result
    assert result.calibration_status == CalibrationStatus.UNCALIBRATED
    assert "uncalibrated" in {lim.code for lim in result.limitations}
    assert all(s.calibration.status == CalibrationStatus.UNCALIBRATED for s in result.sites)


def test_blurred_af_makes_low_patient_inconclusive():
    inp = demo("low")
    target = inp.sites[0]
    target.af.image = cv2.GaussianBlur(target.af.image, (0, 0), 12) * 0.02
    result = analyze_screening(inp, strict=True).result
    failed = next(s for s in result.sites if s.site == target.site)
    assert failed.analysis_status == "inconclusive_qc_failed"
    assert not failed.qc.passed
    assert failed.site_risk_level == RiskLevel.INCONCLUSIVE
    assert result.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert target.site in result.qc_summary.sites_failed
    assert result.qc_summary.retakes_recommended


def test_missing_dark_frame_is_a_warning_not_a_failure():
    inp = demo("low")
    for s in inp.sites:
        s.dark = None
    result = analyze_screening(inp, strict=True).result
    assert result.triage.overall_risk_level == RiskLevel.LOW
    assert "no_dark_frame" in {lim.code for lim in result.limitations}


class _ExplodingSegmenter:
    name = "exploding"
    version = "0"

    def segment(self, wl):
        raise RuntimeError("boom")


def test_crash_in_one_stage_is_contained_unless_strict():
    inp = demo("low")
    result = analyze_screening(inp, segmenter=_ExplodingSegmenter()).result
    assert all(s.analysis_status == "not_analysed" for s in result.sites)
    assert any(i.code == "analysis_error" for s in result.sites for i in s.qc.issues)
    assert result.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    with pytest.raises(RuntimeError):
        analyze_screening(demo("low"), segmenter=_ExplodingSegmenter(), strict=True)


def test_software_crash_is_reported_as_a_software_problem_not_bad_images():
    """honesty.2: a crash is not a capture-quality failure and must not send the health worker
    to retake photos that will crash again; it says to examine clinically and report it."""
    result = analyze_screening(demo("low"), segmenter=_ExplodingSegmenter()).result
    message = (
        "Software error while analysing this site — examine clinically and report the problem."
    )
    for s in result.sites:
        assert s.analysis_status == "not_analysed"
        issue = next(i for i in s.qc.issues if i.code == "analysis_error")
        assert issue.message == message and issue.severity == QCSeverity.ERROR
        assert "retake" not in issue.message.lower()
        assert s.af is not None and "software" in (s.af.reference.note or "").lower()
        assert "quality" not in (s.af.reference.note or "").lower()
        assert s.wl is not None and s.wl.status == "error"
        assert "software" in (s.wl.note or "").lower()
        assert s.site_risk_level == RiskLevel.INCONCLUSIVE
    assert not any("Software error" in r for r in result.qc_summary.retakes_recommended)
    lim = {lim.code: lim.message for lim in result.limitations}
    assert "analysis_error" in lim and "software" in lim["analysis_error"].lower()
    assert result.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    assert result.qc_summary.passed_all is False  # never "all sites passed" after a crash
    assert find_positioning_violations(result.model_dump_json()) == []


class _ExplodingClassifier:
    """WL classifier that crashes in pass 2 (after FVL detection already ran)."""

    def __init__(self) -> None:
        from mukha_pipeline.wl import UnavailableLesionClassifier

        self.info = UnavailableLesionClassifier().info

    def classify(self, wl, mucosa=None):
        raise RuntimeError("boom")


def test_crash_in_the_second_pass_is_also_a_software_error():
    result = analyze_screening(demo("low"), classifier=_ExplodingClassifier()).result
    for s in result.sites:
        assert s.analysis_status == "not_analysed"
        assert s.af is not None and s.af.status == "skipped_qc_failed" and not s.af.regions
        assert "software" in (s.af.reference.note or "").lower()
        assert s.wl is not None and s.wl.status == "error"
    assert result.triage.overall_risk_level == RiskLevel.INCONCLUSIVE


def test_onnx_classifier_output_and_saliency(tmp_path: Path, analysis_cfg):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    build_tiny_onnx(model_dir / "model.onnx")
    (model_dir / "model_card.md").write_text(MODEL_CARD, encoding="utf-8")
    clf = load_lesion_classifier(model_dir, analysis_cfg["wl_classifier"])
    inp = demo("low")
    inp.sites = inp.sites[:2]
    out = analyze_screening(inp, classifier=clf, strict=True)
    site = out.result.sites[0]
    assert site.wl is not None and site.wl.status == "ok"
    assert site.wl.model.validated is False  # the card's claim is ignored
    assert site.wl.saliency_artifact in out.artifacts


def test_result_ids_and_versioning_pass_through():
    inp = demo("low")
    inp.result_id = "res-fixed"
    inp.result_version = 3
    inp.supersedes_result_id = "res-previous"
    result = analyze_screening(inp, strict=True).result
    assert (result.result_id, result.result_version, result.supersedes_result_id) == (
        "res-fixed",
        3,
        "res-previous",
    )
    assert result.created_at == inp.now


def test_real_provenance_with_synthetic_captures_becomes_mixed():
    inp = demo("low")
    inp.data_provenance = DataProvenance.REAL
    result = analyze_screening(inp, strict=True).result
    assert result.synthetic is True
    assert result.data_provenance == DataProvenance.MIXED


# --------------------------------------------------------------------------------------------
# Review fixes: helpers
# --------------------------------------------------------------------------------------------


def _site(inp: ScreeningInput, site: OralSite) -> SiteInput:
    return next(s for s in inp.sites if s.site == site)


def _result_site(result, site: OralSite):
    return next(s for s in result.sites if s.site == site)


def _relight_af(site: SiteInput, gain) -> None:
    """Scale the fluorescence LIGHT of an (sRGB-encoded) AF capture by ``gain`` (scalar or
    (H, W) map), as a different exposure or excitation profile would; the camera-fixed stamp
    text stays identical to the WL stamp, as it would in a real capture app."""
    enc = site.af.image
    lin = srgb_to_linear(enc)
    dark = srgb_to_linear(site.dark.image) if site.dark is not None else np.zeros_like(lin)
    g = np.asarray(gain, np.float32)
    g = g[..., None] if g.ndim == 2 else g
    out = linear_to_srgb(dark + np.clip(lin - dark, 0.0, None) * g)
    stamp = stamp_mask(enc.shape)
    out[stamp] = enc[stamp]
    site.af.image = out


def _profile(drop: float, cx: float = 0.5, cy: float = 0.5) -> np.ndarray:
    """Smooth illumination profile at SIZE: 1 near (cx, cy), ``1 - drop`` at the far corner."""
    h, w = SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx / w - cx) * (w / h)) ** 2 + (yy / h - cy) ** 2
    return (1.0 - drop * r2 / r2.max()).astype(np.float32)


def _strip_clinical_context(inp: ScreeningInput) -> None:
    """Only device evidence left: no visual flags, symptoms, habits or probe readings."""
    for s in inp.sites:
        s.visual_flag = VisualFlag()
    inp.risk_factors = RiskFactors()
    inp.spectral = []
    inp.spectral_calibrations = {}


# --------------------------------------------------------------------------------------------
# science.0: AF radiometry runs in linear light
# --------------------------------------------------------------------------------------------


def test_fvl_depth_is_measured_in_linear_light(outputs):
    """The high case loses 60 % of its green LIGHT. Analysed on encoded values the same lesion
    reads as only ~30 % loss; linearising first recovers the modelled depth."""
    tongue = _result_site(outputs["high"].result, OralSite.LATERAL_TONGUE_RIGHT)
    assert tongue.af is not None and tongue.af.regions
    deepest = min(r.mean_relative_af for r in tongue.af.regions)
    assert deepest < 0.55


def test_device_alone_flags_the_high_lesion_from_encoded_files():
    """Same scene without the worker's observation, symptoms or habits: the fluorescence loss
    alone must keep the patient off LOW (encoded analysis reported LOW here)."""
    inp = demo("high")
    _strip_clinical_context(inp)
    result = analyze_screening(inp, strict=True).result
    tongue = _result_site(result, OralSite.LATERAL_TONGUE_RIGHT)
    assert tongue.af is not None and tongue.af.regions
    assert result.triage.overall_risk_level in (RiskLevel.MODERATE, RiskLevel.HIGH)


def test_unequal_contralateral_exposure_is_normalised_in_linear_light(outputs):
    """Reference side captured at 1.5x exposure, recorded in the metadata: the exposure factor
    must bring it back exactly, so neither side changes (encoded values put the loss on the
    healthy side)."""
    inp = demo("high")
    ref = _site(inp, OralSite.LATERAL_TONGUE_LEFT)
    _relight_af(ref, 1.5)
    ref.af.meta = ref.af.meta.model_copy(
        update={"exposure_time_ms": ref.af.meta.exposure_time_ms * 1.5}
    )
    result = analyze_screening(inp, strict=True).result
    base = outputs["high"].result
    left = _result_site(result, OralSite.LATERAL_TONGUE_LEFT)
    right = _result_site(result, OralSite.LATERAL_TONGUE_RIGHT)
    base_right = _result_site(base, OralSite.LATERAL_TONGUE_RIGHT)
    assert left.af is not None and not left.af.regions
    assert right.af is not None and right.af.regions
    assert right.af.reference.reference_green == pytest.approx(
        base_right.af.reference.reference_green, rel=0.03
    )
    assert min(r.mean_relative_af for r in right.af.regions) == pytest.approx(
        min(r.mean_relative_af for r in base_right.af.regions), abs=0.03
    )
    assert result.triage.overall_risk_level == RiskLevel.HIGH


def test_artifacts_shown_to_people_are_srgb_encoded(outputs):
    """af.jpg is re-encoded for display: it matches the encoded capture, not linear values."""
    out = outputs["low"]
    site = out.result.sites[0]
    img = cv2.imdecode(np.frombuffer(out.artifacts[site.artifacts.af_image], np.uint8), 1)
    shown = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    capture = cv2.resize(demo("low").sites[0].af.image, (shown.shape[1], shown.shape[0]))
    tissue = capture[..., 1] > 0.3
    assert float(np.median(np.abs(shown[..., 1] - capture[..., 1])[tissue])) < 0.03


# --------------------------------------------------------------------------------------------
# science.4: AF gets the violet-LED flat field, WL the white one
# --------------------------------------------------------------------------------------------


def _with_flats(inp: ScreeningInput, white: np.ndarray, violet: np.ndarray | None) -> None:
    cal = next(iter(inp.image_calibrations.values()))
    cal.flat_field = white / white.mean()
    cal.af_flat_field = None if violet is None else violet / violet.mean()


def test_af_is_flat_fielded_with_the_violet_profile():
    """WL captured under a white-LED profile, AF under a different (steeper, off-centre) violet
    profile. Each must be corrected by its own flat field; dividing AF by the white profile
    leaves corner fall-off that reads as fluorescence loss (LOW became MODERATE)."""
    inp = demo("low")
    white = _profile(0.15)
    violet = _profile(0.45, cx=0.6, cy=0.4)
    for s in inp.sites:
        s.wl.image = np.clip(s.wl.image * white[..., None], 0.0, 1.0).astype(np.float32)
        _relight_af(s, violet)
    _with_flats(inp, white, violet)
    result = analyze_screening(inp, strict=True).result
    for s in result.sites:
        assert s.af is not None and not s.af.regions, (s.site, s.af.regions)
    assert result.triage.overall_risk_level == RiskLevel.LOW
    assert "af_flat_field_missing" not in {lim.code for lim in result.limitations}


def test_af_is_never_flat_fielded_with_the_white_profile():
    """No violet profile measured: AF stays uncorrected (and a limitation says so) rather than
    being divided by the white-LED profile, which would brighten its rim and dim the centre."""
    inp = demo("low")
    white = _profile(0.5)
    for s in inp.sites:
        s.wl.image = np.clip(s.wl.image * white[..., None], 0.0, 1.0).astype(np.float32)
    _with_flats(inp, white, None)
    result = analyze_screening(inp, strict=True).result
    for s in result.sites:
        assert s.af is not None and not s.af.regions, (s.site, s.af.regions)
    limitations = {lim.code: lim.message for lim in result.limitations}
    assert "af_flat_field_missing" in limitations
    assert find_positioning_violations(limitations["af_flat_field_missing"]) == []
    uncalibrated = demo("low")
    uncalibrated.image_calibrations = {}
    lims = {lim.code for lim in analyze_screening(uncalibrated, strict=True).result.limitations}
    assert "af_flat_field_missing" not in lims  # 'uncalibrated' already says it


# --------------------------------------------------------------------------------------------
# correctness.0 / correctness.1: capture settings and segmentation status reach QC
# --------------------------------------------------------------------------------------------


def test_auto_white_balance_on_af_is_reported():
    inp = demo("low")
    target = inp.sites[0]
    target.af.meta = target.af.meta.model_copy(update={"white_balance": "auto"})
    result = analyze_screening(inp, strict=True).result
    site = _result_site(result, target.site)
    issue = next(i for i in site.qc.issues if i.code == "af_auto_white_balance")
    assert issue.severity == QCSeverity.WARNING
    af_capture = next(c for c in site.captures if c.metadata.kind.value == "autofluorescence")
    assert "af_auto_white_balance" in {i.code for i in af_capture.qc.issues}
    assert site.analysis_status == "analysed"


def test_greyscale_af_blocks_the_site():
    inp = demo("low")
    target = inp.sites[0]
    target.af.image = np.repeat(target.af.image.mean(axis=2, keepdims=True), 3, axis=2)
    result = analyze_screening(inp, strict=True).result
    site = _result_site(result, target.site)
    assert site.analysis_status == "inconclusive_qc_failed"
    assert result.triage.overall_risk_level == RiskLevel.INCONCLUSIVE


class _ConfidenceSegmenter:
    """The classical segmenter with its confidence overridden (simulates blanched tissue the
    colour rules cannot classify)."""

    name = "classical_test"
    version = "0"

    def __init__(self, confidence: float, analysis_cfg: dict) -> None:
        self.inner = ClassicalSegmenter(analysis_cfg.get("segmentation"))
        self.confidence = confidence

    def segment(self, wl):
        m = self.inner.segment(wl)
        return Masks(m.mucosa, m.teeth, m.background, m.specular, self.confidence)


def test_failed_segmentation_makes_the_site_unanalysable(analysis_cfg):
    result = analyze_screening(
        demo("low"), segmenter=_ConfidenceSegmenter(0.0, analysis_cfg), strict=True
    ).result
    for s in result.sites:
        assert s.segmentation is not None and s.segmentation.status == "failed"
        assert s.analysis_status == "inconclusive_qc_failed"
        assert "segmentation_failed" in {i.code for i in s.qc.issues}
    assert result.triage.overall_risk_level == RiskLevel.INCONCLUSIVE


def test_low_confidence_segmentation_warns_and_cannot_give_low(analysis_cfg):
    result = analyze_screening(
        demo("low"), segmenter=_ConfidenceSegmenter(0.3, analysis_cfg), strict=True
    ).result
    for s in result.sites:
        assert s.segmentation is not None and s.segmentation.status == "low_confidence"
        assert s.analysis_status == "analysed"
        issue = next(i for i in s.qc.issues if i.code == "segmentation_low_confidence")
        assert issue.severity == QCSeverity.WARNING
    assert result.triage.overall_risk_level != RiskLevel.LOW


# --------------------------------------------------------------------------------------------
# clinical.2: reference selection
# --------------------------------------------------------------------------------------------


def test_site_with_a_recorded_lesion_is_never_a_reference():
    inp = demo("high")
    ref = _site(inp, OralSite.LATERAL_TONGUE_LEFT)
    ref.visual_flag = VisualFlag(lesion_seen=True, appearance=[VisualAppearance.RED_PATCH])
    result = analyze_screening(inp, strict=True).result
    right = _result_site(result, OralSite.LATERAL_TONGUE_RIGHT)
    assert right.af is not None
    assert right.af.reference.reference_site != OralSite.LATERAL_TONGUE_LEFT


def test_examined_site_with_a_recorded_appearance_is_never_a_reference():
    inp = demo("low")
    other = _site(inp, OralSite.BUCCAL_MUCOSA_RIGHT)
    assert other.role == SiteRole.EXAMINED
    other.visual_flag = VisualFlag(lesion_seen=False, appearance=[VisualAppearance.WHITE_PATCH])
    result = analyze_screening(inp, strict=True).result
    left = _result_site(result, OralSite.BUCCAL_MUCOSA_LEFT)
    assert left.af is not None
    assert left.af.reference.reference_site != OralSite.BUCCAL_MUCOSA_RIGHT


def test_dimmer_opposite_side_is_reported_and_cannot_hide_a_loss():
    """Opposite side 20 % dimmer (angle, exposure or bilateral change): the patch on the left
    must still be found, and a limitation must say both sides need examining."""
    inp = demo("low")
    left = _site(inp, OralSite.BUCCAL_MUCOSA_LEFT)
    h, w = SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    patch = ((xx - 0.45 * w) / (0.12 * w)) ** 2 + ((yy - 0.45 * h) / (0.12 * h)) ** 2 <= 1.0
    patch_gain = np.where(patch, 0.6, 1.0).astype(np.float32)
    _relight_af(left, cv2.GaussianBlur(patch_gain, (0, 0), 4))
    _relight_af(_site(inp, OralSite.BUCCAL_MUCOSA_RIGHT), 0.8)
    result = analyze_screening(inp, strict=True).result
    left_result = _result_site(result, OralSite.BUCCAL_MUCOSA_LEFT)
    assert left_result.af is not None and left_result.af.regions
    assert result.triage.overall_risk_level != RiskLevel.LOW
    limitations = {lim.code: lim.message for lim in result.limitations}
    assert "possible_bilateral_change" in limitations
    message = limitations["possible_bilateral_change"]
    assert "Examine both sides clinically." in message
    assert left_result.site_label in message
    assert find_positioning_violations(message) == []


def test_no_bilateral_limitation_for_balanced_sides(outputs):
    for out in outputs.values():
        assert "possible_bilateral_change" not in {lim.code for lim in out.result.limitations}


# --------------------------------------------------------------------------------------------
# clinical.6: white areas the colour rules call "teeth" are checked against AF
# --------------------------------------------------------------------------------------------


def _white_patch_at_border(inp: ScreeningInput, af_gain: float) -> SiteInput:
    """A greyish-white plaque running out of the right edge of the left buccal close-up (colour
    rules call it teeth: low saturation, touches the border)."""
    site = _site(inp, OralSite.BUCCAL_MUCOSA_LEFT)
    h, w = SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    patch = ((xx - w) / (0.30 * w)) ** 2 + ((yy - 0.5 * h) / (0.28 * h)) ** 2 <= 1.0
    wl = site.wl.image.copy()
    wl[patch] = np.array([225, 222, 212], np.float32) / 255.0
    site.wl.image = wl
    _relight_af(site, np.where(patch, af_gain, 1.0).astype(np.float32))
    return site


def test_border_white_patch_with_fluorescence_loss_is_analysed():
    inp = demo("low")
    _strip_clinical_context(inp)
    _white_patch_at_border(inp, af_gain=0.45)
    result = analyze_screening(inp, strict=True).result
    left = _result_site(result, OralSite.BUCCAL_MUCOSA_LEFT)
    assert left.af is not None and left.af.regions, "the white patch must reach the AF analysis"
    assert result.triage.overall_risk_level != RiskLevel.LOW


def test_large_white_area_left_out_is_reported():
    """A white area that fluoresces like teeth stays excluded; the report must say so."""
    inp = demo("low")
    _white_patch_at_border(inp, af_gain=2.2)
    result = analyze_screening(inp, strict=True).result
    limitations = {lim.code: lim.message for lim in result.limitations}
    assert "white_area_not_analysed" in limitations
    assert find_positioning_violations(limitations["white_area_not_analysed"]) == []


def test_white_area_limitation_names_only_sites_with_excluded_white_areas(outputs):
    """Teeth beside the buccal mucosa glow like teeth and stay excluded, so those sites are
    named (a clinician checks that it is only teeth); the tongue views have no such area."""
    result = outputs["low"].result
    message = next(
        lim.message for lim in result.limitations if lim.code == "white_area_not_analysed"
    )
    labels = {s.site: s.site_label for s in result.sites}
    assert labels[OralSite.BUCCAL_MUCOSA_LEFT] in message
    assert labels[OralSite.LATERAL_TONGUE_LEFT] not in message
    assert labels[OralSite.LATERAL_TONGUE_RIGHT] not in message
    assert find_positioning_violations(message) == []


def test_working_scale_is_relative_to_the_original_capture_after_decode_downscaling(outputs):
    """Uploads are shrunk while decoding; scale_from_original must still mean working width /
    ORIGINAL capture width, not working width / decoded width."""
    inp = demo("low")
    for s in inp.sites:
        for cap in (s.wl, s.af):
            cap.meta = cap.meta.model_copy(
                update={"extra": {**cap.meta.extra, "decode_scale": 0.5}}
            )
    result = analyze_screening(inp, strict=True).result
    base = {s.site: s.working_resolution for s in outputs["low"].result.sites}
    for s in result.sites:
        assert s.working_resolution is not None
        expected = base[s.site].scale_from_original * 0.5
        assert s.working_resolution.scale_from_original == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["low", "moderate", "high"])
def test_generated_text_never_reassures_or_calls_tissue_healthy(outputs, kind):
    """Reference tissue is a comparison, not 'healthy'; a quiet result is never 'all clear' or
    'no referral'; the placeholder spectral label never reads 'not suspicious'."""
    text = outputs[kind].result.model_dump_json().lower()
    for phrase in ("healthy", "no referral", "not suspicious", "all clear", "model output"):
        assert phrase not in text, phrase
