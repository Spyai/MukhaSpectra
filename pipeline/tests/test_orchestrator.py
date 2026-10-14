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
from mukha_pipeline.schema.export import SCHEMA_PATH
from mukha_pipeline.schema.models import (
    CalibrationStatus,
    DataProvenance,
    ModelComponent,
    ModelStatus,
    RiskLevel,
)
from mukha_pipeline.synthetic import make_demo_patient
from mukha_pipeline.types import ScreeningInput
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
    assert all(s.analysis_status == "inconclusive_qc_failed" for s in result.sites)
    assert any(i.code == "analysis_error" for s in result.sites for i in s.qc.issues)
    assert result.triage.overall_risk_level == RiskLevel.INCONCLUSIVE
    with pytest.raises(RuntimeError):
        analyze_screening(demo("low"), segmenter=_ExplodingSegmenter(), strict=True)


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
