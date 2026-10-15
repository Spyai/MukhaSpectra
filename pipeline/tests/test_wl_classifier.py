"""Tests for mukha_pipeline.wl (classifier interface, ONNX runner, occlusion saliency)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from wlspec_helpers import MODEL_CARD, build_tiny_onnx

from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import ModelComponent, ModelStatus, WLAnalysis, WLLabel
from mukha_pipeline.wl import (
    LesionClassifier,
    OnnxLesionClassifier,
    UnavailableLesionClassifier,
    load_lesion_classifier,
    mucosa_crop_box,
    occlusion_saliency,
    to_probabilities,
)
from mukha_pipeline.wl.classifier import card_field

RED_PATCH = 2


@pytest.fixture
def wl_cfg(analysis_cfg) -> dict:
    return dict(analysis_cfg["wl_classifier"])


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    build_tiny_onnx(tmp_path / "model.onnx", size=32)
    (tmp_path / "model_card.md").write_text(MODEL_CARD)
    return tmp_path


def _grey_with_red_square(h: int = 96, w: int = 128) -> np.ndarray:
    img = np.full((h, w, 3), 0.5, dtype=np.float32)
    img[8:40, 8:40] = (1.0, 0.0, 0.0)
    return img


def _assert_clean(analysis: WLAnalysis) -> None:
    for text in (analysis.note or "", analysis.model.notes or ""):
        assert not find_positioning_violations(text), text


# --------------------------------------------------------------------------------------------
# Unavailable
# --------------------------------------------------------------------------------------------


def test_unavailable_classifier(rng):
    clf = UnavailableLesionClassifier()
    assert isinstance(clf, LesionClassifier)
    info = clf.info
    assert info.component == ModelComponent.WL_LESION_CLASSIFIER
    assert (info.name, info.version, info.status) == (
        "wl-lesion-classifier",
        "none",
        ModelStatus.MODEL_UNAVAILABLE,
    )
    assert info.validated is False
    assert info.notes == (
        "No trained weights installed. White-light lesion classification is not used in triage."
    )
    out = clf.classify(rng.random((50, 60, 3)).astype(np.float32))
    assert out.status == "model_unavailable" and out.predictions == [] and out.top_label is None
    _assert_clean(out)


@pytest.mark.parametrize("make_dir", [None, "empty", "no_card"])
def test_loader_falls_back_to_unavailable(tmp_path: Path, wl_cfg: dict, make_dir: str | None):
    if make_dir is None:
        clf = load_lesion_classifier(None, wl_cfg)
    else:
        if make_dir == "no_card":
            build_tiny_onnx(tmp_path / "model.onnx")
        clf = load_lesion_classifier(tmp_path, wl_cfg)
    assert isinstance(clf, UnavailableLesionClassifier)
    if make_dir == "no_card":
        assert "model_card.md" in (clf.info.notes or "")


# --------------------------------------------------------------------------------------------
# ONNX runner
# --------------------------------------------------------------------------------------------


def test_onnx_classifier_available_path(model_dir: Path, wl_cfg: dict):
    clf = load_lesion_classifier(model_dir, wl_cfg)
    assert isinstance(clf, OnnxLesionClassifier) and isinstance(clf, LesionClassifier)
    info = clf.info
    assert info.status == ModelStatus.AVAILABLE
    assert info.validated is False  # the card claims validated: true; code must ignore it
    assert info.version == "0.0.1-test" and info.name == "wl-tiny-test-model"
    assert info.notes == "Not clinically validated."
    assert info.sha256 and len(info.sha256) == 64
    assert info.model_card and info.model_card.endswith("model_card.md")
    assert clf.input_size == 32  # the model's fixed input size wins over config (224)

    out = clf.classify(_grey_with_red_square())
    assert out.status == "ok"
    probs = [p.probability for p in out.predictions]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) == pytest.approx(1.0, abs=1e-5)
    assert {p.label for p in out.predictions} == set(WLLabel)
    assert out.top_label == out.predictions[0].label
    _assert_clean(out)

    red = clf.predict_proba(np.dstack([np.ones((40, 40)), np.zeros((40, 40)), np.zeros((40, 40))]))
    grey = clf.predict_proba(np.full((40, 40, 3), 0.5))
    assert red[RED_PATCH] > grey[RED_PATCH]


def test_mask_crop_changes_what_the_model_sees(model_dir: Path, wl_cfg: dict):
    clf = load_lesion_classifier(model_dir, wl_cfg)
    img = np.full((100, 100, 3), 0.5, dtype=np.float32)
    img[60:90, 60:90] = (1.0, 0.0, 0.0)
    mask = np.zeros((100, 100), dtype=bool)
    mask[60:90, 60:90] = True
    whole = clf.classify(img)
    cropped = clf.classify(img, mask)
    p_whole = {p.label: p.probability for p in whole.predictions}[WLLabel.RED_PATCH]
    p_crop = {p.label: p.probability for p in cropped.predictions}[WLLabel.RED_PATCH]
    assert p_crop > p_whole

    box, note = mucosa_crop_box(img.shape, mask, 0.10)
    assert box == (57, 93, 57, 93) and note is None


def test_crop_box_fallbacks():
    shape = (50, 40, 3)
    assert mucosa_crop_box(shape, None) == ((0, 50, 0, 40), None)
    box, note = mucosa_crop_box(shape, np.ones((10, 10), dtype=bool))
    assert box == (0, 50, 0, 40) and "did not match" in (note or "")
    tiny = np.zeros((50, 40), dtype=bool)
    tiny[0, 0] = True
    box, note = mucosa_crop_box(shape, tiny)
    assert box == (0, 50, 0, 40) and "Too little mucosa" in (note or "")
    edge = np.zeros((50, 40), dtype=bool)
    edge[:20, :20] = True
    assert mucosa_crop_box(shape, edge, 0.5)[0] == (0, 30, 0, 30)


def test_classify_with_bad_mask_notes_fallback(model_dir: Path, wl_cfg: dict, rng):
    clf = load_lesion_classifier(model_dir, wl_cfg)
    out = clf.classify(rng.random((60, 60)), np.zeros((5, 5), dtype=bool))  # gray image too
    assert out.status == "ok" and "did not match" in (out.note or "")
    _assert_clean(out)


def test_logit_model_gets_softmax_and_dynamic_input(tmp_path: Path, wl_cfg: dict):
    path = build_tiny_onnx(tmp_path / "model.onnx", size="S", softmax=False)
    clf = OnnxLesionClassifier(path, {"wl_classifier": {**wl_cfg, "input_size": 48}})
    assert clf.input_size == 48 and clf.info.version == "unknown"
    assert clf.info.name == "wl-lesion-classifier" and clf.info.model_card is None
    probs = clf.predict_proba(_grey_with_red_square())
    assert probs.sum() == pytest.approx(1.0) and np.all(probs > 0)


def test_card_given_as_text_or_path_string(tmp_path: Path, wl_cfg: dict):
    path = build_tiny_onnx(tmp_path / "model.onnx")
    clf = OnnxLesionClassifier(path, wl_cfg, model_card="version: 2.0\n")
    assert clf.info.version == "2.0"
    card = tmp_path / "card.md"
    card.write_text(MODEL_CARD)
    clf = OnnxLesionClassifier(path, wl_cfg, model_card=str(card))
    assert clf.info.version == "0.0.1-test" and clf.info.model_card == f"{tmp_path.name}/card.md"
    missing = OnnxLesionClassifier(path, wl_cfg, model_card=tmp_path / "nope.md")
    assert missing.info.version == "unknown"


def test_card_field_parsing():
    assert card_field(None, "version") is None
    assert card_field("---\nversion: '1.2'\n---\nversion: 9", "version") == "1.2"
    assert card_field("---\nname: x\n", "version") is None
    assert card_field("# Card\nVersion: 3\n", "version") == "3"
    assert card_field("version:   \n", "version") is None


def test_corrupt_model_file_gives_error_status(tmp_path: Path, wl_cfg: dict, rng):
    (tmp_path / "model.onnx").write_bytes(b"this is not an onnx model")
    (tmp_path / "model_card.md").write_text(MODEL_CARD)
    clf = load_lesion_classifier(tmp_path, wl_cfg)
    assert isinstance(clf, OnnxLesionClassifier)
    assert clf.info.status == ModelStatus.MODEL_UNAVAILABLE and clf.load_error
    assert clf.info.validated is False
    out = clf.classify(rng.random((40, 40, 3)))
    assert out.status == "error" and out.predictions == []
    assert "could not be loaded" in (out.note or "")
    assert clf.saliency(rng.random((40, 40, 3))) is None
    with pytest.raises(RuntimeError):
        clf.predict_proba(rng.random((40, 40, 3)))
    _assert_clean(out)


def test_missing_model_file_gives_error_status(tmp_path: Path, wl_cfg: dict):
    clf = OnnxLesionClassifier(tmp_path / "absent.onnx", wl_cfg)
    assert clf.load_error and clf.info.sha256 is None
    assert clf.classify(np.zeros((40, 40, 3))).status == "error"


def test_output_label_mismatch_is_an_error_not_a_crash(tmp_path: Path, wl_cfg: dict):
    path = build_tiny_onnx(tmp_path / "model.onnx", n_out=4)
    clf = OnnxLesionClassifier(path, wl_cfg)
    assert clf.info.status == ModelStatus.AVAILABLE
    out = clf.classify(np.zeros((40, 40, 3)))
    assert out.status == "error" and "4 scores" in (out.note or "")
    _assert_clean(out)


def test_non_finite_output_is_an_error(model_dir: Path, wl_cfg: dict, monkeypatch):
    clf = load_lesion_classifier(model_dir, wl_cfg)

    class NaNSession:
        def run(self, *args, **kwargs):
            return [np.full((1, 5), np.nan, dtype=np.float32)]

    monkeypatch.setattr(clf, "_session", NaNSession())
    out = clf.classify(np.zeros((40, 40, 3)))
    assert out.status == "error" and "non-finite" in (out.note or "")


def test_model_input_shape_checks(tmp_path: Path, wl_cfg: dict):
    path = build_tiny_onnx(tmp_path / "model.onnx", channels=1)
    clf = OnnxLesionClassifier(path, wl_cfg)
    assert clf.load_error and "input channels" in clf.load_error


def test_bad_label_config_is_a_load_error(model_dir: Path, wl_cfg: dict):
    clf = OnnxLesionClassifier(model_dir / "model.onnx", {**wl_cfg, "labels": ["normal", "cancer"]})
    assert clf.load_error and "Unknown WL label" in clf.load_error
    assert clf.classify(np.zeros((40, 40, 3))).status == "error"


def test_to_probabilities():
    np.testing.assert_allclose(to_probabilities(np.array([0.2, 0.8])), [0.2, 0.8])
    p = to_probabilities(np.array([1000.0, 0.0, -1000.0]))
    assert np.isfinite(p).all() and p[0] == pytest.approx(1.0)
    q = to_probabilities(np.array([0.5, 0.5, 0.5]))  # sums to 1.5: treated as logits
    np.testing.assert_allclose(q, 1 / 3)


# --------------------------------------------------------------------------------------------
# Saliency
# --------------------------------------------------------------------------------------------


def test_occlusion_saliency_highlights_the_evidence(model_dir: Path, wl_cfg: dict):
    clf = load_lesion_classifier(model_dir, wl_cfg)
    img = _grey_with_red_square()
    sal = occlusion_saliency(clf.predict_proba, img, RED_PATCH, wl_cfg)
    assert sal.shape == img.shape[:2] and sal.dtype == np.float32
    assert sal.min() >= 0.0 and sal.max() == pytest.approx(1.0)
    inside = sal[8:40, 8:40].mean()
    outside = sal[60:, 70:].mean()
    assert inside > 0.5 and outside < 0.05
    peak = np.unravel_index(np.argmax(sal), sal.shape)
    assert 0 <= peak[0] < 48 and 0 <= peak[1] < 48


def test_saliency_is_resampled_for_large_images_and_handles_edge_cases(wl_cfg: dict):
    calls: list[tuple[int, ...]] = []

    def predict(x: np.ndarray) -> np.ndarray:
        calls.append(x.shape)
        r = float(x[..., 0].mean())
        return np.array([1 - r, r])

    big = np.full((900, 1200, 3), 0.2, dtype=np.float32)
    big[100:300, 100:300, 0] = 1.0
    sal = occlusion_saliency(predict, big, 1, {"wl_classifier": wl_cfg})
    assert sal.shape == (900, 1200)
    assert max(calls[0][:2]) <= 448  # occlusion runs at reduced resolution
    assert sal[150:250, 150:250].mean() > sal[600:, 900:].mean()

    flat = occlusion_saliency(predict, np.full((20, 30, 3), 0.5), 0, {})
    assert flat.shape == (20, 30) and flat.max() == 0.0  # no evidence -> all zero

    tiny = occlusion_saliency(predict, np.random.default_rng(0).random((3, 2, 3)), 1, {})
    assert tiny.shape == (3, 2)
    assert occlusion_saliency(predict, np.zeros((0, 5, 3)), 0, {}).shape == (0, 5)

    with pytest.raises(ValueError, match="target_index"):
        occlusion_saliency(predict, np.zeros((10, 10, 3)), 5, {})


def test_saliency_restores_input_even_if_predict_fails():
    img = np.full((16, 16, 3), 0.3, dtype=np.float32)
    original = img.copy()
    state = {"n": 0}

    def flaky(x: np.ndarray) -> np.ndarray:
        state["n"] += 1
        if state["n"] > 1:
            raise RuntimeError("boom")
        return np.array([1.0])

    with pytest.raises(RuntimeError):
        occlusion_saliency(flaky, img, 0, {})
    np.testing.assert_array_equal(img, original)


def test_classifier_saliency_in_full_frame(model_dir: Path, wl_cfg: dict, monkeypatch):
    clf = load_lesion_classifier(model_dir, wl_cfg)
    img = np.full((100, 120, 3), 0.5, dtype=np.float32)
    img[20:50, 30:60] = (1.0, 0.0, 0.0)
    mask = np.zeros((100, 120), dtype=bool)
    mask[10:60, 20:70] = True
    sal = clf.saliency(img, mask, target_index=RED_PATCH)
    assert sal is not None and sal.shape == (100, 120)
    assert sal[70:, 80:].max() == 0.0  # outside the crop
    assert sal[20:50, 30:60].mean() > 0.3
    auto = clf.saliency(img)
    assert auto is not None and auto.shape == (100, 120)

    def broken(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(clf, "predict_proba", broken)
    assert clf.saliency(img) is None
