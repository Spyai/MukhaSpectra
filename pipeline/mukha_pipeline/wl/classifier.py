"""White-light lesion classifier: interface, ONNX Runtime runner and the "no model" fallback.

No trained weights ship with Phase 1, so the default is :class:`UnavailableLesionClassifier` and
triage ignores white-light model output. When ``ml/`` exports a model, dropping ``model.onnx`` and
its ``model_card.md`` into the model directory switches to :class:`OnnxLesionClassifier` with no
code change. Whatever the model card says, ``validated`` stays False: only a documented clinical
validation may change that, and never from code.

The runner never raises during classification. Load or inference failures become a
``WLAnalysis`` with status ``error`` and a plain note, so one bad model cannot stop a screening.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from mukha_pipeline.io.images import to_float_rgb
from mukha_pipeline.schema.models import (
    ModelComponent,
    ModelInfo,
    ModelStatus,
    WLAnalysis,
    WLLabel,
    WLPrediction,
)
from mukha_pipeline.wl.saliency import occlusion_saliency, wl_section

MODEL_NAME = "wl-lesion-classifier"
MODEL_FILE = "model.onnx"
MODEL_CARD_FILE = "model_card.md"
UNAVAILABLE_NOTE = (
    "No trained weights installed. White-light lesion classification is not used in triage."
)
NOT_VALIDATED_NOTE = "Not clinically validated."
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_LABELS = tuple(label.value for label in WLLabel)
DEFAULT_CROP_MARGIN = 0.10
MIN_MASK_PIXELS = 64  # fewer mucosa pixels than this cannot define a meaningful crop

CropBox = tuple[int, int, int, int]  # y0, y1, x0, x1


@runtime_checkable
class LesionClassifier(Protocol):
    info: ModelInfo

    def classify(self, wl: np.ndarray, mucosa: np.ndarray | None = None) -> WLAnalysis: ...


class UnavailableLesionClassifier:
    """Used when no weights are installed: the pipeline runs, triage ignores the WL model."""

    def __init__(self, reason: str | None = None) -> None:
        notes = f"{UNAVAILABLE_NOTE} {reason}" if reason else UNAVAILABLE_NOTE
        self.info = ModelInfo(
            component=ModelComponent.WL_LESION_CLASSIFIER,
            name=MODEL_NAME,
            version="none",
            status=ModelStatus.MODEL_UNAVAILABLE,
            validated=False,
            notes=notes,
        )

    def classify(self, wl: np.ndarray, mucosa: np.ndarray | None = None) -> WLAnalysis:
        return WLAnalysis(status="model_unavailable", model=self.info, note=self.info.notes)


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def card_field(card_text: str | None, key: str) -> str | None:
    """Value of ``key:`` in the model card's YAML front matter (or anywhere if there is none)."""
    if not card_text:
        return None
    lines = card_text.splitlines()
    block = lines
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
        block = lines[1:end] if end is not None else lines[1:]
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", re.IGNORECASE)
    for line in block:
        m = pattern.match(line)
        if m:
            value = m.group(1).strip().strip("'\"").strip()
            return value or None
    return None


def _read_card(model_card: str | Path | None) -> tuple[str | None, str | None]:
    """(card text, card reference for ModelInfo). A Path (or a path-like .md string) is read."""
    if model_card is None:
        return None, None
    path: Path | None = None
    if isinstance(model_card, Path):
        path = model_card
    elif "\n" not in model_card and model_card.strip().lower().endswith(".md"):
        candidate = Path(model_card.strip())
        path = candidate if candidate.is_file() else None
    if path is None:
        return str(model_card), None
    ref = f"{path.parent.name}/{path.name}" if path.parent.name else path.name
    try:
        return path.read_text(encoding="utf-8", errors="replace"), ref
    except OSError:
        return None, ref


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _labels(raw: Any) -> list[WLLabel]:
    values = list(raw) if raw else list(DEFAULT_LABELS)
    try:
        return [WLLabel(str(v)) for v in values]
    except ValueError as err:
        raise ValueError(
            f"Unknown WL label in configuration ({err}); allowed: {', '.join(DEFAULT_LABELS)}."
        ) from None


def to_probabilities(raw: np.ndarray) -> np.ndarray:
    """Softmax, unless the outputs already look like probabilities (in [0, 1], sum ~ 1)."""
    x = np.asarray(raw, dtype=np.float64).reshape(-1)
    if np.all(x >= 0.0) and np.all(x <= 1.0) and abs(float(x.sum()) - 1.0) <= 1e-3:
        return x / float(x.sum())
    z = np.exp(x - float(x.max()))
    return z / float(z.sum())


def mucosa_crop_box(
    shape: tuple[int, ...], mucosa: np.ndarray | None, margin: float = DEFAULT_CROP_MARGIN
) -> tuple[CropBox, str | None]:
    """Bounding box of the mucosa mask plus ``margin`` of its size on each side.

    Cropping keeps the lesion large in the model's small input. Falls back to the whole image
    (with a note) when the mask is missing, mis-sized or too small to trust.
    """
    H, W = int(shape[0]), int(shape[1])
    full: CropBox = (0, H, 0, W)
    if mucosa is None:
        return full, None
    mask = np.asarray(mucosa)
    if mask.shape[:2] != (H, W):
        return full, "The mucosa mask did not match the image size; the whole image was used."
    mask = mask.astype(bool)
    if int(mask.sum()) < MIN_MASK_PIXELS:
        return full, "Too little mucosa was segmented to crop; the whole image was used."
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    my, mx = math.ceil(margin * (y1 - y0)), math.ceil(margin * (x1 - x0))
    return (max(0, y0 - my), min(H, y1 + my), max(0, x0 - mx), min(W, x1 + mx)), None


# --------------------------------------------------------------------------------------------
# ONNX runner
# --------------------------------------------------------------------------------------------


class OnnxLesionClassifier:
    """Runs an exported image classifier with ONNX Runtime on CPU.

    Input: the mucosa crop resized to ``input_size`` (or the model's own fixed size), ImageNet
    normalisation from config, NCHW float32. Output: one score per configured label.
    """

    def __init__(
        self, model_path: Path | str, cfg: Mapping[str, Any], model_card: str | Path | None = None
    ) -> None:
        section = wl_section(cfg)
        self.model_path = Path(model_path)
        self.cfg: Mapping[str, Any] = section
        self.input_size = int(section.get("input_size", 224))
        self.mean = np.asarray(section.get("mean", IMAGENET_MEAN), np.float32).reshape(3, 1, 1)
        self.std = np.asarray(section.get("std", IMAGENET_STD), np.float32).reshape(3, 1, 1)
        self.crop_margin = float(section.get("crop_margin_fraction", DEFAULT_CROP_MARGIN))
        self.load_error: str | None = None
        self.labels: list[WLLabel] = []
        self._session: Any = None
        self._input_name = ""

        card_text, card_ref = _read_card(model_card)
        sha = None
        try:
            self.labels = _labels(section.get("labels"))
            sha = _sha256_file(self.model_path)
            self._open_session()
        except Exception as err:
            self.load_error = _short_error(err)
            self._session = None

        notes = NOT_VALIDATED_NOTE
        if self.load_error:
            notes = (
                f"The model file could not be loaded ({self.load_error}); white-light lesion "
                f"classification is not used in triage. {NOT_VALIDATED_NOTE}"
            )
        self.info = ModelInfo(
            component=ModelComponent.WL_LESION_CLASSIFIER,
            name=card_field(card_text, "name") or MODEL_NAME,
            version=card_field(card_text, "version") or "unknown",
            status=ModelStatus.MODEL_UNAVAILABLE if self.load_error else ModelStatus.AVAILABLE,
            validated=False,  # never taken from the model card
            model_card=card_ref,
            sha256=sha,
            notes=notes,
        )

    def _open_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as err:  # pragma: no cover - onnxruntime is installed in dev
            raise RuntimeError("ONNX Runtime is not installed (optional 'onnx' extra)") from err
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        shape = list(model_input.shape)
        if len(shape) != 4:
            raise ValueError(f"model input must be NCHW, got shape {shape}")
        if isinstance(shape[1], int) and shape[1] != 3:
            raise ValueError(f"model expects {shape[1]} input channels, not 3 (RGB)")
        if isinstance(shape[2], int) and isinstance(shape[3], int) and shape[2] == shape[3]:
            self.input_size = int(shape[2])  # the model's fixed size wins over config

    # ----------------------------------------------------------------------------------------

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """RGB [0, 1] image -> (1, 3, S, S) float32, normalised."""
        arr = to_float_rgb(img)
        s = self.input_size
        h, w = arr.shape[:2]
        interp = cv2.INTER_AREA if (h > s or w > s) else cv2.INTER_LINEAR
        resized = cv2.resize(arr, (s, s), interpolation=interp)
        chw = (np.transpose(resized, (2, 0, 1)) - self.mean) / self.std
        return np.ascontiguousarray(chw[None], dtype=np.float32)

    def predict_proba(self, img: np.ndarray) -> np.ndarray:
        """Probabilities per configured label for the whole of ``img`` (no cropping).

        Raises on failure; :meth:`classify` turns failures into an ``error`` status.
        """
        if self._session is None:
            raise RuntimeError(self.load_error or "model is not loaded")
        outputs = self._session.run(None, {self._input_name: self.preprocess(img)})
        raw = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if raw.size != len(self.labels):
            raise ValueError(
                f"the model returned {raw.size} scores but the configuration lists "
                f"{len(self.labels)} labels"
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("the model returned non-finite scores")
        return to_probabilities(raw)

    def crop(
        self, wl: np.ndarray, mucosa: np.ndarray | None = None
    ) -> tuple[np.ndarray, CropBox, str | None]:
        box, note = mucosa_crop_box(np.shape(wl), mucosa, self.crop_margin)
        y0, y1, x0, x1 = box
        return to_float_rgb(wl)[y0:y1, x0:x1], box, note

    def classify(self, wl: np.ndarray, mucosa: np.ndarray | None = None) -> WLAnalysis:
        if self._session is None:
            return WLAnalysis(status="error", model=self.info, note=self.info.notes)
        try:
            crop, _, crop_note = self.crop(wl, mucosa)
            probs = self.predict_proba(crop)
        except Exception as err:
            return WLAnalysis(
                status="error",
                model=self.info,
                note=f"The white-light model could not run on this image ({_short_error(err)}); "
                "its output was not used.",
            )
        predictions = sorted(
            (
                WLPrediction(label=label, probability=float(np.clip(p, 0.0, 1.0)))
                for label, p in zip(self.labels, probs, strict=True)
            ),
            key=lambda pred: pred.probability,
            reverse=True,
        )
        note = "Model output for clinician review. " + NOT_VALIDATED_NOTE
        if crop_note:
            note = f"{crop_note} {note}"
        return WLAnalysis(
            status="ok",
            model=self.info,
            predictions=predictions,
            top_label=predictions[0].label,
            note=note,
        )

    def saliency(
        self, wl: np.ndarray, mucosa: np.ndarray | None = None, target_index: int | None = None
    ) -> np.ndarray | None:
        """Occlusion saliency in the full WL frame (zero outside the crop); None on failure."""
        if self._session is None:
            return None
        try:
            crop, (y0, y1, x0, x1), _ = self.crop(wl, mucosa)
            if target_index is None:
                target_index = int(np.argmax(self.predict_proba(crop)))
            sal = occlusion_saliency(self.predict_proba, crop, target_index, self.cfg)
        except Exception:
            return None
        H, W = np.shape(wl)[:2]
        full = np.zeros((H, W), dtype=np.float32)
        full[y0:y1, x0:x1] = sal
        return full


def _short_error(err: BaseException) -> str:
    text = " ".join(str(err).split()) or type(err).__name__
    return text if len(text) <= 160 else text[:157] + "..."


def load_lesion_classifier(
    model_dir: Path | str | None, cfg: Mapping[str, Any]
) -> OnnxLesionClassifier | UnavailableLesionClassifier:
    """``model_dir/model.onnx`` + ``model_dir/model_card.md`` -> ONNX runner; else unavailable.

    A model without its card is not used: the card states intended use and limitations, and
    governance requires both to travel together.
    """
    if model_dir is None:
        return UnavailableLesionClassifier()
    directory = Path(model_dir)
    model = directory / MODEL_FILE
    card = directory / MODEL_CARD_FILE
    if not model.is_file():
        return UnavailableLesionClassifier()
    if not card.is_file():
        return UnavailableLesionClassifier(
            reason=f"A model file was found without its {MODEL_CARD_FILE}, so it is not used."
        )
    return OnnxLesionClassifier(model, cfg, model_card=card)
