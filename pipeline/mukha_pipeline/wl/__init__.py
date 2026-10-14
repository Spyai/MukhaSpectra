"""White-light lesion classification: pluggable model interface, ONNX runner, saliency."""

from mukha_pipeline.wl.classifier import (
    LesionClassifier,
    OnnxLesionClassifier,
    UnavailableLesionClassifier,
    load_lesion_classifier,
    mucosa_crop_box,
    to_probabilities,
)
from mukha_pipeline.wl.saliency import occlusion_saliency

__all__ = [
    "LesionClassifier",
    "OnnxLesionClassifier",
    "UnavailableLesionClassifier",
    "load_lesion_classifier",
    "mucosa_crop_box",
    "occlusion_saliency",
    "to_probabilities",
]
