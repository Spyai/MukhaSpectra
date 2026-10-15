"""Confirm-probe spectral analysis: reflectance summary, features, lesion-vs-reference rules."""

from mukha_pipeline.spectral.analyze import analyze_spectral, find_calibration
from mukha_pipeline.spectral.classifier import RuleBasedSpectralClassifier, SpectralClassifier
from mukha_pipeline.spectral.features import (
    compute_features,
    delta_features,
    hb_absorption_proxy,
    quotient_spectrum,
    reference_delta,
)
from mukha_pipeline.spectral.pairing import auto_pair, auto_pair_inputs
from mukha_pipeline.spectral.summary import calibration_info, summarise_reading

__all__ = [
    "RuleBasedSpectralClassifier",
    "SpectralClassifier",
    "analyze_spectral",
    "auto_pair",
    "auto_pair_inputs",
    "calibration_info",
    "compute_features",
    "delta_features",
    "find_calibration",
    "hb_absorption_proxy",
    "quotient_spectrum",
    "reference_delta",
    "summarise_reading",
]
