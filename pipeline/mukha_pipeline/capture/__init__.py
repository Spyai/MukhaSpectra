"""Capture sources: one interface, many adapters (upload, synthetic, devices in later phases)."""

from mukha_pipeline.capture.arrays import ArraySource, SyntheticSource
from mukha_pipeline.capture.base import LED_FOR_KIND, CaptureSource, image_metadata
from mukha_pipeline.capture.upload import UploadAdapter, UploadDecodeError

__all__ = [
    "LED_FOR_KIND",
    "ArraySource",
    "CaptureSource",
    "SyntheticSource",
    "UploadAdapter",
    "UploadDecodeError",
    "image_metadata",
]
