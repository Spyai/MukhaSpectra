"""Decoding of uploaded captures: images (JPEG/PNG/TIFF/HEIC/DNG) and spectral files (CSV/JSON).

This is the only place where raw bytes become pipeline types. Everything downstream works on
float32 RGB arrays and typed dataclasses, so file-format quirks never leak into analysis code.
"""

from mukha_pipeline.io.images import (
    MAX_IMAGE_PIXELS,
    MIN_SIDE_PX,
    ExifInfo,
    ImageDecodeError,
    LoadedImage,
    RawWhiteBalance,
    encode_png,
    load_image_bytes,
    resize_max_side,
    to_float_rgb,
    to_uint8,
)
from mukha_pipeline.io.sanitize import SanitizedImage, strip_identifying_metadata
from mukha_pipeline.io.spectral import (
    SPECTRAL_FORMAT_V1,
    ParsedSpectral,
    SpectralParseError,
    parse_spectral_file,
)

__all__ = [
    "MAX_IMAGE_PIXELS",
    "MIN_SIDE_PX",
    "SPECTRAL_FORMAT_V1",
    "ExifInfo",
    "ImageDecodeError",
    "LoadedImage",
    "ParsedSpectral",
    "RawWhiteBalance",
    "SanitizedImage",
    "SpectralParseError",
    "encode_png",
    "load_image_bytes",
    "parse_spectral_file",
    "resize_max_side",
    "strip_identifying_metadata",
    "to_float_rgb",
    "to_uint8",
]
