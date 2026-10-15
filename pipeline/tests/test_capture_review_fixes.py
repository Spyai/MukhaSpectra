"""Regression tests for review findings in mukha_pipeline.capture.upload.

* security.1 / correctness.2: uploads are decoded at most 2048 px on the longest side (2x the
  1024 px working resolution); the original size and the decode scale are recorded, and
  ``px_per_mm`` is converted to the decoded pixels so millimetre measurements stay right.
* science.3: DNG autofluorescence and dark frames are decoded without white balance, white-light
  DNGs with the camera's.
* honesty.1: a SYNTHETIC file name or SYNTHETIC metadata marks the capture synthetic even when the
  caller did not say so.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image, PngImagePlugin
from wlspec_helpers import image_bytes, jpeg_with_exif

from mukha_pipeline.capture import UploadAdapter
from mukha_pipeline.capture.upload import UPLOAD_DECODE_MAX_SIDE, raw_white_balance_for
from mukha_pipeline.schema.models import CaptureKind


def _png(color: tuple[int, int, int], size: tuple[int, int] = (120, 80)) -> bytes:
    return image_bytes(Image.new("RGB", size, color), "PNG")


def test_uploads_are_decoded_at_twice_the_working_resolution():
    assert UPLOAD_DECODE_MAX_SIDE == 2048
    wl = image_bytes(Image.new("RGB", (3000, 2000), (200, 90, 90)), "JPEG")
    af = image_bytes(Image.new("RGB", (3000, 2000), (20, 160, 20)), "JPEG")
    site = UploadAdapter().site_input("buccal_mucosa_left", wl, af, px_per_mm=30.0)
    assert site.wl.image.shape == (1365, 2048, 3)
    assert site.af.image.shape == (1365, 2048, 3)
    meta = site.wl.meta
    # The metadata describes the capture as taken; the decode is recorded next to it.
    assert (meta.width_px, meta.height_px) == (3000, 2000)
    assert meta.extra["decoded_width_px"] == 2048 and meta.extra["decoded_height_px"] == 1365
    assert meta.extra["decode_scale"] == pytest.approx(2048 / 3000, abs=1e-6)
    # px_per_mm refers to the pixels the pipeline receives.
    assert site.px_per_mm == pytest.approx(30.0 * 2048 / 3000, rel=1e-6)


def test_max_side_can_be_changed_or_disabled():
    data = image_bytes(Image.new("RGB", (900, 600), (200, 90, 90)), "PNG")
    small = UploadAdapter(max_side=300).image_capture(data, CaptureKind.WHITE_LIGHT, "hard_palate")
    assert small.image.shape == (200, 300, 3)
    full = UploadAdapter(max_side=None).image_capture(data, CaptureKind.WHITE_LIGHT, "hard_palate")
    assert full.image.shape == (600, 900, 3)
    assert "decode_scale" not in full.meta.extra  # nothing to record when not downscaled


def test_small_uploads_keep_px_per_mm_and_metadata():
    site = UploadAdapter().site_input("hard_palate", _png((1, 1, 1)), _png((1, 1, 1)), px_per_mm=9)
    assert site.px_per_mm == 9
    assert (site.wl.meta.width_px, site.wl.meta.height_px) == (120, 80)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (CaptureKind.WHITE_LIGHT, "camera"),
        (CaptureKind.AUTOFLUORESCENCE, "unity"),
        (CaptureKind.DARK_FRAME, "unity"),
    ],
)
def test_raw_white_balance_by_capture_kind(kind, expected):
    assert raw_white_balance_for(kind) == expected


def test_dng_af_is_decoded_without_white_balance(monkeypatch):
    calls: list[dict] = []

    class Raw:
        sizes = types.SimpleNamespace(width=96, height=72, flip=0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def postprocess(self, **kwargs):
            calls.append(kwargs)
            return np.full((72, 96, 3), 20000, dtype=np.uint16)

    monkeypatch.setitem(sys.modules, "rawpy", types.SimpleNamespace(imread=lambda f: Raw()))
    site = UploadAdapter().site_input(
        "buccal_mucosa_left",
        b"II*\x00fake-dng",
        b"II*\x00fake-dng",
        b"II*\x00fake-dng",
        filenames={"wl": "wl.dng", "af": "af.dng", "dark": "dark.dng"},
    )
    wl_call, af_call, dark_call = calls
    assert wl_call["use_camera_wb"] is True
    assert af_call["use_camera_wb"] is False and list(af_call["user_wb"]) == [1, 1, 1, 1]
    assert dark_call["use_camera_wb"] is False
    assert "without white balance" in str(site.af.meta.extra["decode_notes"])


# --------------------------------------------------------------------------------------------
# SYNTHETIC detection (honesty.1)
# --------------------------------------------------------------------------------------------


def test_synthetic_file_name_marks_the_capture_synthetic():
    cap = UploadAdapter().image_capture(
        _png((9, 9, 9)),
        CaptureKind.WHITE_LIGHT,
        "hard_palate",
        file_name="hard_palate-wl.png",  # neutral name chosen by the server
        source_file_name="SYNTHETIC_hard_palate_wl.png",  # what the user uploaded
    )
    assert cap.meta.synthetic is True
    assert cap.meta.extra["synthetic_marker"] == "file_name"
    assert cap.meta.file_name == "hard_palate-wl.png"  # the uploaded name is never stored


def test_synthetic_exif_marks_the_capture_synthetic():
    cap = UploadAdapter().image_capture(
        jpeg_with_exif(make="MukhaSpectra SYNTHETIC"), CaptureKind.AUTOFLUORESCENCE, "hard_palate"
    )
    assert cap.meta.synthetic is True
    assert cap.meta.extra["synthetic_marker"] == "exif:Make"


def test_synthetic_png_text_marks_the_capture_synthetic():
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "mukha_pipeline.synthetic")
    data = image_bytes(Image.new("RGB", (80, 80)), "PNG", pnginfo=info)
    cap = UploadAdapter().image_capture(data, CaptureKind.DARK_FRAME, "hard_palate")
    assert cap.meta.synthetic is True and cap.meta.extra["synthetic_marker"] == "png:Software"


def test_real_captures_are_not_marked_synthetic():
    site = UploadAdapter().site_input(
        "hard_palate",
        jpeg_with_exif(),
        _png((5, 90, 5)),
        source_filenames={"wl": "IMG_0001.jpg", "af": "asha_af.png"},
    )
    assert site.wl.meta.synthetic is False and site.af.meta.synthetic is False
    assert "synthetic_marker" not in site.wl.meta.extra


def test_one_synthetic_capture_is_reported_per_capture():
    site = UploadAdapter().site_input(
        "hard_palate",
        _png((200, 90, 90)),
        _png((5, 90, 5)),
        source_filenames={"wl": "SYNTHETIC_wl.png", "af": "IMG_0002.png"},
    )
    assert site.wl.meta.synthetic is True and site.af.meta.synthetic is False


def test_exif_orientation_is_recorded_for_the_stored_copy():
    cap = UploadAdapter().image_capture(
        jpeg_with_exif(size=(120, 80), orientation=6), CaptureKind.WHITE_LIGHT, "hard_palate"
    )
    assert cap.meta.extra["exif_orientation"] == 6
    assert cap.image.shape == (120, 80, 3)
