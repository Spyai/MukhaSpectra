"""Tests for mukha_pipeline.capture (CaptureSource, UploadAdapter, ArraySource)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from wlspec_helpers import image_bytes, jpeg_with_exif, make_reading, spectral_csv

from mukha_pipeline.capture import (
    LED_FOR_KIND,
    ArraySource,
    CaptureSource,
    SyntheticSource,
    UploadAdapter,
    UploadDecodeError,
)
from mukha_pipeline.io.images import ImageDecodeError
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import (
    CaptureKind,
    Diascopy,
    LedMode,
    SiteRole,
    SourceAdapter,
    VisualAppearance,
    VisualFlag,
)
from mukha_pipeline.sites import OralSite
from mukha_pipeline.types import SiteInput


def _png(color: tuple[int, int, int], size: tuple[int, int] = (120, 80)) -> bytes:
    return image_bytes(Image.new("RGB", size, color), "PNG")


def test_adapters_satisfy_capture_source_protocol():
    assert isinstance(UploadAdapter(), CaptureSource)
    assert isinstance(ArraySource(SourceAdapter.MOCK_DEVICE), CaptureSource)
    assert UploadAdapter.adapter == SourceAdapter.UPLOAD
    assert LED_FOR_KIND[CaptureKind.AUTOFLUORESCENCE] == LedMode.VIOLET


def test_site_input_from_upload_builds_full_metadata():
    flag = VisualFlag(lesion_seen=True, appearance=[VisualAppearance.WHITE_PATCH])
    site = UploadAdapter().site_input(
        "buccal_mucosa_left",
        jpeg_with_exif(size=(160, 120)),
        _png((10, 160, 20), (160, 120)),
        _png((2, 2, 2), (160, 120)),
        filenames={"wl": "wl_bl.jpg", "af": "af_bl.png", "dark": "dark_bl.png"},
        role="examined",
        visual_flag=flag,
        diascopy="fvl_persists",
        px_per_mm=12.5,
        device_id="scout-01",
        firmware_version="0.1.0",
        synthetic=True,
    )
    assert isinstance(site, SiteInput)
    assert site.site == OralSite.BUCCAL_MUCOSA_LEFT
    assert site.role == SiteRole.EXAMINED and site.diascopy == Diascopy.FVL_PERSISTS
    assert site.visual_flag == flag and site.px_per_mm == 12.5

    wl, af, dark = site.wl.meta, site.af.meta, site.dark.meta  # type: ignore[union-attr]
    assert (wl.kind, wl.led_mode) == (CaptureKind.WHITE_LIGHT, LedMode.WHITE)
    assert (af.kind, af.led_mode) == (CaptureKind.AUTOFLUORESCENCE, LedMode.VIOLET)
    assert (dark.kind, dark.led_mode) == (CaptureKind.DARK_FRAME, LedMode.OFF)
    for meta in (wl, af, dark):
        assert meta.source_adapter == SourceAdapter.UPLOAD
        assert meta.oral_site == OralSite.BUCCAL_MUCOSA_LEFT
        assert meta.device_id == "scout-01" and meta.firmware_version == "0.1.0"
        assert meta.synthetic is True
        assert (meta.width_px, meta.height_px) == (160, 120)
        assert meta.file_sha256 and len(meta.file_sha256) == 64
    assert len({wl.capture_id, af.capture_id, dark.capture_id}) == 3
    assert wl.file_name == "wl_bl.jpg" and wl.mime_type == "image/jpeg"
    assert wl.exposure_time_ms == pytest.approx(8.0) and wl.iso == 200
    assert wl.white_balance == "manual" and wl.captured_at is not None
    assert wl.extra["camera_make"] == "MukhaTest" and wl.extra["bit_depth"] == 8
    assert af.mime_type == "image/png" and af.exposure_time_ms is None
    assert "camera_make" not in af.extra  # None values are dropped
    assert site.af.image.dtype == np.float32 and site.af.image.shape == (120, 160, 3)
    assert site.af.image[..., 1].mean() > site.af.image[..., 0].mean()


def test_site_input_without_dark_frame_and_defaults():
    site = UploadAdapter().site_input(OralSite.TONGUE_DORSUM, _png((200, 90, 90)), _png((5, 90, 5)))
    assert site.dark is None
    assert site.role == SiteRole.EXAMINED and site.visual_flag == VisualFlag()
    assert site.wl.meta.file_name is None and site.wl.meta.synthetic is False


def test_bad_upload_names_the_capture():
    adapter = UploadAdapter()
    with pytest.raises(UploadDecodeError) as info:
        adapter.site_input(
            "buccal_mucosa_left",
            _png((200, 90, 90)),
            b"not an image",
            filenames={"af": "af.png"},
        )
    err = info.value
    assert isinstance(err, ImageDecodeError)
    assert err.kind == CaptureKind.AUTOFLUORESCENCE and err.file_name == "af.png"
    assert str(err).startswith("Autofluorescence image (af.png): ")
    assert not find_positioning_violations(str(err))

    with pytest.raises(UploadDecodeError, match=r"^Dark frame: The image is too small"):
        adapter.site_input(
            "buccal_mucosa_left", _png((1, 1, 1)), _png((1, 1, 1)), _png((0, 0, 0), (32, 32))
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale_is_rejected(bad: float):
    with pytest.raises(ValueError, match="px_per_mm"):
        UploadAdapter().site_input("hard_palate", _png((1, 1, 1)), _png((1, 1, 1)), px_per_mm=bad)


def test_unknown_site_is_rejected():
    with pytest.raises(ValueError):
        UploadAdapter().site_input("cheek", _png((1, 1, 1)), _png((1, 1, 1)))


def test_spectral_upload_delegates_to_parser(rng, fixed_now):
    reading = make_reading(rng, "s1", OralSite.BUCCAL_MUCOSA_LEFT, "lesion", 2.0, n=3)
    parsed = UploadAdapter().spectral(
        spectral_csv([reading]), "probe.csv", device_id="x", synthetic=True, now=fixed_now
    )
    assert parsed.readings[0].meta.source_adapter == SourceAdapter.UPLOAD
    assert parsed.readings[0].meta.synthetic is True


def test_array_source_marks_synthetic_and_normalises_images():
    source = SyntheticSource()
    assert source.adapter == SourceAdapter.SYNTHETIC and source.synthetic
    wl = np.full((40, 50, 3), 200, dtype=np.uint8)
    af = np.random.default_rng(0).random((40, 50, 3))
    dark = np.zeros((40, 50))
    site = source.site_input(
        "lateral_tongue_left",
        wl,
        af,
        dark,
        role=SiteRole.REFERENCE_ONLY,
        exposure_time_ms={"af": 100.0},
        device_id="mock-scout",
    )
    assert site.wl.image.dtype == np.float32 and site.wl.image.max() == pytest.approx(200 / 255)
    assert site.dark is not None and site.dark.image.shape == (40, 50, 3)
    assert site.af.meta.exposure_time_ms == 100.0 and site.wl.meta.exposure_time_ms is None
    for cap in (site.wl, site.af, site.dark):
        assert cap.meta.synthetic is True
        assert cap.meta.source_adapter == SourceAdapter.SYNTHETIC
        assert len(str(cap.meta.extra["array_sha256"])) == 64
    assert site.role == SiteRole.REFERENCE_ONLY

    mock = ArraySource(SourceAdapter.MOCK_DEVICE)
    cap = mock.image_capture(wl, CaptureKind.WHITE_LIGHT, OralSite.HARD_PALATE)
    assert cap.meta.synthetic is False and cap.meta.led_mode == LedMode.WHITE
    no_dark = mock.site_input("hard_palate", wl, af)
    assert no_dark.dark is None
