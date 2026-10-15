"""Regression tests for review findings in mukha_pipeline.io.

* security.1 / correctness.2: decode budget. ``max_side`` downscales DURING decode (JPEG draft,
  Pillow reduce, INTER_AREA on integer pixels), so a huge upload never becomes a full-resolution
  float32 array; the per-image limit is 50 MP.
* science.3: DNG decoding is not auto-brightened, uses the sRGB transfer curve, and can skip
  white balance (AF and dark frames keep the sensor's channel ratios).
* honesty.1: SYNTHETIC markers in EXIF / PNG text are reported by the decoder.
* security.4: identifying metadata is stripped from stored JPEG / PNG files without changing the
  decoded pixels.
"""

from __future__ import annotations

import io
import sys
import types
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms, PngImagePlugin, TiffImagePlugin
from wlspec_helpers import image_bytes, jpeg_with_exif

from mukha_pipeline.io import images as images_mod
from mukha_pipeline.io.images import (
    MAX_IMAGE_PIXELS,
    ImageDecodeError,
    load_image_bytes,
)
from mukha_pipeline.io.sanitize import strip_identifying_metadata


def _gradient(width: int, height: int) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width]
    arr = np.zeros((height, width, 3), np.uint8)
    arr[..., 0] = (xx * 255 // max(width - 1, 1)).astype(np.uint8)
    arr[..., 1] = (yy * 255 // max(height - 1, 1)).astype(np.uint8)
    arr[..., 2] = 90
    return Image.fromarray(arr, mode="RGB")


def _record_float_conversion(monkeypatch) -> list[tuple[int, ...]]:
    """Record the shape of every array handed to the float conversion."""
    seen: list[tuple[int, ...]] = []
    original = images_mod._pil_to_float_rgb

    def spy(pil):
        seen.append((pil.height, pil.width))
        return original(pil)

    monkeypatch.setattr(images_mod, "_pil_to_float_rgb", spy)
    return seen


# --------------------------------------------------------------------------------------------
# Decode budget (security.1, correctness.2)
# --------------------------------------------------------------------------------------------


def test_per_image_pixel_limit_is_50_mp():
    assert MAX_IMAGE_PIXELS == 50_000_000
    with pytest.raises(ImageDecodeError, match="too large"):
        images_mod._check_size(7100, 7100, MAX_IMAGE_PIXELS)  # 50.4 MP
    images_mod._check_size(8000, 6000, MAX_IMAGE_PIXELS)  # a 48 MP phone photo still fits


def test_jpeg_is_downscaled_during_decode_never_at_full_resolution(monkeypatch):
    data = image_bytes(_gradient(4000, 3000), "JPEG", quality=92)
    seen = _record_float_conversion(monkeypatch)
    out = load_image_bytes(data, "big.jpg", max_side=1000)
    assert out.image.shape == (750, 1000, 3) and out.image.dtype == np.float32
    assert (out.width_px, out.height_px) == (1000, 750)
    assert (out.original_width_px, out.original_height_px) == (4000, 3000)
    assert out.decode_scale == pytest.approx(0.25)
    # The JPEG decoder itself produced the reduced image (DCT scaling via draft()).
    assert seen and max(seen[0]) <= 2 * 1000
    assert any("Downscaled" in note for note in out.notes)
    # Same picture as a full decode followed by a resize.
    full = load_image_bytes(data, "big.jpg")
    reference = cv2.resize(full.image, (1000, 750), interpolation=cv2.INTER_AREA)
    assert np.abs(out.image - reference).mean() < 0.01


def test_png_is_reduced_before_float_conversion(monkeypatch):
    data = image_bytes(_gradient(3000, 2000), "PNG")
    seen = _record_float_conversion(monkeypatch)
    out = load_image_bytes(data, "big.png", max_side=500)
    assert out.image.shape == (333, 500, 3)
    assert (out.original_width_px, out.original_height_px) == (3000, 2000)
    assert seen and max(seen[0]) < 2 * 500


def test_downscaling_keeps_exif_orientation():
    data = jpeg_with_exif(size=(1600, 1200), orientation=6)
    out = load_image_bytes(data, "rotated.jpg", max_side=400)
    assert out.image.shape == (400, 300, 3)
    assert (out.original_width_px, out.original_height_px) == (1200, 1600)
    assert out.decode_scale == pytest.approx(0.25)


def test_16bit_png_is_downscaled_on_integers_and_keeps_precision():
    arr = np.zeros((800, 1000, 3), dtype=np.uint16)
    arr[..., 0] = 1000
    arr[..., 1] = 30000
    arr[..., 2] = 65535
    ok, buf = cv2.imencode(".png", arr[..., ::-1])
    assert ok
    out = load_image_bytes(buf.tobytes(), "deep.png", max_side=250)
    assert out.bit_depth == 16 and out.image.shape == (200, 250, 3)
    np.testing.assert_allclose(out.image[10, 10], [1000 / 65535, 30000 / 65535, 1.0], atol=1e-6)


@pytest.mark.parametrize("mode", ["P", "1", "I;16", "LA", "RGBA"])
def test_downscaling_handles_every_pillow_mode(mode: str):
    img = Image.new(mode, (900, 600))
    out = load_image_bytes(image_bytes(img, "PNG"), "m.png", max_side=300)
    assert out.image.shape == (200, 300, 3)


def test_small_images_are_not_touched_by_max_side():
    data = image_bytes(_gradient(320, 240), "PNG")
    plain = load_image_bytes(data)
    capped = load_image_bytes(data, max_side=2048)
    np.testing.assert_array_equal(plain.image, capped.image)
    assert capped.decode_scale == 1.0 and capped.original_width_px == 320


def test_max_side_below_the_minimum_side_is_rejected():
    with pytest.raises(ValueError):
        load_image_bytes(image_bytes(_gradient(320, 240), "PNG"), max_side=10)


# --------------------------------------------------------------------------------------------
# DNG (science.3)
# --------------------------------------------------------------------------------------------


class _RecordingRaw:
    calls: list[dict] = []

    def __init__(self, w: int, h: int) -> None:
        self.sizes = types.SimpleNamespace(width=w, height=h, flip=0)
        self._w, self._h = w, h

    def __enter__(self) -> _RecordingRaw:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def postprocess(self, **kwargs) -> np.ndarray:
        _RecordingRaw.calls.append(kwargs)
        div = 2 if kwargs.get("half_size") else 1
        return np.full((self._h // div, self._w // div, 3), 32768, dtype=np.uint16)


@pytest.fixture
def fake_rawpy(monkeypatch):
    _RecordingRaw.calls = []

    def install(w: int = 96, h: int = 72):
        monkeypatch.setitem(
            sys.modules, "rawpy", types.SimpleNamespace(imread=lambda f: _RecordingRaw(w, h))
        )
        return _RecordingRaw.calls

    return install


def test_dng_is_not_auto_brightened_and_uses_srgb_gamma(fake_rawpy):
    calls = fake_rawpy()
    out = load_image_bytes(b"II*\x00fake-dng", "wl.dng")
    (kwargs,) = calls
    assert kwargs["no_auto_bright"] is True
    assert kwargs["gamma"] == (2.4, 12.92)
    assert kwargs["output_bps"] == 16
    assert kwargs["use_camera_wb"] is True  # white light keeps the camera white balance
    np.testing.assert_allclose(out.image, 32768 / 65535, atol=1e-6)


def test_dng_white_balance_can_be_disabled_for_fluorescence(fake_rawpy):
    calls = fake_rawpy()
    out = load_image_bytes(b"II*\x00fake-dng", "af.dng", raw_white_balance="unity")
    (kwargs,) = calls
    assert kwargs["use_camera_wb"] is False and kwargs["use_auto_wb"] is False
    assert list(kwargs["user_wb"]) == [1.0, 1.0, 1.0, 1.0]
    assert kwargs["no_auto_bright"] is True and kwargs["gamma"] == (2.4, 12.92)
    assert any("without white balance" in note for note in out.notes)


def test_dng_honours_max_side_with_half_size_decoding(fake_rawpy):
    calls = fake_rawpy(4000, 3000)
    out = load_image_bytes(b"II*\x00fake-dng", "big.dng", max_side=1000)
    assert calls[0]["half_size"] is True
    assert out.image.shape == (750, 1000, 3)
    assert (out.original_width_px, out.original_height_px) == (4000, 3000)


def test_unknown_raw_white_balance_is_rejected():
    with pytest.raises(ValueError):
        load_image_bytes(b"II*\x00fake", "x.dng", raw_white_balance="daylight")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# SYNTHETIC markers (honesty.1)
# --------------------------------------------------------------------------------------------


def test_synthetic_exif_make_is_reported():
    out = load_image_bytes(jpeg_with_exif(make="MukhaSpectra SYNTHETIC"), "x.jpg")
    assert out.synthetic_markers == ["exif:Make"]


def test_synthetic_png_text_is_reported():
    info = PngImagePlugin.PngInfo()
    info.add_text("Description", "SYNTHETIC image generated by mukha_pipeline.synthetic")
    data = image_bytes(Image.new("RGB", (80, 80)), "PNG", pnginfo=info)
    assert load_image_bytes(data).synthetic_markers == ["png:Description"]


def test_real_capture_has_no_synthetic_marker():
    assert load_image_bytes(jpeg_with_exif(), "x.jpg").synthetic_markers == []


def test_shipped_synthetic_samples_are_recognised():
    from pathlib import Path

    folder = Path(__file__).resolve().parents[2] / "sample_data" / "synthetic" / "high"
    files = sorted(p for p in folder.glob("SYNTHETIC_*") if p.suffix in (".jpg", ".png"))
    if not files:
        pytest.skip("sample data not generated")
    for path in files[:2]:
        assert load_image_bytes(path.read_bytes(), path.name).synthetic_markers


# --------------------------------------------------------------------------------------------
# Metadata stripping (security.4)
# --------------------------------------------------------------------------------------------

GPS_IFD = 0x8825
ARTIST = 315


def _identifying_exif(orientation: int | None) -> Image.Exif:
    exif = Image.Exif()
    exif[271] = "PhoneMaker"
    exif[ARTIST] = "Asha Ramesh (patient)"
    exif[0xA431] = "SERIAL-123456"  # BodySerialNumber
    if orientation:
        exif[274] = orientation
    gps = exif.get_ifd(GPS_IFD)
    gps[1] = "N"
    gps[2] = (
        TiffImagePlugin.IFDRational(12, 1),
        TiffImagePlugin.IFDRational(58, 1),
        TiffImagePlugin.IFDRational(30, 1),
    )
    sub = exif.get_ifd(0x8769)
    sub[33434] = TiffImagePlugin.IFDRational(1, 125)
    return exif


def _xmp_segment(text: bytes) -> bytes:
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + text
    return b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def _with_extra_jpeg_segments(data: bytes) -> bytes:
    """Insert XMP, IPTC (APP13) and a comment after SOI, and append a trailer image."""
    extra = (
        _xmp_segment(b"<x:xmpmeta><dc:creator>Asha Ramesh</dc:creator></x:xmpmeta>")
        + _segment(0xED, b"Photoshop 3.0\x008BIM\x04\x04\x00\x00\x00\x00\x00\x05Asha!")
        + _segment(0xFE, b"patient Asha Ramesh, Hoskote")
    )
    trailer = image_bytes(Image.new("RGB", (64, 64), (1, 2, 3)), "JPEG", exif=_identifying_exif(1))
    return data[:2] + extra + data[2:] + trailer


def _photo(orientation: int | None = 6, *, icc: bool = True) -> bytes:
    img = _gradient(160, 120)
    kwargs = {"exif": _identifying_exif(orientation), "quality": 93}
    if icc:
        kwargs["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return _with_extra_jpeg_segments(image_bytes(img, "JPEG", **kwargs))


def _assert_no_identifying_metadata(data: bytes) -> None:
    lowered = data.lower()
    for needle in (b"asha", b"ramesh", b"hoskote", b"serial-123456", b"phonemaker", b"xmp"):
        assert needle not in lowered, needle
    with Image.open(io.BytesIO(data)) as img:
        exif = img.getexif()
        assert ARTIST not in exif and 271 not in exif
        assert not exif.get_ifd(GPS_IFD)


@pytest.mark.parametrize("orientation", [None, 1, 3, 6, 8])
def test_jpeg_metadata_is_stripped_and_pixels_are_unchanged(orientation):
    original = _photo(orientation)
    result = strip_identifying_metadata(original)
    assert result.sanitised
    assert {"EXIF", "XMP", "IPTC", "comment"} <= set(result.removed)
    _assert_no_identifying_metadata(result.data)
    before = load_image_bytes(original, "x.jpg")
    after = load_image_bytes(result.data, "x.jpg")
    np.testing.assert_array_equal(before.image, after.image)
    with Image.open(io.BytesIO(result.data)) as img:
        assert img.info.get("icc_profile")  # colour profile kept
        assert img.getexif().get(274, 1) == (orientation or 1)


def test_jpeg_trailing_images_are_dropped():
    trailer = image_bytes(Image.new("RGB", (64, 64), (1, 2, 3)), "JPEG")
    clean = image_bytes(_gradient(160, 120), "JPEG")
    result = strip_identifying_metadata(clean + trailer)
    assert result.sanitised and "trailing data" in result.removed
    assert result.data == clean


def test_progressive_jpeg_survives_stripping():
    original = image_bytes(
        _gradient(200, 150), "JPEG", progressive=True, exif=_identifying_exif(3), quality=90
    )
    result = strip_identifying_metadata(original)
    _assert_no_identifying_metadata(result.data)
    np.testing.assert_array_equal(
        load_image_bytes(original).image, load_image_bytes(result.data).image
    )


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    body = ctype + payload
    return len(payload).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")


def _png_with_metadata(orientation: int | None) -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "Asha Ramesh")
    info.add_itxt("Comment", "Hoskote camp, patient Asha")
    info.add_text("Location", "12.97N 77.59E Hoskote", zip=True)
    data = image_bytes(
        _gradient(150, 100), "PNG", pnginfo=info, exif=_identifying_exif(orientation)
    )
    # A text chunk after the image data (Pillow only reads these after decoding).
    iend = data.rindex(b"IEND") - 4
    return data[:iend] + _png_chunk(b"tEXt", b"Comment\x00Asha Ramesh again") + data[iend:]


@pytest.mark.parametrize("orientation", [None, 3, 6])
def test_png_metadata_is_stripped_and_pixels_are_unchanged(orientation):
    original = _png_with_metadata(orientation)
    result = strip_identifying_metadata(original)
    assert result.sanitised
    assert {"eXIf", "tEXt", "iTXt", "zTXt"} <= set(result.removed)
    _assert_no_identifying_metadata(result.data)
    np.testing.assert_array_equal(
        load_image_bytes(original).image, load_image_bytes(result.data).image
    )
    with Image.open(io.BytesIO(result.data)) as img:
        img.load()
        assert img.getexif().get(274, 1) == (orientation or 1)


def test_clean_files_are_returned_byte_for_byte():
    png = image_bytes(_gradient(80, 80), "PNG")
    assert strip_identifying_metadata(png).data == png
    jpeg = image_bytes(_gradient(80, 80), "JPEG")
    assert strip_identifying_metadata(jpeg).data == jpeg


def test_other_formats_are_stored_as_uploaded():
    tiff = image_bytes(_gradient(80, 80), "TIFF")
    result = strip_identifying_metadata(tiff)
    assert result.data == tiff and not result.sanitised and result.removed == ()


@pytest.mark.parametrize(
    "data",
    [b"\xff\xd8\xff\xe1\x00", b"\xff\xd8garbage", b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"],
)
def test_malformed_files_are_returned_unchanged(data: bytes):
    result = strip_identifying_metadata(data)
    assert result.data == data and not result.sanitised
