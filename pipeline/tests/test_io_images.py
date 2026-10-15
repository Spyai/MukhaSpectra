"""Tests for mukha_pipeline.io.images (decoding, EXIF, encoding, resizing)."""

from __future__ import annotations

import hashlib
import io
import sys
import types
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps
from wlspec_helpers import dng_tagged_tiff, image_bytes, jpeg_with_exif, png_bytes

from mukha_pipeline.io import images as images_mod
from mukha_pipeline.io.images import (
    ExifInfo,
    ImageDecodeError,
    apply_exif_orientation,
    encode_png,
    exif_from_pil,
    load_image_bytes,
    resize_max_side,
    to_float_rgb,
    to_uint8,
)
from mukha_pipeline.positioning import find_positioning_violations

# --------------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------------


def test_jpeg_exif_is_parsed():
    data = jpeg_with_exif()
    out = load_image_bytes(data, "wl.jpg")
    assert out.mime_type == "image/jpeg"
    assert out.sha256 == hashlib.sha256(data).hexdigest()
    assert out.image.dtype == np.float32 and out.image.shape == (80, 120, 3)
    assert (out.width_px, out.height_px) == (120, 80)
    assert out.image.min() >= 0.0 and out.image.max() <= 1.0
    exif = out.exif
    assert exif.exposure_time_ms == pytest.approx(8.0)
    assert exif.iso == 200
    assert exif.white_balance == "manual"
    assert exif.make == "MukhaTest" and exif.model == "Scout-1"
    assert exif.captured_at == datetime(
        2026, 10, 14, 9, 15, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )


def test_exif_auto_white_balance_and_naive_time():
    out = load_image_bytes(jpeg_with_exif(white_balance=0, offset=None), "x.jpg")
    assert out.exif.white_balance == "auto"
    assert out.exif.captured_at == datetime(2026, 10, 14, 9, 15)
    assert out.exif.captured_at.tzinfo is None


def test_missing_exif_gives_none_fields():
    data = image_bytes(Image.new("RGB", (100, 90), (10, 20, 30)), "JPEG")
    assert load_image_bytes(data).exif == ExifInfo()


def test_odd_exif_values_become_none():
    out = load_image_bytes(
        jpeg_with_exif(
            exposure=(1, 0),
            iso=None,
            white_balance=7,
            datetime_original="0000:00:00 00:00:00",
            offset="bogus",
            make=None,
            model=None,
        )
    )
    assert out.exif.exposure_time_ms is None
    assert out.exif.iso is None
    assert out.exif.white_balance is None
    assert out.exif.captured_at is None
    assert out.exif.make is None


def test_exif_orientation_is_applied():
    out = load_image_bytes(jpeg_with_exif(size=(120, 80), orientation=6), "rotated.jpg")
    assert out.image.shape == (120, 80, 3)
    assert (out.width_px, out.height_px) == (80, 120)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_numpy_orientation_matches_pillow(orientation: int):
    rng = np.random.default_rng(orientation)
    arr = rng.integers(0, 255, size=(5, 7, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[274] = orientation
    reopened = Image.open(io.BytesIO(png_bytes(img, exif=exif)))
    expected = np.asarray(ImageOps.exif_transpose(reopened))
    np.testing.assert_array_equal(apply_exif_orientation(arr, orientation), expected)


def test_png_16bit_rgb_keeps_precision():
    arr = np.zeros((70, 80, 3), dtype=np.uint16)
    arr[..., 0] = 1000  # would collapse to 4/255 at 8 bits
    arr[..., 1] = 30000
    arr[..., 2] = 65535
    ok, buf = cv2.imencode(".png", arr[..., ::-1])  # OpenCV writes BGR
    assert ok
    out = load_image_bytes(buf.tobytes(), "deep.png")
    assert out.bit_depth == 16
    np.testing.assert_allclose(out.image[0, 0], [1000 / 65535, 30000 / 65535, 1.0], atol=1e-6)


def test_png_16bit_rgba_is_composited():
    arr = np.full((70, 80, 4), 65535, dtype=np.uint16)
    arr[..., 3] = 0  # fully transparent
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    out = load_image_bytes(buf.tobytes())
    assert out.image.max() == 0.0


def test_png_16bit_gray():
    arr = np.full((70, 80), 1234, dtype=np.uint16)
    out = load_image_bytes(png_bytes(Image.fromarray(arr)))
    assert out.bit_depth == 16
    np.testing.assert_allclose(out.image, 1234 / 65535, atol=1e-6)
    assert out.image.shape == (70, 80, 3)


def test_rgba_composited_on_black_and_palette_and_bilevel():
    rgba = Image.new("RGBA", (80, 80), (255, 255, 255, 0))
    assert load_image_bytes(png_bytes(rgba)).image.max() == 0.0

    pal = Image.new("P", (80, 80), 1)
    pal.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    out = load_image_bytes(png_bytes(pal))
    np.testing.assert_allclose(out.image[0, 0], [1.0, 0.0, 0.0])

    pal_t = Image.new("P", (80, 80), 1)
    pal_t.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    out_t = load_image_bytes(png_bytes(pal_t, transparency=1))
    assert out_t.image.max() == 0.0

    bilevel = Image.new("1", (80, 80), 1)
    np.testing.assert_allclose(load_image_bytes(png_bytes(bilevel)).image, 1.0)

    gray = Image.new("LA", (80, 80), (128, 255))
    np.testing.assert_allclose(load_image_bytes(png_bytes(gray)).image, 128 / 255, atol=1e-6)


def test_cmyk_jpeg_and_tiff_modes():
    cmyk = Image.new("CMYK", (80, 80), (0, 0, 0, 0))
    out = load_image_bytes(image_bytes(cmyk, "JPEG"), "c.jpg")
    assert out.image.shape == (80, 80, 3) and out.image.mean() > 0.9

    int_img = Image.fromarray(np.full((70, 70), 40000, dtype=np.int32), mode="I")
    out_i = load_image_bytes(image_bytes(int_img, "TIFF"), "i.tif")
    np.testing.assert_allclose(out_i.image, 40000 / 65535, atol=1e-6)
    assert out_i.mime_type == "image/tiff"

    big = Image.fromarray(np.full((70, 70), 200000, dtype=np.int32), mode="I")
    np.testing.assert_allclose(load_image_bytes(image_bytes(big, "TIFF")).image, 1.0)

    float_img = Image.fromarray(np.full((70, 70), 0.25, dtype=np.float32), mode="F")
    np.testing.assert_allclose(load_image_bytes(image_bytes(float_img, "TIFF")).image, 0.25)

    hdr = np.full((70, 70), 2.0, dtype=np.float32)
    hdr[0, 0] = 4.0
    out_f = load_image_bytes(image_bytes(Image.fromarray(hdr, mode="F"), "TIFF"))
    assert out_f.image.max() == pytest.approx(1.0) and out_f.image[1, 1, 0] == pytest.approx(0.5)


def test_heic_roundtrip():
    pytest.importorskip("pillow_heif")
    try:
        data = image_bytes(Image.new("RGB", (96, 80), (10, 200, 30)), "HEIF")
    except Exception:  # pragma: no cover - encoder not available on this platform
        pytest.skip("pillow-heif cannot encode HEIF here")
    out = load_image_bytes(data, "capture.heic")
    assert out.mime_type == "image/heif"
    assert out.image.shape == (80, 96, 3)
    expected = [10 / 255, 200 / 255, 30 / 255]
    np.testing.assert_allclose(out.image.mean(axis=(0, 1)), expected, atol=0.03)


def test_srgb_profile_is_left_alone_and_other_profiles_are_converted(monkeypatch):
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    img = Image.new("RGB", (80, 80), (120, 60, 30))
    out = load_image_bytes(png_bytes(img, icc_profile=srgb))
    assert out.notes == []

    monkeypatch.setattr(images_mod.ImageCms, "getProfileDescription", lambda p: "Display P3")
    out = load_image_bytes(png_bytes(img, icc_profile=srgb))
    assert any("Display P3" in n for n in out.notes)
    np.testing.assert_allclose(out.image[0, 0], [120 / 255, 60 / 255, 30 / 255], atol=0.02)


def test_broken_icc_profile_is_noted():
    out = load_image_bytes(png_bytes(Image.new("RGB", (80, 80)), icc_profile=b"garbage"))
    assert any("could not be applied" in n for n in out.notes)


# --------------------------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------------------------


def _decode_error(data: bytes, filename: str | None = None, **kw) -> str:
    with pytest.raises(ImageDecodeError) as info:
        load_image_bytes(data, filename, **kw)
    message = str(info.value)
    assert not find_positioning_violations(message)
    return message


def test_empty_garbage_and_truncated():
    assert "empty" in _decode_error(b"")
    assert "not a supported image" in _decode_error(b"this is not an image at all")
    data = jpeg_with_exif(size=(400, 300))
    assert "damaged or incomplete" in _decode_error(data[: len(data) // 2], "cut.jpg")


def test_too_small_and_too_large(monkeypatch):
    small = image_bytes(Image.new("RGB", (40, 200)), "PNG")
    assert "too small (40 x 200 px)" in _decode_error(small)

    big = image_bytes(Image.new("RGB", (200, 200)), "PNG")
    assert "too large (200 x 200 px" in _decode_error(big, max_pixels=10_000)

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
    assert "too large to open safely" in _decode_error(big)


def test_decompression_bomb_during_decode(monkeypatch):
    data = image_bytes(Image.new("RGB", (100, 100)), "PNG")

    def boom(*args, **kwargs):
        raise Image.DecompressionBombError("bomb")

    monkeypatch.setattr(images_mod, "_decode_pil", boom)
    assert "too large to open safely" in _decode_error(data)


def test_unexpected_decoder_failure_is_wrapped(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("codec crashed")

    monkeypatch.setattr(images_mod, "_decode_pil", fail)
    assert "damaged" in _decode_error(image_bytes(Image.new("RGB", (80, 80)), "PNG"))

    monkeypatch.setattr(images_mod.Image, "open", fail)
    assert "damaged" in _decode_error(b"\x89PNG....")


def test_high_bit_decode_failure(monkeypatch):
    arr = np.full((70, 80, 3), 1000, dtype=np.uint16)
    ok, buf = cv2.imencode(".png", arr)
    monkeypatch.setattr(images_mod.cv2, "imdecode", lambda *a, **k: None)
    assert "damaged" in _decode_error(buf.tobytes())


def test_dng_without_rawpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "rawpy", None)
    message = _decode_error(b"II*\x00fake", "photo.DNG")
    assert "DNG support needs the optional 'raw' extra" in message


def test_dng_tag_routes_to_raw_decoder_and_bad_dng_fails():
    message = _decode_error(dng_tagged_tiff(), "no-extension")
    assert "DNG file could not be decoded" in message
    assert "could not be decoded" in _decode_error(b"II*\x00garbage", "x.dng")


class _FakeRaw:
    def __init__(self, w: int, h: int) -> None:
        self.sizes = types.SimpleNamespace(width=w, height=h)
        self._w, self._h = w, h

    def __enter__(self) -> _FakeRaw:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def postprocess(self, **kwargs: object) -> np.ndarray:
        # Linear-safe LibRaw settings: no auto-brightening, sRGB curve, 16-bit output.
        assert kwargs["use_camera_wb"] and kwargs["output_bps"] == 16
        assert kwargs["no_auto_bright"] is True and kwargs["gamma"] == (2.4, 12.92)
        return np.full((self._h, self._w, 3), 32768, dtype=np.uint16)


def test_dng_success_path_with_fake_rawpy(monkeypatch):
    fake = types.SimpleNamespace(imread=lambda f: _FakeRaw(96, 72))
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    out = load_image_bytes(b"II*\x00fake-dng", "photo.dng")
    assert out.mime_type == "image/x-adobe-dng"
    assert out.bit_depth == 16 and out.image.shape == (72, 96, 3)
    np.testing.assert_allclose(out.image, 32768 / 65535, atol=1e-6)
    assert out.exif == ExifInfo()

    fake_big = types.SimpleNamespace(imread=lambda f: _FakeRaw(1000, 1000))
    monkeypatch.setitem(sys.modules, "rawpy", fake_big)
    assert "too large" in _decode_error(b"II*\x00fake", "big.dng", max_pixels=100_000)


def test_exif_from_pil_survives_broken_getexif():
    class Broken:
        def getexif(self):
            raise ValueError("bad exif")

    assert exif_from_pil(Broken()) == ExifInfo()  # type: ignore[arg-type]


def test_exif_helper_edge_cases():
    assert images_mod._positive_float((5, 2), rational=True) == 2.5
    assert images_mod._positive_float((1, 0), rational=True) is None
    assert images_mod._positive_float([400, 800]) == 400
    assert images_mod._positive_float(()) is None
    assert images_mod._positive_float("abc") is None
    assert images_mod._positive_float(-3) is None
    assert images_mod._white_balance("x") is None
    assert images_mod._clean_text(b"Canon\x00") == "Canon"
    assert images_mod._exif_offset("+0A:00") is None
    assert images_mod._exif_datetime("garbage", None) is None
    assert images_mod._orientation(types.SimpleNamespace(getexif=lambda: {274: 99})) == 1
    assert images_mod._orientation(types.SimpleNamespace()) == 1
    assert images_mod._mime_type(None, "x.webp") == "image/webp"
    assert images_mod._mime_type(None, None) == "application/octet-stream"


# --------------------------------------------------------------------------------------------
# Conversions, encoding, resizing
# --------------------------------------------------------------------------------------------


def test_to_uint8_handles_dtypes_and_nan():
    f = np.array([[0.0, 0.5, 1.0, np.nan, 2.0, -1.0]], dtype=np.float32)
    np.testing.assert_array_equal(to_uint8(f), [[0, 128, 255, 0, 255, 0]])
    np.testing.assert_array_equal(to_uint8(np.array([True, False])), [255, 0])
    np.testing.assert_array_equal(to_uint8(np.array([0, 65535], dtype=np.uint16)), [0, 255])
    np.testing.assert_array_equal(to_uint8(np.array([-5, 300])), [0, 255])
    u = np.array([1, 2], dtype=np.uint8)
    out = to_uint8(u)
    assert out is not u and out.tolist() == [1, 2]


def test_to_float_rgb_variants():
    assert to_float_rgb(np.zeros((4, 5), dtype=np.uint8)).shape == (4, 5, 3)
    assert to_float_rgb(np.ones((4, 5, 1), dtype=bool)).max() == 1.0
    np.testing.assert_allclose(to_float_rgb(np.full((2, 2, 4), 65535, np.uint16)), 1.0)
    with pytest.raises(ValueError):
        to_float_rgb(np.zeros((2, 2, 2)))


@pytest.mark.parametrize(
    ("img", "bit_depth", "mode"),
    [
        (np.random.default_rng(0).random((20, 30, 3)).astype(np.float32), 8, "RGB"),
        (np.random.default_rng(1).random((20, 30)).astype(np.float32), 8, "L"),
        (np.random.default_rng(2).random((20, 30, 1)).astype(np.float32), 16, "I;16"),
        (np.random.default_rng(3).random((20, 30, 4)).astype(np.float32), 8, "RGBA"),
        (np.zeros((20, 30), dtype=bool), 16, "I;16"),
        (np.zeros((20, 30, 3), dtype=np.uint8), 16, "RGB"),
    ],
)
def test_encode_png_roundtrip(img: np.ndarray, bit_depth: int, mode: str):
    data = encode_png(img, bit_depth=bit_depth)  # type: ignore[arg-type]
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    decoded = Image.open(io.BytesIO(data))
    assert decoded.mode == mode
    assert decoded.size == (30, 20)


def test_encode_png_16bit_gray_precision_and_rgb_order():
    gray = np.full((10, 12), 1000 / 65535, dtype=np.float32)
    decoded = np.asarray(Image.open(io.BytesIO(encode_png(gray, bit_depth=16))))
    assert int(decoded[0, 0]) == 1000
    rgb = np.zeros((10, 12, 3), dtype=np.float32)
    rgb[..., 0] = 1.0
    out = np.asarray(Image.open(io.BytesIO(encode_png(rgb))))
    assert out[0, 0].tolist() == [255, 0, 0]
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(encode_png(np.full((4, 4), 7, np.uint16), bit_depth=16)))),
        7,
    )
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(encode_png(np.full((4, 4), 70000), bit_depth=16)))),
        65535,
    )


def test_encode_png_rejects_bad_input():
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4, 2)))
    with pytest.raises(ValueError):
        encode_png(np.zeros((0, 4)))
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4)), bit_depth=12)  # type: ignore[arg-type]


def test_resize_max_side():
    img = np.random.default_rng(0).random((300, 400, 3)).astype(np.float32)
    small, scale = resize_max_side(img, 100)
    assert small.shape == (75, 100, 3) and scale == pytest.approx(0.25)
    assert small.dtype == np.float32

    same, scale1 = resize_max_side(img, 400)
    assert same is img and scale1 == 1.0
    never_up, scale2 = resize_max_side(img, 4000)
    assert never_up.shape == img.shape and scale2 == 1.0

    mask = np.zeros((300, 400), dtype=bool)
    mask[100:200, 100:300] = True
    m, _ = resize_max_side(mask, 40)
    assert m.dtype == bool and m.shape == (30, 40) and m.any()

    single = np.ones((300, 400, 1), dtype=np.float32)
    assert resize_max_side(single, 100)[0].shape == (75, 100, 1)

    with pytest.raises(ValueError):
        resize_max_side(img, 0)


def test_resize_preserves_mean_when_shrinking():
    img = np.zeros((400, 400), dtype=np.float32)
    img[::2, ::2] = 1.0  # fine checkerboard: INTER_AREA averages instead of aliasing
    small, _ = resize_max_side(img, 100)
    assert small.mean() == pytest.approx(0.25, abs=0.01)
    assert small.std() < 0.01
