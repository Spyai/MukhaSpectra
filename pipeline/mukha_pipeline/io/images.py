"""Image decoding and encoding.

Every image entering the pipeline goes through :func:`load_image_bytes`, which returns float32 RGB
in [0, 1] (see ``mukha_pipeline.types``) plus the capture settings found in EXIF. Decoding is
defensive because uploads come from many phones and apps: EXIF orientation is applied, 16-bit
PNG/TIFF keep their full precision, embedded colour profiles (e.g. Display P3 on iPhones) are
converted to sRGB, and anything unreadable raises :class:`ImageDecodeError` with a message a
health worker can act on.

Memory is bounded: with ``max_side`` the image is shrunk DURING decoding (JPEG DCT scaling via
``draft()``, Pillow ``reduce()`` / OpenCV INTER_AREA on integer pixels), so a 48 MP upload never
becomes a full-resolution float32 array. Analysis runs at 1024 px, so uploads are decoded at
2048 px (see ``capture.upload``). The original size and the decode scale are recorded.

GPS and other identifying EXIF fields are deliberately never read (and ``io.sanitize`` removes
them from stored files). The only free-text fields inspected are those that can carry the
SYNTHETIC label written by ``mukha_pipeline.synthetic``.
"""

from __future__ import annotations

import hashlib
import io
import math
import mimetypes
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import PurePath
from typing import Any, Literal, get_args

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageCms, ImageOps, UnidentifiedImageError

try:  # HEIC/HEIF (iPhone default) support
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover - pillow-heif is a core dependency
    HEIF_AVAILABLE = False

# 50 MP: above the default output of phone cameras (12 MP binned, 48 MP full sensor) and small
# enough that one decode cannot exhaust a camp laptop's memory.
MAX_IMAGE_PIXELS = 50_000_000
MIN_SIDE_PX = 64  # smaller images cannot carry enough mucosa detail for any analysis step

# Pillow warns above MAX_IMAGE_PIXELS and raises DecompressionBombError above twice that. We also
# check the header size ourselves, so the limit holds even if another library changes this global.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

RAW_EXTENSIONS = frozenset({".dng"})
DNG_VERSION_TAG = 50706  # TIFF tag present in every DNG file
DNG_MIME = "image/x-adobe-dng"
_MIME_OVERRIDES = {"MPO": "image/jpeg", "HEIF": "image/heif"}

_EXIF_EXPOSURE_TIME = 33434
_EXIF_ISO = 34855  # ISOSpeedRatings, renamed PhotographicSensitivity in EXIF 2.3
_EXIF_ISO_SPEED = 34867
_EXIF_RECOMMENDED_EI = 34866
_EXIF_WHITE_BALANCE = 41987
_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME_DIGITIZED = 36868
_EXIF_OFFSET_TIME_ORIGINAL = 36881
_TIFF_DATETIME = 306
_TIFF_MAKE = 271
_TIFF_MODEL = 272
_TIFF_IMAGE_DESCRIPTION = 270
_TIFF_SOFTWARE = 305
_TIFF_ARTIST = 315
_EXIF_USER_COMMENT = 37510

# Free-text fields that the synthetic generator (or another tool) may use to label an image as
# SYNTHETIC. Only the field names are recorded, never their content.
_SYNTHETIC_EXIF_FIELDS: dict[int, str] = {
    _TIFF_MAKE: "Make",
    _TIFF_MODEL: "Model",
    _TIFF_IMAGE_DESCRIPTION: "ImageDescription",
    _TIFF_SOFTWARE: "Software",
    _TIFF_ARTIST: "Artist",
    _EXIF_USER_COMMENT: "UserComment",
}
SYNTHETIC_TOKEN = "synthetic"

RawWhiteBalance = Literal["camera", "unity"]
# LibRaw parameters: the sRGB transfer curve (power 2.4, toe slope 12.92), exactly what JPEG / PNG
# captures carry, so the pipeline's sRGB linearisation treats every format alike; and no per-image
# auto-brightening (it would rescale each capture by its own histogram and erase real exposure and
# fluorescence differences between a site and its contralateral reference).
RAW_GAMMA = (2.4, 12.92)


class ImageDecodeError(ValueError):
    """The bytes could not be turned into an image. ``str(err)`` is safe to show to users."""


@dataclass
class ExifInfo:
    exposure_time_ms: float | None = None
    iso: float | None = None
    white_balance: str | None = None  # "auto" | "manual"
    captured_at: datetime | None = None  # tz-aware only if the camera recorded an offset
    make: str | None = None
    model: str | None = None


@dataclass
class LoadedImage:
    image: np.ndarray  # float32 RGB [0, 1], (H, W, 3), EXIF orientation applied
    exif: ExifInfo
    mime_type: str
    sha256: str
    width_px: int
    height_px: int
    bit_depth: int = 8  # bits per channel of the decoded source
    notes: list[str] = field(default_factory=list)  # decoding decisions worth recording
    # Size of the capture as stored in the file (EXIF orientation applied), before any
    # decode-time downscaling; ``decode_scale`` = decoded width / original width.
    original_width_px: int | None = None
    original_height_px: int | None = None
    decode_scale: float = 1.0
    orientation: int = 1  # EXIF orientation that was applied (1 = none)
    # Metadata fields that label the image SYNTHETIC (e.g. "exif:Make", "png:Description").
    synthetic_markers: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------------


def load_image_bytes(
    data: bytes,
    filename: str | None = None,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_side: int | None = None,
    raw_white_balance: RawWhiteBalance = "camera",
) -> LoadedImage:
    """Decode JPEG, PNG, TIFF, HEIC/HEIF or DNG bytes into float32 RGB in [0, 1].

    ``max_side`` shrinks the image during decoding so its longest side is at most that many
    pixels (never upscales). ``raw_white_balance`` applies to DNG only: ``"camera"`` uses the
    camera's white balance (white-light photos), ``"unity"`` keeps the sensor's channel ratios
    (autofluorescence and dark frames, whose green/red ratio is the measurement).

    Raises :class:`ImageDecodeError` with a plain-language message on any failure.
    """
    if max_side is not None and max_side < MIN_SIDE_PX:
        raise ValueError(f"max_side must be at least {MIN_SIDE_PX} px")
    if raw_white_balance not in get_args(RawWhiteBalance):
        raise ValueError(f"raw_white_balance must be one of {get_args(RawWhiteBalance)}")
    if not data:
        raise ImageDecodeError("The file is empty.")
    digest = hashlib.sha256(data).hexdigest()
    if _has_raw_extension(filename):
        return _load_raw(data, digest, max_pixels, max_side, raw_white_balance)

    try:
        with warnings.catch_warnings():
            # We enforce our own pixel limit below; the warning would only add noise.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            pil = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError as err:
        raise ImageDecodeError(_too_large_message(None, None, max_pixels)) from err
    except UnidentifiedImageError as err:
        raise ImageDecodeError(
            "The file is not a supported image. Use JPEG, PNG, TIFF, HEIC/HEIF or DNG."
        ) from err
    except Exception as err:
        raise ImageDecodeError(_DAMAGED_MESSAGE) from err

    with pil:
        if pil.format == "TIFF" and _is_dng(pil):
            return _load_raw(data, digest, max_pixels, max_side, raw_white_balance)
        _check_size(pil.width, pil.height, max_pixels)
        high_bit = _is_high_bit_colour(pil)  # before EXIF: reading PNG EXIF loads the pixels
        exif = exif_from_pil(pil)
        orientation = _orientation(pil)
        original_w, original_h = _oriented_size(pil.width, pil.height, orientation)
        mime = _mime_type(pil.format, filename)
        try:
            image, bit_depth, notes = _decode_pil(pil, data, high_bit, orientation, max_side)
        except ImageDecodeError:
            raise
        except Image.DecompressionBombError as err:
            raise ImageDecodeError(_too_large_message(None, None, max_pixels)) from err
        except Exception as err:
            raise ImageDecodeError(_DAMAGED_MESSAGE) from err
        markers = synthetic_markers(pil)  # after decoding: PNG text after IDAT is read then

    h, w = image.shape[:2]
    if (w, h) != (original_w, original_h):
        notes.append(
            f"Downscaled at decode from {original_w} x {original_h} to {w} x {h} px "
            f"(longest side {max_side} px)."
        )
    return LoadedImage(
        image=image,
        exif=exif,
        mime_type=mime,
        sha256=digest,
        width_px=int(w),
        height_px=int(h),
        bit_depth=bit_depth,
        notes=notes,
        original_width_px=int(original_w),
        original_height_px=int(original_h),
        decode_scale=float(w) / float(original_w) if original_w else 1.0,
        orientation=orientation,
        synthetic_markers=markers,
    )


_DAMAGED_MESSAGE = (
    "The image file is damaged or incomplete and could not be read. "
    "Please export or capture it again."
)


def _too_large_message(w: int | None, h: int | None, max_pixels: int) -> str:
    limit = f"The limit is {max_pixels / 1e6:.0f} MP; please upload a smaller image."
    if w is None or h is None:
        return f"The image is too large to open safely. {limit}"
    return f"The image is too large ({w} x {h} px = {w * h / 1e6:.0f} MP). {limit}"


def _check_size(w: int, h: int, max_pixels: int) -> None:
    if w * h > max_pixels:
        raise ImageDecodeError(_too_large_message(w, h, max_pixels))
    if min(w, h) < MIN_SIDE_PX:
        raise ImageDecodeError(
            f"The image is too small ({w} x {h} px). Each side must be at least {MIN_SIDE_PX} px."
        )


def _has_raw_extension(filename: str | None) -> bool:
    return bool(filename) and PurePath(str(filename)).suffix.lower() in RAW_EXTENSIONS


def _is_dng(pil: Image.Image) -> bool:
    tags = getattr(pil, "tag_v2", None)
    return tags is not None and DNG_VERSION_TAG in tags


def _mime_type(fmt: str | None, filename: str | None) -> str:
    if fmt and fmt in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[fmt]
    if fmt and Image.MIME.get(fmt):
        return Image.MIME[fmt]
    guessed = mimetypes.guess_type(filename or "")[0]
    return guessed or "application/octet-stream"


def _oriented_size(width: int, height: int, orientation: int) -> tuple[int, int]:
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def _needs_shrink(width: int, height: int, max_side: int | None) -> bool:
    return max_side is not None and max(width, height) > max_side


def _target_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """(width, height) after shrinking so the longest side is ``max_side`` (as resize_max_side)."""
    if max(width, height) <= max_side:
        return width, height
    factor = max_side / float(max(width, height))
    return (
        max(1, min(max_side, int(round(width * factor)))),
        max(1, min(max_side, int(round(height * factor)))),
    )


def _fit(arr: np.ndarray, original: tuple[int, int], max_side: int | None) -> np.ndarray:
    """Bring a (possibly pre-reduced) decode to exactly the size a full decode + resize_max_side
    would give, so every decoder path agrees on the working size."""
    if max_side is None:
        return arr
    target = _target_size(original[0], original[1], max_side)
    if (arr.shape[1], arr.shape[0]) == target:
        return arr
    out = cv2.resize(np.ascontiguousarray(arr), target, interpolation=cv2.INTER_AREA)
    return out[..., None] if arr.ndim == 3 and out.ndim == 2 else out


def _decode_pil(
    pil: Image.Image,
    data: bytes,
    high_bit: bool,
    orientation: int = 1,
    max_side: int | None = None,
) -> tuple[np.ndarray, int, list[str]]:
    """Pixels -> float32 RGB [0, 1] with EXIF orientation applied, shrunk to ``max_side``."""
    notes: list[str] = []
    original = _oriented_size(pil.width, pil.height, orientation)
    if high_bit:
        # Pillow truncates 16-bit colour to 8 bits; OpenCV keeps the full precision.
        arr = _decode_high_bit_colour(data, max_side)
        arr = apply_exif_orientation(arr, orientation)
        return _finalise(_fit(arr, original, max_side)), 16, notes

    if pil.format == "JPEG" and _needs_shrink(pil.width, pil.height, max_side):
        # DCT-domain scaling (1/2, 1/4, 1/8): the decoder never produces the full-size image.
        factor = float(max_side) / float(max(pil.width, pil.height))  # type: ignore[arg-type]
        pil.draft(pil.mode, (math.ceil(pil.width * factor), math.ceil(pil.height * factor)))
    oriented = ImageOps.exif_transpose(pil)
    forced_depth: int | None = None
    if _needs_shrink(oriented.width, oriented.height, max_side):
        oriented, forced_depth = _reduce_pil(oriented, max_side)  # type: ignore[arg-type]
    oriented = _to_srgb(oriented, notes)
    if (
        max_side is not None
        and oriented.mode in ("RGB", "L")
        and _needs_shrink(oriented.width, oriented.height, max_side)
    ):
        # Final INTER_AREA step on the 8-bit pixels, so no float copy is ever larger than the
        # target size.
        target = _target_size(original[0], original[1], max_side)
        small = cv2.resize(np.asarray(oriented), target, interpolation=cv2.INTER_AREA)
        oriented = Image.fromarray(small)
    arr, bit_depth = _pil_to_float_rgb(oriented)
    return _finalise(_fit(arr, original, max_side)), forced_depth or bit_depth, notes


def _reduce_pil(pil: Image.Image, max_side: int) -> tuple[Image.Image, int | None]:
    """Integer box reduction in Pillow (8/16-bit pixels, before any float copy).

    The result keeps a longest side >= ``max_side``; the exact size is reached afterwards with
    INTER_AREA. Returns the image and the source bit depth when the mode had to be widened.
    """
    forced_depth: int | None = None
    if pil.mode == "P":
        pil = pil.convert("RGBA" if "transparency" in pil.info else "RGB")
    elif pil.mode == "PA":
        pil = pil.convert("RGBA")
    elif pil.mode == "1":
        pil = pil.convert("L")
    elif pil.mode.startswith("I;16"):
        pil, forced_depth = pil.convert("I"), 16
    factor = max(pil.width, pil.height) // max_side
    if factor >= 2:
        pil = pil.reduce(factor)
    return pil, forced_depth


def _is_high_bit_colour(pil: Image.Image) -> bool:
    if pil.mode not in ("RGB", "RGBA") or pil.format not in ("PNG", "TIFF"):
        return False
    return any(";16" in str(getattr(tile, "args", "")) for tile in (pil.tile or []))


def _decode_high_bit_colour(data: bytes, max_side: int | None = None) -> np.ndarray:
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 3 or raw.shape[2] not in (3, 4):
        raise ImageDecodeError(_DAMAGED_MESSAGE)
    scale = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 1.0
    if max_side is not None and raw.shape[2] == 3:
        raw, _ = resize_max_side(raw, max_side)  # INTER_AREA on the 16-bit integers
    arr = raw.astype(np.float32) / scale
    if arr.shape[2] == 4:  # BGRA: composite onto black, like the 8-bit path
        arr = arr[..., :3] * arr[..., 3:4]
        if max_side is not None:  # (rare) alpha images are composited before shrinking
            arr, _ = resize_max_side(arr, max_side)
    return arr[..., ::-1]  # BGR -> RGB


def _orientation(pil: Image.Image) -> int:
    try:
        value = pil.getexif().get(ExifTags.Base.Orientation, 1)
        return int(value) if 1 <= int(value) <= 8 else 1
    except Exception:
        return 1


def apply_exif_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    """NumPy equivalent of ``ImageOps.exif_transpose`` for arrays Pillow cannot hold."""
    ops = {
        2: lambda a: a[:, ::-1],
        3: lambda a: a[::-1, ::-1],
        4: lambda a: a[::-1, :],
        5: lambda a: np.swapaxes(a, 0, 1),
        6: lambda a: np.rot90(a, -1),
        7: lambda a: np.swapaxes(a[::-1, ::-1], 0, 1),
        8: lambda a: np.rot90(a, 1),
    }
    op = ops.get(orientation)
    return np.ascontiguousarray(op(arr) if op else arr)


def _to_srgb(pil: Image.Image, notes: list[str]) -> Image.Image:
    """Convert an embedded non-sRGB colour profile (e.g. Display P3) to sRGB.

    Colour thresholds (segmentation, confounders) assume sRGB; a wide-gamut phone photo would
    otherwise look slightly desaturated to them.
    """
    icc = pil.info.get("icc_profile")
    if not icc or pil.mode != "RGB":
        return pil
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = (ImageCms.getProfileDescription(src) or "").strip()
        if "srgb" in desc.lower():
            return pil
        converted = ImageCms.profileToProfile(
            pil, src, ImageCms.createProfile("sRGB"), outputMode="RGB"
        )
    except Exception:
        notes.append("Embedded colour profile could not be applied; colours used as stored.")
        return pil
    if converted is None:  # pragma: no cover - only when inPlace=True
        return pil
    notes.append(f"Converted from embedded colour profile '{desc or 'unnamed'}' to sRGB.")
    return converted


def _pil_to_float_rgb(pil: Image.Image) -> tuple[np.ndarray, int]:
    mode = pil.mode
    if mode.startswith("I;16"):
        gray = np.asarray(pil).astype(np.float32) / 65535.0
        return np.repeat(gray[..., None], 3, axis=2), 16
    if mode == "I":
        gray = np.asarray(pil).astype(np.float64)
        top = float(np.nanmax(gray)) if gray.size else 0.0
        scale = 65535.0 if top <= 65535.0 else top
        return np.repeat((gray / scale)[..., None], 3, axis=2), 16 if top > 255 else 8
    if mode == "F":
        gray = np.asarray(pil).astype(np.float64)
        finite = gray[np.isfinite(gray)]
        top = float(finite.max()) if finite.size else 1.0
        gray = gray / top if top > 1.0 else gray
        return np.repeat(gray[..., None], 3, axis=2), 32
    if "A" in mode or "a" in mode or "transparency" in pil.info:
        rgba = np.asarray(pil.convert("RGBA")).astype(np.float32) / 255.0
        # Transparent pixels carry no tissue signal: composite onto black (= background).
        return rgba[..., :3] * rgba[..., 3:4], 8
    return np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0, 8


def _finalise(arr: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0), dtype=np.float32)


def _load_raw(
    data: bytes,
    digest: str,
    max_pixels: int,
    max_side: int | None = None,
    white_balance: RawWhiteBalance = "camera",
) -> LoadedImage:
    """Decode DNG with rawpy (LibRaw): 16-bit output, sRGB transfer curve, no auto-brightening,
    orientation applied; camera white balance or none (``white_balance="unity"``)."""
    try:
        import rawpy
    except ImportError as err:
        raise ImageDecodeError(
            "DNG support needs the optional 'raw' extra (pip install 'mukha-pipeline[raw]'). "
            "Please upload a JPEG, PNG or HEIC instead."
        ) from err
    params: dict[str, Any] = {"output_bps": 16, "no_auto_bright": True, "gamma": RAW_GAMMA}
    if white_balance == "camera":
        params["use_camera_wb"] = True
    else:
        params.update(use_camera_wb=False, use_auto_wb=False, user_wb=[1.0, 1.0, 1.0, 1.0])
    try:
        with rawpy.imread(io.BytesIO(data)) as raw:
            sizes = raw.sizes
            width, height = int(sizes.width), int(sizes.height)
            _check_size(width, height, max_pixels)
            original_w, original_h = _oriented_size(
                width, height, 6 if int(getattr(sizes, "flip", 0) or 0) in (5, 6) else 1
            )
            # Demosaic at half resolution (2x2 binning) when that still covers the target size.
            params["half_size"] = max_side is not None and max(width, height) >= 2 * max_side
            rgb = raw.postprocess(**params)
    except ImageDecodeError:
        raise
    except Exception as err:
        raise ImageDecodeError(
            "The DNG file could not be decoded. It may be damaged or from an unsupported camera."
        ) from err
    arr = _fit(np.asarray(rgb), (original_w, original_h), max_side)  # INTER_AREA on uint16
    image = _finalise(arr.astype(np.float32) / 65535.0)
    h, w = image.shape[:2]
    _check_size(int(w), int(h), max_pixels)
    if white_balance == "camera":
        note = "Decoded from DNG raw with the camera's white balance"
    else:
        note = "Decoded from DNG raw without white balance (sensor channel ratios kept)"
    notes = [f"{note}, sRGB transfer curve, no auto-brightening."]
    if (w, h) != (original_w, original_h):
        notes.append(
            f"Downscaled at decode from {original_w} x {original_h} to {w} x {h} px "
            f"(longest side {max_side} px)."
        )
    return LoadedImage(
        image=image,
        exif=_exif_from_bytes(data),
        mime_type=DNG_MIME,
        sha256=digest,
        width_px=int(w),
        height_px=int(h),
        bit_depth=16,
        notes=notes,
        original_width_px=int(original_w),
        original_height_px=int(original_h),
        decode_scale=float(w) / float(original_w) if original_w else 1.0,
        synthetic_markers=_synthetic_markers_from_bytes(data),
    )


# --------------------------------------------------------------------------------------------
# EXIF
# --------------------------------------------------------------------------------------------


def _exif_from_bytes(data: bytes) -> ExifInfo:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(io.BytesIO(data)) as pil:
                return exif_from_pil(pil)
    except Exception:
        return ExifInfo()


def _synthetic_markers_from_bytes(data: bytes) -> list[str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(io.BytesIO(data)) as pil:
                return synthetic_markers(pil)
    except Exception:
        return []


def _mentions_synthetic(value: Any) -> bool:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return isinstance(value, str) and SYNTHETIC_TOKEN in value.lower()


def synthetic_markers(pil: Image.Image) -> list[str]:
    """Names of the metadata fields that label this image SYNTHETIC (never raises).

    Looks at the EXIF text fields the synthetic generator writes (Make, Model, ImageDescription,
    Software, ...), PNG text chunks, JPEG comments and XMP. A real photo has none of these.
    """
    found: list[str] = []
    try:
        base = pil.getexif()
        try:
            sub: Any = base.get_ifd(ExifTags.IFD.Exif)
        except Exception:
            sub = {}
        for code, name in _SYNTHETIC_EXIF_FIELDS.items():
            value = base.get(code)
            if value is None and sub:
                value = sub.get(code)
            if _mentions_synthetic(value):
                found.append(f"exif:{name}")
    except Exception:
        pass
    info = getattr(pil, "info", {}) or {}
    for key, value in info.items():
        if key in ("exif", "icc_profile"):
            continue
        if not _mentions_synthetic(value):
            continue
        if key in ("xmp", "XML:com.adobe.xmp"):
            found.append("xmp")
        elif key == "comment":
            found.append("jpeg:comment")
        elif pil.format == "PNG":
            found.append(f"png:{key}")
        else:
            found.append(f"info:{key}")
    return sorted(set(found))


def exif_from_pil(pil: Image.Image) -> ExifInfo:
    """Extract capture settings from EXIF. Never raises; missing or odd values become None."""
    try:
        base = pil.getexif()
    except Exception:
        return ExifInfo()
    try:
        sub: Any = base.get_ifd(ExifTags.IFD.Exif)
    except Exception:
        sub = {}

    def tag(code: int) -> Any:
        value = sub.get(code) if sub else None
        return base.get(code) if value is None else value

    exposure_s = _positive_float(tag(_EXIF_EXPOSURE_TIME), rational=True)
    iso = None
    for code in (_EXIF_ISO, _EXIF_ISO_SPEED, _EXIF_RECOMMENDED_EI):
        iso = _positive_float(tag(code))
        if iso is not None:
            break
    return ExifInfo(
        exposure_time_ms=exposure_s * 1000.0 if exposure_s is not None else None,
        iso=iso,
        white_balance=_white_balance(tag(_EXIF_WHITE_BALANCE)),
        captured_at=_exif_datetime(
            tag(_EXIF_DATETIME_ORIGINAL) or tag(_EXIF_DATETIME_DIGITIZED) or tag(_TIFF_DATETIME),
            tag(_EXIF_OFFSET_TIME_ORIGINAL),
        ),
        make=_clean_text(tag(_TIFF_MAKE)),
        model=_clean_text(tag(_TIFF_MODEL)),
    )


def _positive_float(value: Any, *, rational: bool = False) -> float | None:
    """Finite positive float, else None. Sequences: a (num, den) pair when ``rational``, else
    the first value (ISO may be stored as several values)."""
    if isinstance(value, tuple | list):
        if not value:
            return None
        if rational and len(value) == 2 and all(isinstance(v, int | float) for v in value):
            value = value[0] / value[1] if value[1] else None
        else:
            value = value[0]
    try:
        out = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return out if np.isfinite(out) and out > 0 else None


def _white_balance(value: Any) -> str | None:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    return {0: "auto", 1: "manual"}.get(code)


def _exif_datetime(value: Any, offset: Any) -> datetime | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        dt = datetime.strptime(text[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    tz = _exif_offset(offset)
    return dt.replace(tzinfo=tz) if tz is not None else dt


def _exif_offset(value: Any) -> timezone | None:
    text = _clean_text(value)
    if not text or len(text) < 6 or text[0] not in "+-" or text[3] != ":":
        return None
    try:
        hours, minutes = int(text[1:3]), int(text[4:6])
    except ValueError:
        return None
    sign = 1 if text[0] == "+" else -1
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).replace("\x00", "").strip()
    return text[:64] or None


# --------------------------------------------------------------------------------------------
# Encoding and resizing
# --------------------------------------------------------------------------------------------


def to_float_rgb(img: np.ndarray) -> np.ndarray:
    """Any (H, W), (H, W, 1|3|4) float / int / bool array -> float32 RGB in [0, 1], NaN-safe.

    Lets in-memory sources (synthetic generator, mock devices, tests) meet the same image
    convention as decoded uploads. Integer images are scaled by their dtype's maximum.
    """
    arr = np.asarray(img)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.float32)
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected an (H, W) or (H, W, 1|3|4) image, got shape {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return _finalise(arr[..., :3])


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Float [0, 1] (or bool / uint16 / int) image -> uint8, NaN-safe."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.copy()
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if arr.dtype == np.uint16:
        return ((arr.astype(np.uint32) + 128) // 257).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8)
    f = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)


def _to_uint16(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        return arr.copy()
    if arr.dtype == np.uint8:
        return arr.astype(np.uint16) * 257
    if arr.dtype == np.bool_:
        return arr.astype(np.uint16) * 65535
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 65535).astype(np.uint16)
    f = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.rint(f * 65535.0), 0, 65535).astype(np.uint16)


def encode_png(img: np.ndarray, *, bit_depth: Literal[8, 16] = 8) -> bytes:
    """Encode float [0, 1] or integer RGB / RGBA / gray as PNG. ``bit_depth=16`` keeps precision
    (useful for gray maps such as relative AF)."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (3, 4)) or arr.size == 0:
        raise ValueError(f"encode_png expects (H, W), (H, W, 3) or (H, W, 4); got {arr.shape}")
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")
    out = _to_uint16(arr) if bit_depth == 16 else to_uint8(arr)
    if out.ndim == 3:
        code = cv2.COLOR_RGB2BGR if out.shape[2] == 3 else cv2.COLOR_RGBA2BGRA
        out = cv2.cvtColor(np.ascontiguousarray(out), code)
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(out), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:  # pragma: no cover - OpenCV only fails on unsupported shapes, checked above
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink so the longest side is at most ``max_side``; never upscale.

    Returns ``(image, scale)`` where scale = new width / original width (1.0 when unchanged).
    INTER_AREA averages pixels when shrinking (no aliasing); bool masks use nearest neighbour so
    they stay bool.
    """
    if max_side < 1:
        raise ValueError("max_side must be >= 1")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    if max(h, w) <= max_side or h == 0 or w == 0:
        return arr, 1.0
    factor = max_side / float(max(h, w))
    new_w = max(1, min(max_side, int(round(w * factor))))
    new_h = max(1, min(max_side, int(round(h * factor))))
    if arr.dtype == np.bool_:
        small = cv2.resize(arr.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        out = small.astype(bool)
    else:
        out = cv2.resize(np.ascontiguousarray(arr), (new_w, new_h), interpolation=cv2.INTER_AREA)
        if arr.ndim == 3 and out.ndim == 2:  # OpenCV drops a singleton channel axis
            out = out[..., None]
    return out, new_w / float(w)
