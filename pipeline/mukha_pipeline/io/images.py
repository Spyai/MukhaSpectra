"""Image decoding and encoding.

Every image entering the pipeline goes through :func:`load_image_bytes`, which returns float32 RGB
in [0, 1] (see ``mukha_pipeline.types``) plus the capture settings found in EXIF. Decoding is
defensive because uploads come from many phones and apps: EXIF orientation is applied, 16-bit
PNG/TIFF keep their full precision, embedded colour profiles (e.g. Display P3 on iPhones) are
converted to sRGB, and anything unreadable raises :class:`ImageDecodeError` with a message a
health worker can act on.

GPS and other identifying EXIF fields are deliberately never read.
"""

from __future__ import annotations

import hashlib
import io
import mimetypes
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import PurePath
from typing import Any, Literal

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageCms, ImageOps, UnidentifiedImageError

try:  # HEIC/HEIF (iPhone default) support
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover - pillow-heif is a core dependency
    HEIF_AVAILABLE = False

MAX_IMAGE_PIXELS = 100_000_000  # ~100 MP: above any phone sensor, below decompression-bomb sizes
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


# --------------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------------


def load_image_bytes(
    data: bytes, filename: str | None = None, *, max_pixels: int = MAX_IMAGE_PIXELS
) -> LoadedImage:
    """Decode JPEG, PNG, TIFF, HEIC/HEIF or DNG bytes into float32 RGB in [0, 1].

    Raises :class:`ImageDecodeError` with a plain-language message on any failure.
    """
    if not data:
        raise ImageDecodeError("The file is empty.")
    digest = hashlib.sha256(data).hexdigest()
    if _has_raw_extension(filename):
        return _load_raw(data, digest, max_pixels)

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
            return _load_raw(data, digest, max_pixels)
        _check_size(pil.width, pil.height, max_pixels)
        high_bit = _is_high_bit_colour(pil)  # before EXIF: reading PNG EXIF loads the pixels
        exif = exif_from_pil(pil)
        mime = _mime_type(pil.format, filename)
        try:
            image, bit_depth, notes = _decode_pil(pil, data, high_bit)
        except ImageDecodeError:
            raise
        except Image.DecompressionBombError as err:
            raise ImageDecodeError(_too_large_message(None, None, max_pixels)) from err
        except Exception as err:
            raise ImageDecodeError(_DAMAGED_MESSAGE) from err

    h, w = image.shape[:2]
    return LoadedImage(
        image=image,
        exif=exif,
        mime_type=mime,
        sha256=digest,
        width_px=int(w),
        height_px=int(h),
        bit_depth=bit_depth,
        notes=notes,
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


def _decode_pil(pil: Image.Image, data: bytes, high_bit: bool) -> tuple[np.ndarray, int, list[str]]:
    """Pixels -> float32 RGB [0, 1] with EXIF orientation applied."""
    notes: list[str] = []
    if high_bit:
        # Pillow truncates 16-bit colour to 8 bits; OpenCV keeps the full precision.
        arr = _decode_high_bit_colour(data)
        arr = apply_exif_orientation(arr, _orientation(pil))
        return _finalise(arr), 16, notes

    oriented = ImageOps.exif_transpose(pil)
    oriented = _to_srgb(oriented, notes)
    arr, bit_depth = _pil_to_float_rgb(oriented)
    return _finalise(arr), bit_depth, notes


def _is_high_bit_colour(pil: Image.Image) -> bool:
    if pil.mode not in ("RGB", "RGBA") or pil.format not in ("PNG", "TIFF"):
        return False
    return any(";16" in str(getattr(tile, "args", "")) for tile in (pil.tile or []))


def _decode_high_bit_colour(data: bytes) -> np.ndarray:
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 3 or raw.shape[2] not in (3, 4):
        raise ImageDecodeError(_DAMAGED_MESSAGE)
    scale = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 1.0
    arr = raw.astype(np.float32) / scale
    if arr.shape[2] == 4:  # BGRA: composite onto black, like the 8-bit path
        arr = arr[..., :3] * arr[..., 3:4]
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


def _load_raw(data: bytes, digest: str, max_pixels: int) -> LoadedImage:
    """Decode DNG with rawpy (LibRaw): camera white balance, 16-bit output, orientation applied."""
    try:
        import rawpy
    except ImportError as err:
        raise ImageDecodeError(
            "DNG support needs the optional 'raw' extra (pip install 'mukha-pipeline[raw]'). "
            "Please upload a JPEG, PNG or HEIC instead."
        ) from err
    try:
        with rawpy.imread(io.BytesIO(data)) as raw:
            sizes = raw.sizes
            _check_size(int(sizes.width), int(sizes.height), max_pixels)
            rgb = raw.postprocess(use_camera_wb=True, output_bps=16)
    except ImageDecodeError:
        raise
    except Exception as err:
        raise ImageDecodeError(
            "The DNG file could not be decoded. It may be damaged or from an unsupported camera."
        ) from err
    image = _finalise(np.asarray(rgb, dtype=np.float32) / 65535.0)
    h, w = image.shape[:2]
    _check_size(int(w), int(h), max_pixels)
    return LoadedImage(
        image=image,
        exif=_exif_from_bytes(data),
        mime_type=DNG_MIME,
        sha256=digest,
        width_px=int(w),
        height_px=int(h),
        bit_depth=16,
        notes=["Decoded from DNG raw with the camera's white balance."],
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
