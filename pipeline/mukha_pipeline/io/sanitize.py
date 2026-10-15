"""Remove identifying metadata from captures before they are stored (DPDP data minimisation).

Phone photos carry far more than pixels: GPS position (a patient's home during door-to-door
screening), owner / artist names, device serial numbers, maker notes, XMP and IPTC blocks,
comments and trailing data such as secondary images or motion-photo video. Re-analysis needs only
the pixels, the EXIF orientation and the capture settings, and the settings are already copied
into the capture metadata when the image is decoded.

:func:`strip_identifying_metadata` works on the container, never on the compressed image data, so
the decoded pixels are bit-for-bit unchanged:

* **JPEG**: every APPn segment except APP0 (JFIF), APP2 ICC profile and APP14 (Adobe colour
  transform) is dropped, as are comments (COM) and anything after the first EOI. When the
  original had an orientation, a minimal EXIF block with only that tag is written back.
* **PNG**: ``eXIf``, ``tEXt``, ``iTXt`` and ``zTXt`` chunks are dropped, as is anything after
  ``IEND``; a minimal ``eXIf`` with only the orientation is written back when needed.

Limitation: other formats (HEIC/HEIF, DNG, TIFF, WebP, BMP) are returned unchanged
(``sanitised=False``), so those files are stored as uploaded, with their metadata.
Malformed files are also returned unchanged; they decoded, so they are still analysed.
"""

from __future__ import annotations

import io
import warnings
import zlib
from dataclasses import dataclass

from PIL import ExifTags, Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_EXIF_HEADER = b"Exif\x00\x00"
_ICC_HEADER = b"ICC_PROFILE\x00"
_PNG_DROP = (b"eXIf", b"tEXt", b"iTXt", b"zTXt")

_APP0, _APP1, _APP2, _APP13, _APP14 = 0xE0, 0xE1, 0xE2, 0xED, 0xEE
_SOS, _EOI, _COM = 0xDA, 0xD9, 0xFE
_STANDALONE = frozenset({0x01, *range(0xD0, 0xD8)})


@dataclass(frozen=True)
class SanitizedImage:
    data: bytes
    removed: tuple[str, ...]  # what was dropped, e.g. ("EXIF", "XMP", "comment")
    sanitised: bool  # False: format not handled or malformed; ``data`` is the input unchanged
    orientation: int = 1  # EXIF orientation kept in the output


class _MalformedError(ValueError):
    pass


def strip_identifying_metadata(data: bytes, *, orientation: int | None = None) -> SanitizedImage:
    """Return ``data`` without identifying metadata (JPEG / PNG), pixels unchanged.

    ``orientation`` is the EXIF orientation the decoder applied; when omitted it is read from the
    file (which, for PNG, may mean decoding it).
    """
    try:
        if data.startswith(b"\xff\xd8"):
            return _strip_jpeg(data, orientation)
        if data.startswith(PNG_SIGNATURE):
            return _strip_png(data, orientation)
    except _MalformedError:
        pass
    return SanitizedImage(data=data, removed=(), sanitised=False)


def _orientation_of(data: bytes) -> int:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(io.BytesIO(data)) as pil:
                value = int(pil.getexif().get(ExifTags.Base.Orientation, 1))
    except Exception:
        return 1
    return value if 1 <= value <= 8 else 1


def _minimal_exif(orientation: int) -> bytes:
    """``Exif\\0\\0`` + a TIFF block holding only the orientation tag."""
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    blob = exif.tobytes()
    return blob if blob.startswith(_EXIF_HEADER) else _EXIF_HEADER + blob


# --------------------------------------------------------------------------------------------
# JPEG
# --------------------------------------------------------------------------------------------


def _jpeg_drop_reason(marker: int, payload: bytes) -> str | None:
    if marker == _COM:
        return "comment"
    if marker == _APP1:
        if payload.startswith(_EXIF_HEADER):
            return "EXIF"
        if b"ns.adobe.com/xap" in payload[:64] or b"ns.adobe.com/xmp" in payload[:64]:
            return "XMP"
        return "APP1"
    if marker == _APP2:
        return None if payload.startswith(_ICC_HEADER) else "APP2"
    if marker == _APP13:
        return "IPTC"
    if 0xE0 <= marker <= 0xEF and marker not in (_APP0, _APP14):
        return f"APP{marker - 0xE0}"
    return None


def _strip_jpeg(data: bytes, orientation: int | None) -> SanitizedImage:
    n = len(data)
    header: list[tuple[int, bytes]] = []  # (marker, raw segment bytes) before the image data
    removed: list[str] = []
    body = bytearray()  # first SOS onwards
    i = 2
    in_header = True
    ended = False
    while i < n:
        if data[i] != 0xFF:
            raise _MalformedError("expected a marker")
        while i < n and data[i] == 0xFF:  # fill bytes
            i += 1
        if i >= n:
            raise _MalformedError("truncated marker")
        marker = data[i]
        i += 1
        if marker == _EOI:
            body += b"\xff\xd9"
            ended = True
            break
        if marker in _STANDALONE:
            if in_header:
                header.append((marker, bytes([0xFF, marker])))
            else:
                body += bytes([0xFF, marker])
            continue
        if i + 2 > n:
            raise _MalformedError("truncated segment length")
        length = int.from_bytes(data[i : i + 2], "big")
        if length < 2 or i + length > n:
            raise _MalformedError("bad segment length")
        segment = bytes([0xFF, marker]) + data[i : i + length]
        payload = data[i + 2 : i + length]
        i += length
        reason = _jpeg_drop_reason(marker, payload)
        if reason is not None:
            removed.append(reason)
        elif in_header and marker != _SOS:
            header.append((marker, segment))
        else:
            body += segment
        if marker == _SOS:
            in_header = False
            # Entropy-coded data runs to the next marker that is not stuffing (FF00), a restart
            # marker (FFD0-FFD7) or a fill byte.
            j = i
            while True:
                k = data.find(b"\xff", j)
                if k < 0 or k + 1 >= n:
                    raise _MalformedError("no end of image")
                following = data[k + 1]
                if following == 0x00 or 0xD0 <= following <= 0xD7:
                    j = k + 2
                    continue
                if following == 0xFF:
                    j = k + 1
                    continue
                break
            body += data[i:k]
            i = k
    if not ended:
        raise _MalformedError("no end of image")
    if in_header:
        raise _MalformedError("no image data")
    if i < n:
        removed.append("trailing data")
    if not removed:
        return SanitizedImage(data=data, removed=(), sanitised=True, orientation=orientation or 1)

    kept_orientation = orientation if orientation is not None else _orientation_of(data)
    out = bytearray(b"\xff\xd8")
    inserted = False
    for marker, segment in header:
        if not inserted and marker != _APP0:
            out += _exif_segment(kept_orientation)
            inserted = True
        out += segment
    if not inserted:
        out += _exif_segment(kept_orientation)
    out += body
    return SanitizedImage(
        data=bytes(out),
        removed=tuple(dict.fromkeys(removed)),
        sanitised=True,
        orientation=kept_orientation,
    )


def _exif_segment(orientation: int) -> bytes:
    if orientation == 1:
        return b""
    payload = _minimal_exif(orientation)
    return b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload


# --------------------------------------------------------------------------------------------
# PNG
# --------------------------------------------------------------------------------------------


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    body = ctype + payload
    return len(payload).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")


def _strip_png(data: bytes, orientation: int | None) -> SanitizedImage:
    n = len(data)
    pos = len(PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    removed: list[str] = []
    ended = False
    while pos < n:
        if pos + 12 > n:
            raise _MalformedError("truncated chunk")
        length = int.from_bytes(data[pos : pos + 4], "big")
        ctype = data[pos + 4 : pos + 8]
        end = pos + 12 + length
        if end > n:
            raise _MalformedError("truncated chunk")
        chunk = data[pos:end]
        pos = end
        if ctype in _PNG_DROP:
            removed.append(ctype.decode("ascii"))
            continue
        chunks.append((ctype, chunk))
        if ctype == b"IEND":
            ended = True
            break
    if not ended or not chunks or chunks[0][0] != b"IHDR":
        raise _MalformedError("not a complete PNG")
    if pos < n:
        removed.append("trailing data")
    if not removed:
        return SanitizedImage(data=data, removed=(), sanitised=True, orientation=orientation or 1)

    kept_orientation = orientation if orientation is not None else _orientation_of(data)
    out = bytearray(PNG_SIGNATURE)
    inserted = kept_orientation == 1
    for ctype, chunk in chunks:
        if not inserted and ctype in (b"IDAT", b"IEND"):
            out += _png_chunk(b"eXIf", _minimal_exif(kept_orientation)[len(_EXIF_HEADER) :])
            inserted = True
        out += chunk
    return SanitizedImage(
        data=bytes(out),
        removed=tuple(dict.fromkeys(removed)),
        sanitised=True,
        orientation=kept_orientation,
    )
