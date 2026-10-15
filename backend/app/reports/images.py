"""Artifact images for the report: read, downscale, mark the flagged region, embed as data URI.

A missing, unreadable or undecodable artifact never breaks a report: the template shows a calm
"Image not available" placeholder instead.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageDraw, UnidentifiedImageError

from app.reports.types import ArtifactReader
from mukha_pipeline.schema.models import NormBox

log = logging.getLogger(__name__)

MAX_SIDE_PX = 720  # ~ 55 mm wide panel at > 300 dpi; keeps the PDF small
JPEG_QUALITY = 86


@dataclass
class EmbeddedImage:
    data_uri: str | None
    key: str | None
    reason: str | None = None  # why it is missing, for the placeholder

    @property
    def available(self) -> bool:
        return self.data_uri is not None


def data_uri(payload: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def svg_data_uri(svg: str) -> str:
    return data_uri(svg.encode("utf-8"), "image/svg+xml")


def read_artifact(reader: ArtifactReader, key: str | None) -> bytes | None:
    if not key:
        return None
    try:
        payload = reader(key)
    except Exception:  # a storage error must not break the report
        log.warning("artifact reader failed for %s", key, exc_info=True)
        return None
    if payload is None or not isinstance(payload, bytes | bytearray) or len(payload) == 0:
        return None
    return bytes(payload)


def union_box(boxes: list[NormBox] | None) -> NormBox | None:
    """Smallest box containing every box (``None`` for no boxes)."""
    if not boxes:
        return None
    x0 = min(b.x for b in boxes)
    y0 = min(b.y for b in boxes)
    x1 = max(b.x + b.w for b in boxes)
    y1 = max(b.y + b.h for b in boxes)
    return NormBox(x=x0, y=y0, w=min(1.0, x1) - x0, h=min(1.0, y1) - y0)


def crop_window(
    size: tuple[int, int], aspect: tuple[float, float], focus: NormBox | None
) -> tuple[int, int, int, int]:
    """Largest ``aspect`` (width, height) window inside an image of ``size`` px, centred on
    ``focus`` (normalised box; image centre when ``None``) and kept inside the image."""
    w, h = size
    ratio = aspect[0] / aspect[1]
    if w / h > ratio:
        win_w, win_h = h * ratio, float(h)
    else:
        win_w, win_h = float(w), w / ratio
    cx, cy = w / 2, h / 2
    if focus is not None:
        cx = (focus.x + focus.w / 2) * w
        cy = (focus.y + focus.h / 2) * h
    x0 = min(max(0.0, cx - win_w / 2), w - win_w)
    y0 = min(max(0.0, cy - win_h / 2), h - win_h)
    return round(x0), round(y0), round(x0 + win_w), round(y0 + win_h)


def _boxes_in_window(
    boxes: list[NormBox], size: tuple[int, int], window: tuple[int, int, int, int]
) -> list[NormBox]:
    """Re-express normalised boxes of the full image in the cropped window (clipped)."""
    w, h = size
    x0, y0, x1, y1 = window
    ww, wh = max(1, x1 - x0), max(1, y1 - y0)
    out: list[NormBox] = []
    for b in boxes:
        bx0 = min(max((b.x * w - x0) / ww, 0.0), 1.0)
        by0 = min(max((b.y * h - y0) / wh, 0.0), 1.0)
        bx1 = min(max(((b.x + b.w) * w - x0) / ww, 0.0), 1.0)
        by1 = min(max(((b.y + b.h) * h - y0) / wh, 0.0), 1.0)
        if bx1 > bx0 and by1 > by0:
            out.append(NormBox(x=bx0, y=by0, w=bx1 - bx0, h=by1 - by0))
    return out


def _draw_region_marks(img: Image.Image, boxes: list[NormBox]) -> None:
    """Corner brackets around each flagged region (white outer + dark inner stroke)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    stroke = max(2, round(max(w, h) / 260))
    for box in boxes:
        pad = 0.02
        x0 = max(0.0, box.x - pad) * w
        y0 = max(0.0, box.y - pad) * h
        x1 = min(1.0, box.x + box.w + pad) * w
        y1 = min(1.0, box.y + box.h + pad) * h
        arm = max(6.0, 0.28 * min(x1 - x0, y1 - y0))
        segments = [
            ((x0, y0), (x0 + arm, y0)), ((x0, y0), (x0, y0 + arm)),
            ((x1, y0), (x1 - arm, y0)), ((x1, y0), (x1, y0 + arm)),
            ((x0, y1), (x0 + arm, y1)), ((x0, y1), (x0, y1 - arm)),
            ((x1, y1), (x1 - arm, y1)), ((x1, y1), (x1, y1 - arm)),
        ]  # fmt: skip
        for seg in segments:
            draw.line(seg, fill=(255, 255, 255), width=stroke + 3)
        for seg in segments:
            draw.line(seg, fill=(17, 24, 39), width=stroke)


def embed_artifact(
    reader: ArtifactReader,
    key: str | None,
    *,
    boxes: list[NormBox] | None = None,
    max_side: int = MAX_SIDE_PX,
    aspect: tuple[float, float] | None = None,
    focus: NormBox | None = None,
) -> EmbeddedImage:
    """Read an image artifact and return it as a compact JPEG data URI (or a placeholder).

    ``boxes`` (normalised to this image) get corner brackets. With ``aspect`` (width, height,
    e.g. the printed frame in mm) the image is cropped to that shape around ``focus`` (default:
    the union of ``boxes``; the image centre without either), so the frame is filled instead
    of letterboxed. Panels that share a frame get the same crop when given the same focus.
    """
    if not key:
        return EmbeddedImage(None, None, "Not captured")
    payload = read_artifact(reader, key)
    if payload is None:
        return EmbeddedImage(None, key, "Image not available")
    try:
        with Image.open(io.BytesIO(payload)) as src:
            src.load()
            img = src
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.getchannel("A"))
                img = background
            else:
                img = img.convert("RGB")
            marks = list(boxes or [])
            if aspect is not None:
                centre = focus if focus is not None else union_box(marks)
                window = crop_window(img.size, aspect, centre)
                marks = _boxes_in_window(marks, img.size, window)
                img = img.crop(window)
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            if marks:
                _draw_region_marks(img, marks)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    except (UnidentifiedImageError, OSError, ValueError):
        log.warning("artifact %s could not be decoded", key, exc_info=True)
        return EmbeddedImage(None, key, "Image could not be read")
    return EmbeddedImage(data_uri(out.getvalue(), "image/jpeg"), key)
