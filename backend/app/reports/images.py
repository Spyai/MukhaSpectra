"""Artifact images for the report: read, downscale, mark the flagged region, embed as data URI.

A missing, unreadable or undecodable artifact never breaks a report: the template shows a calm
"Image not available" placeholder instead.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

from mukha_pipeline.schema.models import NormBox
from PIL import Image, ImageDraw, UnidentifiedImageError

from app.reports.types import ArtifactReader

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
) -> EmbeddedImage:
    """Read an image artifact and return it as a compact JPEG data URI (or a placeholder)."""
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
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            if boxes:
                _draw_region_marks(img, boxes)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    except (UnidentifiedImageError, OSError, ValueError):
        log.warning("artifact %s could not be decoded", key, exc_info=True)
        return EmbeddedImage(None, key, "Image could not be read")
    return EmbeddedImage(data_uri(out.getvalue(), "image/jpeg"), key)
