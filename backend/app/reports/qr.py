"""QR code linking the printed report to the secure (login-protected) online record."""

from __future__ import annotations

import qrcode
import qrcode.image.svg
from qrcode.constants import ERROR_CORRECT_M

from app.reports.images import svg_data_uri


def qr_svg(url: str) -> str:
    """Crisp vector QR code (SVG path) for ``url``; error correction M survives print smudges."""
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_M, box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(image_factory=qrcode.image.svg.SvgPathImage)
    svg = img.to_string(encoding="unicode")
    return svg if isinstance(svg, str) else svg.decode("utf-8")


def qr_data_uri(url: str) -> str:
    return svg_data_uri(qr_svg(url))
