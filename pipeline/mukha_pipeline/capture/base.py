"""Capture-source abstraction shared by every way data can enter the pipeline.

Server-side mirror of the PWA ``DeviceAdapter``: upload, phone camera, Scout clip (BLE),
dedicated camera, Confirm probe, mock device and the synthetic generator all produce the same
``SiteInput`` / ``SpectralInput`` types with complete ``CaptureMetadata``, so analysis never
needs to know where a capture came from.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable

from mukha_pipeline.schema.models import CaptureKind, CaptureMetadata, LedMode, SourceAdapter
from mukha_pipeline.sites import OralSite

ExtraValue = str | float | int | bool | None


@runtime_checkable
class CaptureSource(Protocol):
    """Anything that produces captures. ``adapter`` is recorded in every capture's metadata."""

    adapter: SourceAdapter


# The LED that is on for each capture kind: WL under white light, AF under the 405 nm violet LED
# behind the long-pass emission filter, dark frame with every LED off (ambient-light estimate).
LED_FOR_KIND: dict[CaptureKind, LedMode] = {
    CaptureKind.WHITE_LIGHT: LedMode.WHITE,
    CaptureKind.AUTOFLUORESCENCE: LedMode.VIOLET,
    CaptureKind.DARK_FRAME: LedMode.OFF,
    CaptureKind.SPECTRAL: LedMode.WHITE,
}

KIND_LABEL: dict[CaptureKind, str] = {
    CaptureKind.WHITE_LIGHT: "White-light image",
    CaptureKind.AUTOFLUORESCENCE: "Autofluorescence image",
    CaptureKind.DARK_FRAME: "Dark frame",
    CaptureKind.SPECTRAL: "Spectral reading",
}


def image_metadata(
    kind: CaptureKind,
    site: OralSite | str,
    *,
    capture_id: str,
    source_adapter: SourceAdapter,
    width_px: int,
    height_px: int,
    exposure_time_ms: float | None = None,
    iso: float | None = None,
    white_balance: str | None = None,
    captured_at: datetime | None = None,
    file_name: str | None = None,
    file_sha256: str | None = None,
    mime_type: str | None = None,
    device_id: str | None = None,
    firmware_version: str | None = None,
    synthetic: bool = False,
    extra: Mapping[str, ExtraValue] | None = None,
) -> CaptureMetadata:
    """CaptureMetadata for one image, with the LED mode implied by its kind."""
    return CaptureMetadata(
        capture_id=capture_id,
        kind=kind,
        oral_site=OralSite(site),
        source_adapter=source_adapter,
        device_id=device_id,
        firmware_version=firmware_version,
        led_mode=LED_FOR_KIND[kind],
        exposure_time_ms=exposure_time_ms,
        iso=iso,
        white_balance=white_balance,
        captured_at=captured_at,
        width_px=width_px,
        height_px=height_px,
        file_name=file_name,
        file_sha256=file_sha256,
        mime_type=mime_type,
        synthetic=synthetic,
        extra={k: v for k, v in (extra or {}).items() if v is not None},
    )
