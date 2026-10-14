"""UploadAdapter: WL / AF / dark images and spectral files uploaded as bytes.

Phase 1's only real input path. Images are decoded by ``io.images`` (orientation, bit depth,
colour profile) and their EXIF capture settings are recorded, so exposure normalisation and the
report can show what the camera actually did.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from datetime import datetime

from mukha_pipeline.capture.base import KIND_LABEL, image_metadata
from mukha_pipeline.io.images import ImageDecodeError, load_image_bytes
from mukha_pipeline.io.spectral import ParsedSpectral, parse_spectral_file
from mukha_pipeline.schema.models import (
    CaptureKind,
    Diascopy,
    SiteRole,
    SourceAdapter,
    VisualFlag,
)
from mukha_pipeline.sites import OralSite
from mukha_pipeline.types import ImageCapture, SiteInput


class UploadDecodeError(ImageDecodeError):
    """An uploaded image could not be decoded; the message names which one."""

    def __init__(self, kind: CaptureKind, file_name: str | None, message: str) -> None:
        self.kind = kind
        self.file_name = file_name
        label = KIND_LABEL[kind]
        where = f"{label} ({file_name})" if file_name else label
        super().__init__(f"{where}: {message}")


class UploadAdapter:
    """Builds ``SiteInput`` and spectral readings from uploaded bytes."""

    adapter: SourceAdapter = SourceAdapter.UPLOAD

    def image_capture(
        self,
        data: bytes,
        kind: CaptureKind,
        site: OralSite | str,
        *,
        file_name: str | None = None,
        device_id: str | None = None,
        firmware_version: str | None = None,
        synthetic: bool = False,
    ) -> ImageCapture:
        """Decode one image and attach its metadata. Raises :class:`UploadDecodeError`."""
        try:
            loaded = load_image_bytes(data, file_name)
        except ImageDecodeError as err:
            raise UploadDecodeError(kind, file_name, str(err)) from err
        exif = loaded.exif
        meta = image_metadata(
            kind,
            site,
            capture_id=str(uuid.uuid4()),
            source_adapter=self.adapter,
            width_px=loaded.width_px,
            height_px=loaded.height_px,
            exposure_time_ms=exif.exposure_time_ms,
            iso=exif.iso,
            white_balance=exif.white_balance,
            captured_at=exif.captured_at,
            file_name=file_name,
            file_sha256=loaded.sha256,
            mime_type=loaded.mime_type,
            device_id=device_id,
            firmware_version=firmware_version,
            synthetic=synthetic,
            extra={
                "camera_make": exif.make,
                "camera_model": exif.model,
                "bit_depth": loaded.bit_depth,
                "decode_notes": " ".join(loaded.notes) or None,
            },
        )
        return ImageCapture(image=loaded.image, meta=meta)

    def site_input(
        self,
        site: OralSite | str,
        wl: bytes,
        af: bytes,
        dark: bytes | None = None,
        *,
        filenames: Mapping[str, str | None] | None = None,
        role: SiteRole | str = SiteRole.EXAMINED,
        visual_flag: VisualFlag | None = None,
        diascopy: Diascopy | str = Diascopy.NOT_DONE,
        px_per_mm: float | None = None,
        device_id: str | None = None,
        firmware_version: str | None = None,
        synthetic: bool = False,
    ) -> SiteInput:
        """WL + AF (+ optional dark frame) for one oral site.

        ``filenames`` maps ``"wl"``, ``"af"`` and ``"dark"`` to the uploaded file names. The dark
        frame is optional: without it, ambient light cannot be subtracted and QC reports that.
        """
        oral_site = OralSite(site)
        if px_per_mm is not None and not (math.isfinite(px_per_mm) and px_per_mm > 0):
            raise ValueError("px_per_mm must be a positive number when given.")
        names = dict(filenames or {})
        common = {
            "device_id": device_id,
            "firmware_version": firmware_version,
            "synthetic": synthetic,
        }
        wl_cap = self.image_capture(
            wl, CaptureKind.WHITE_LIGHT, oral_site, file_name=names.get("wl"), **common
        )
        af_cap = self.image_capture(
            af, CaptureKind.AUTOFLUORESCENCE, oral_site, file_name=names.get("af"), **common
        )
        dark_cap = None
        if dark:
            dark_cap = self.image_capture(
                dark, CaptureKind.DARK_FRAME, oral_site, file_name=names.get("dark"), **common
            )
        return SiteInput(
            site=oral_site,
            wl=wl_cap,
            af=af_cap,
            dark=dark_cap,
            role=SiteRole(role),
            visual_flag=visual_flag or VisualFlag(),
            diascopy=Diascopy(diascopy),
            px_per_mm=px_per_mm,
        )

    def spectral(
        self,
        data: bytes,
        filename: str,
        *,
        device_id: str | None = None,
        synthetic: bool = False,
        now: datetime | None = None,
    ) -> ParsedSpectral:
        """Parse an uploaded Confirm-probe CSV/JSON file. Raises ``SpectralParseError``."""
        return parse_spectral_file(
            data,
            filename,
            source_adapter=self.adapter,
            device_id=device_id,
            synthetic=synthetic,
            now=now,
        )
