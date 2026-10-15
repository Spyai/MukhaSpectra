"""UploadAdapter: WL / AF / dark images and spectral files uploaded as bytes.

Phase 1's only real input path. Images are decoded by ``io.images`` (orientation, bit depth,
colour profile) and their EXIF capture settings are recorded, so exposure normalisation and the
report can show what the camera actually did.

* Memory: images are decoded at most ``UPLOAD_DECODE_MAX_SIDE`` (2048) px on the longest side,
  twice the 1024 px working resolution, so a 48 MP photo never becomes a full-size float array.
  The metadata keeps the capture's own size; the decoded size and scale go into ``extra``, and
  ``px_per_mm`` (given for the original pixels) is converted to the decoded pixels.
* DNG: white-light captures use the camera's white balance; autofluorescence and dark frames are
  decoded without white balance, because their green/red ratio is the measurement.
* SYNTHETIC: a capture is marked synthetic when the caller says so, when the uploaded file name
  starts with ``SYNTHETIC`` or when its metadata carries the SYNTHETIC label written by
  ``mukha_pipeline.synthetic``. The uploaded file name is only inspected, never stored.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from datetime import datetime

from mukha_pipeline.capture.base import KIND_LABEL, image_metadata
from mukha_pipeline.io.images import ImageDecodeError, RawWhiteBalance, load_image_bytes
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

# 2x the analysis working resolution (analysis.v1.yaml working_max_side_px = 1024).
UPLOAD_DECODE_MAX_SIDE = 2048
SYNTHETIC_FILE_PREFIX = "synthetic"


def raw_white_balance_for(kind: CaptureKind) -> RawWhiteBalance:
    """DNG white balance per capture kind: never white-balance fluorescence (or its dark frame,
    which is subtracted from it); white light keeps the camera's balance."""
    return "camera" if CaptureKind(kind) == CaptureKind.WHITE_LIGHT else "unity"


def is_synthetic_file_name(name: str | None) -> bool:
    """True for file names produced by the synthetic generator (``SYNTHETIC_...``)."""
    if not name:
        return False
    base = name.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    return base.startswith(SYNTHETIC_FILE_PREFIX)


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

    def __init__(self, *, max_side: int | None = UPLOAD_DECODE_MAX_SIDE) -> None:
        self.max_side = max_side

    def image_capture(
        self,
        data: bytes,
        kind: CaptureKind,
        site: OralSite | str,
        *,
        file_name: str | None = None,
        source_file_name: str | None = None,
        device_id: str | None = None,
        firmware_version: str | None = None,
        synthetic: bool = False,
    ) -> ImageCapture:
        """Decode one image and attach its metadata. Raises :class:`UploadDecodeError`.

        ``file_name`` is recorded in the metadata (and decides DNG handling); the server passes
        a neutral name. ``source_file_name`` is the name the user uploaded: it is only checked
        for the SYNTHETIC prefix and never stored, because it may contain the patient's name.
        """
        try:
            loaded = load_image_bytes(
                data,
                file_name or source_file_name,
                max_side=self.max_side,
                raw_white_balance=raw_white_balance_for(kind),
            )
        except ImageDecodeError as err:
            raise UploadDecodeError(kind, file_name, str(err)) from err
        exif = loaded.exif
        markers = (["file_name"] if is_synthetic_file_name(source_file_name) else []) + [
            *loaded.synthetic_markers
        ]
        if is_synthetic_file_name(file_name) and "file_name" not in markers:
            markers.insert(0, "file_name")
        downscaled = (loaded.width_px, loaded.height_px) != (
            loaded.original_width_px or loaded.width_px,
            loaded.original_height_px or loaded.height_px,
        )
        meta = image_metadata(
            kind,
            site,
            capture_id=str(uuid.uuid4()),
            source_adapter=self.adapter,
            width_px=loaded.original_width_px or loaded.width_px,
            height_px=loaded.original_height_px or loaded.height_px,
            exposure_time_ms=exif.exposure_time_ms,
            iso=exif.iso,
            white_balance=exif.white_balance,
            captured_at=exif.captured_at,
            file_name=file_name,
            file_sha256=loaded.sha256,
            mime_type=loaded.mime_type,
            device_id=device_id,
            firmware_version=firmware_version,
            synthetic=synthetic or bool(markers),
            extra={
                "camera_make": exif.make,
                "camera_model": exif.model,
                "bit_depth": loaded.bit_depth,
                "decode_notes": " ".join(loaded.notes) or None,
                "decoded_width_px": loaded.width_px if downscaled else None,
                "decoded_height_px": loaded.height_px if downscaled else None,
                "decode_scale": round(loaded.decode_scale, 6) if downscaled else None,
                "exif_orientation": loaded.orientation if loaded.orientation != 1 else None,
                "synthetic_marker": ",".join(markers) or None,
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
        source_filenames: Mapping[str, str | None] | None = None,
        role: SiteRole | str = SiteRole.EXAMINED,
        visual_flag: VisualFlag | None = None,
        diascopy: Diascopy | str = Diascopy.NOT_DONE,
        px_per_mm: float | None = None,
        device_id: str | None = None,
        firmware_version: str | None = None,
        synthetic: bool = False,
    ) -> SiteInput:
        """WL + AF (+ optional dark frame) for one oral site.

        ``filenames`` maps ``"wl"``, ``"af"`` and ``"dark"`` to the file names recorded in the
        metadata; ``source_filenames`` to the names as uploaded (checked for the SYNTHETIC
        prefix only, never stored). The dark frame is optional: without it, ambient light cannot
        be subtracted and QC reports that. ``px_per_mm`` is the scale of the uploaded white-light
        image; the returned site carries it for the decoded (possibly downscaled) pixels.
        """
        oral_site = OralSite(site)
        if px_per_mm is not None and not (math.isfinite(px_per_mm) and px_per_mm > 0):
            raise ValueError("px_per_mm must be a positive number when given.")
        names = dict(filenames or {})
        sources = dict(source_filenames or {})
        common = {
            "device_id": device_id,
            "firmware_version": firmware_version,
            "synthetic": synthetic,
        }
        wl_cap = self.image_capture(
            wl,
            CaptureKind.WHITE_LIGHT,
            oral_site,
            file_name=names.get("wl"),
            source_file_name=sources.get("wl"),
            **common,
        )
        af_cap = self.image_capture(
            af,
            CaptureKind.AUTOFLUORESCENCE,
            oral_site,
            file_name=names.get("af"),
            source_file_name=sources.get("af"),
            **common,
        )
        dark_cap = None
        if dark:
            dark_cap = self.image_capture(
                dark,
                CaptureKind.DARK_FRAME,
                oral_site,
                file_name=names.get("dark"),
                source_file_name=sources.get("dark"),
                **common,
            )
        return SiteInput(
            site=oral_site,
            wl=wl_cap,
            af=af_cap,
            dark=dark_cap,
            role=SiteRole(role),
            visual_flag=visual_flag or VisualFlag(),
            diascopy=Diascopy(diascopy),
            px_per_mm=decoded_px_per_mm(px_per_mm, wl_cap),
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


def decoded_px_per_mm(px_per_mm: float | None, capture: ImageCapture) -> float | None:
    """Convert a scale given for the uploaded image to the decoded image's pixels.

    ``capture.meta.width_px`` is the capture's own (oriented) width; the array may be smaller
    when it was downscaled during decoding.
    """
    if px_per_mm is None:
        return None
    original = capture.meta.width_px
    decoded = int(capture.image.shape[1])
    if not original or original == decoded:
        return px_per_mm
    return px_per_mm * decoded / float(original)
