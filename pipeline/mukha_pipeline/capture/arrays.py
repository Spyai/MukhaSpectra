"""ArraySource: captures from in-memory arrays (synthetic generator, mock devices, tests).

Gives non-file sources the same metadata discipline as uploads. With the SYNTHETIC adapter every
capture is marked synthetic and this cannot be switched off, so generated pixels can never be
mistaken for a patient's.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping

import numpy as np

from mukha_pipeline.capture.base import ExtraValue, image_metadata
from mukha_pipeline.io.images import to_float_rgb
from mukha_pipeline.schema.models import CaptureKind, Diascopy, SiteRole, SourceAdapter, VisualFlag
from mukha_pipeline.sites import OralSite
from mukha_pipeline.types import ImageCapture, SiteInput


class ArraySource:
    def __init__(
        self, adapter: SourceAdapter = SourceAdapter.SYNTHETIC, *, synthetic: bool = False
    ) -> None:
        self.adapter = SourceAdapter(adapter)
        self.synthetic = synthetic or self.adapter == SourceAdapter.SYNTHETIC

    def image_capture(
        self,
        image: np.ndarray,
        kind: CaptureKind,
        site: OralSite | str,
        *,
        device_id: str | None = None,
        firmware_version: str | None = None,
        exposure_time_ms: float | None = None,
        iso: float | None = None,
        file_name: str | None = None,
        extra: Mapping[str, ExtraValue] | None = None,
    ) -> ImageCapture:
        img = to_float_rgb(image)
        meta = image_metadata(
            kind,
            site,
            capture_id=str(uuid.uuid4()),
            source_adapter=self.adapter,
            width_px=int(img.shape[1]),
            height_px=int(img.shape[0]),
            exposure_time_ms=exposure_time_ms,
            iso=iso,
            file_name=file_name,
            device_id=device_id,
            firmware_version=firmware_version,
            synthetic=self.synthetic,
            extra={"array_sha256": hashlib.sha256(img.tobytes()).hexdigest(), **(extra or {})},
        )
        return ImageCapture(image=img, meta=meta)

    def site_input(
        self,
        site: OralSite | str,
        wl: np.ndarray,
        af: np.ndarray,
        dark: np.ndarray | None = None,
        *,
        role: SiteRole | str = SiteRole.EXAMINED,
        visual_flag: VisualFlag | None = None,
        diascopy: Diascopy | str = Diascopy.NOT_DONE,
        px_per_mm: float | None = None,
        device_id: str | None = None,
        firmware_version: str | None = None,
        exposure_time_ms: Mapping[str, float] | None = None,
    ) -> SiteInput:
        """``exposure_time_ms`` maps "wl" / "af" / "dark" to the simulated exposure, if any."""
        oral_site = OralSite(site)
        exposures = dict(exposure_time_ms or {})
        common = {"device_id": device_id, "firmware_version": firmware_version}
        return SiteInput(
            site=oral_site,
            wl=self.image_capture(
                wl,
                CaptureKind.WHITE_LIGHT,
                oral_site,
                exposure_time_ms=exposures.get("wl"),
                **common,
            ),
            af=self.image_capture(
                af,
                CaptureKind.AUTOFLUORESCENCE,
                oral_site,
                exposure_time_ms=exposures.get("af"),
                **common,
            ),
            dark=(
                None
                if dark is None
                else self.image_capture(
                    dark,
                    CaptureKind.DARK_FRAME,
                    oral_site,
                    exposure_time_ms=exposures.get("dark"),
                    **common,
                )
            ),
            role=SiteRole(role),
            visual_flag=visual_flag or VisualFlag(),
            diascopy=Diascopy(diascopy),
            px_per_mm=px_per_mm,
        )


def SyntheticSource() -> ArraySource:  # noqa: N802 - reads like a class at call sites
    """ArraySource for the synthetic generator: always labelled synthetic."""
    return ArraySource(SourceAdapter.SYNTHETIC)
