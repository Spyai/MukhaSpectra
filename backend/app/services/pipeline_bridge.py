"""The single seam between the API and the analysis library.

Every call into ``mukha_pipeline`` modules other than schema/sites/config goes through a
``PipelineBridge``. The default implementations import lazily, so the API starts (and its tests
run) even while parts of the pipeline are missing, and tests can swap in fakes with
``PipelineBridge(analyze=fake, ...)`` or ``DEFAULT_BRIDGE.replace(analyze=fake)``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.errors import InputDecodeError
from mukha_pipeline.schema.models import SourceAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mukha_pipeline.types import AnalysisOutput, ScreeningInput


def default_analyze(inp: ScreeningInput, *, classifier: Any = None) -> AnalysisOutput:
    from app import __version__
    from mukha_pipeline.orchestrator import analyze_screening

    return analyze_screening(inp, classifier=classifier, software_version=__version__)


def default_upload_adapter() -> Any:
    from mukha_pipeline.capture.upload import UploadAdapter

    return UploadAdapter()


def default_parse_spectral(
    data: bytes,
    filename: str,
    *,
    source_adapter: SourceAdapter,
    device_id: str | None,
    synthetic: bool,
    now: datetime | None,
) -> Any:
    from mukha_pipeline.io.spectral import parse_spectral_file

    return parse_spectral_file(
        data,
        filename,
        source_adapter=source_adapter,
        device_id=device_id,
        synthetic=synthetic,
        now=now,
    )


def default_load_image(data: bytes, filename: str | None) -> np.ndarray:
    """Decode stored capture bytes to float32 RGB [0, 1] (EXIF orientation applied)."""
    from mukha_pipeline.io.images import load_image_bytes

    return load_image_bytes(data, filename).image


def default_make_demo_patient(kind: str) -> ScreeningInput:
    from mukha_pipeline.synthetic import make_demo_patient

    return make_demo_patient(kind, seed=0)  # type: ignore[arg-type]


def default_load_classifier(model_dir: Path | None, cfg: dict[str, Any]) -> Any:
    from mukha_pipeline.wl import load_lesion_classifier

    return load_lesion_classifier(model_dir, cfg)


def default_spectral_classifier_info() -> Any:
    from mukha_pipeline.config import PipelineConfig
    from mukha_pipeline.spectral import RuleBasedSpectralClassifier

    cfg = PipelineConfig.default().analysis.data.get("spectral", {})
    return RuleBasedSpectralClassifier(cfg).info


def default_input_error_types() -> tuple[type[BaseException], ...]:
    """Exceptions that mean "the uploaded capture is unreadable" (-> HTTP 422)."""
    types: list[type[BaseException]] = [InputDecodeError]
    try:
        from mukha_pipeline.io.images import ImageDecodeError

        types.append(ImageDecodeError)
    except ImportError:  # pragma: no cover - depends on pipeline build state
        pass
    try:
        from mukha_pipeline.io.spectral import SpectralParseError

        types.append(SpectralParseError)
    except ImportError:  # pragma: no cover - depends on pipeline build state
        pass
    return tuple(types)


@dataclasses.dataclass(frozen=True)
class PipelineBridge:
    analyze: Callable[..., Any] = default_analyze
    upload_adapter_factory: Callable[[], Any] = default_upload_adapter
    parse_spectral: Callable[..., Any] = default_parse_spectral
    load_image: Callable[[bytes, str | None], np.ndarray] = default_load_image
    make_demo_patient: Callable[[str], Any] = default_make_demo_patient
    load_classifier: Callable[[Path | None, dict[str, Any]], Any] = default_load_classifier
    spectral_classifier_info: Callable[[], Any] = default_spectral_classifier_info
    input_error_types: Callable[[], tuple[type[BaseException], ...]] = default_input_error_types

    def replace(self, **changes: Any) -> PipelineBridge:
        return dataclasses.replace(self, **changes)


DEFAULT_BRIDGE = PipelineBridge()
