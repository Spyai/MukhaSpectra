"""Runs the pipeline and stores immutable, versioned results.

Re-analysis never overwrites: each run writes ``result_version = n + 1`` with
``supersedes_result_id`` pointing at version n, and its artifacts under
``screenings/{id}/results/v{n+1}/``.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.errors import AnalysisFailedError, ApiError, ConflictError
from app.models import AnalysisResult, Screening, ScreeningStatus, new_id, utcnow
from app.services.audit import AuditAction, record_event
from app.services.inputs import CaptureStore, StoredInputsError
from app.services.pipeline_bridge import PipelineBridge
from app.services.records import latest_result_row
from app.settings import Settings
from app.storage import Storage, join_key
from mukha_pipeline.config import PipelineConfig
from mukha_pipeline.schema.models import ModelComponent, ModelInfo, ModelStatus, ScreeningResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mukha_pipeline.types import ScreeningInput

logger = logging.getLogger("app.analysis")

WL_UNAVAILABLE_NOTE = "White-light classifier: not available — not used in triage"
SPECTRAL_PLACEHOLDER_NOTE = "Spectral classifier: placeholder heuristic, not validated"
RULES_NOTE = "Rule-based triage: baseline heuristic, unvalidated"
SEGMENTER_NOTE = "Classical colour-rule segmentation: baseline, not validated"
STORED_INPUTS_UNREADABLE = (
    "The stored captures for this screening could not be read, so it was not re-analysed."
)


def results_prefix(screening_id: str, version: int) -> str:
    return join_key("screenings", screening_id, "results", f"v{version}") + "/"


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        bridge: PipelineBridge,
        capture_store: CaptureStore,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.bridge = bridge
        self.capture_store = capture_store
        self._classifier: Any = None
        self._classifier_loaded = False
        self._classifier_lock = threading.Lock()
        self._screening_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # -- models --------------------------------------------------------------------------------

    def classifier(self) -> Any:
        """The WL lesion classifier, built once from ``settings.wl_model_dir``.

        Returns None when the WL module cannot be loaded; the orchestrator then falls back to
        its own "model unavailable" classifier and the result says so.
        """
        if not self._classifier_loaded:
            with self._classifier_lock:
                if not self._classifier_loaded:
                    try:
                        cfg = PipelineConfig.default().analysis.data.get("wl_classifier", {})
                        self._classifier = self.bridge.load_classifier(
                            self.settings.wl_model_dir, cfg
                        )
                    except Exception:  # noqa: BLE001 - degrade to "model unavailable"
                        logger.warning("WL classifier could not be loaded", exc_info=True)
                        self._classifier = None
                    self._classifier_loaded = True
        return self._classifier

    def model_infos(self) -> list[ModelInfo]:
        """Honest status of every model component (none is clinically validated)."""
        config = PipelineConfig.default()
        seg_cfg = config.analysis.data.get("segmentation", {})
        infos = [
            ModelInfo(
                component=ModelComponent.SEGMENTER,
                name=str(seg_cfg.get("method", "classical_hsv_lab")),
                version=str(seg_cfg.get("version", "1.0.0")),
                status=ModelStatus.BASELINE_RULES,
                notes=SEGMENTER_NOTE,
            )
        ]

        wl_info = getattr(self.classifier(), "info", None)
        if not isinstance(wl_info, ModelInfo):
            wl_info = ModelInfo(
                component=ModelComponent.WL_LESION_CLASSIFIER,
                name="wl-lesion-classifier",
                version="none",
                status=ModelStatus.MODEL_UNAVAILABLE,
            )
        if wl_info.status == ModelStatus.MODEL_UNAVAILABLE and not wl_info.notes:
            wl_info = wl_info.model_copy(update={"notes": WL_UNAVAILABLE_NOTE})
        infos.append(wl_info)

        try:
            spectral_info = self.bridge.spectral_classifier_info()
        except Exception:  # noqa: BLE001 - module may not be installed yet
            spectral_info = None
        if not isinstance(spectral_info, ModelInfo):
            method = config.analysis.data.get("spectral", {}).get("classifier", {})
            spectral_info = ModelInfo(
                component=ModelComponent.SPECTRAL_CLASSIFIER,
                name=str(method.get("method", "rules_v1")),
                version=config.analysis.ref.version,
                status=ModelStatus.PLACEHOLDER,
            )
        if not spectral_info.notes:
            spectral_info = spectral_info.model_copy(update={"notes": SPECTRAL_PLACEHOLDER_NOTE})
        infos.append(spectral_info)

        triage = config.triage.ref
        infos.append(
            ModelInfo(
                component=ModelComponent.FUSION,
                name=triage.name,
                version=triage.version,
                status=ModelStatus.BASELINE_RULES,
                sha256=triage.sha256,
                notes=RULES_NOTE,
            )
        )
        return infos

    # -- running -------------------------------------------------------------------------------

    def _lock_for(self, screening_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._screening_locks.setdefault(screening_id, threading.Lock())

    def run(
        self, session: Session, screening: Screening, inp: ScreeningInput, *, reanalysis: bool
    ) -> tuple[AnalysisResult, ScreeningResult]:
        """Analyse ``inp`` as the next result version of ``screening`` and store it."""
        with self._lock_for(screening.id):
            previous = latest_result_row(session, screening.id)
            version = previous.version + 1 if previous is not None else 1
            supersedes = previous.id if previous is not None else None
            inp.result_version = version
            inp.supersedes_result_id = supersedes
            inp.result_id = new_id()
            prefix = results_prefix(screening.id, version)

            try:
                output = self.bridge.analyze(inp, classifier=self.classifier())
                result = self._normalise(output.result, screening.id, version, supersedes)
                self.storage.delete_prefix(prefix)  # leftovers of an interrupted attempt
                for key, data in (output.artifacts or {}).items():
                    self.storage.put(prefix + key.lstrip("/"), data)
                    # (put validates the full key, so a malformed artifact key fails here)
            except Exception as exc:
                logger.exception("Analysis of screening %s failed", screening.id)
                self._discard(prefix)
                session.rollback()
                screening.status = ScreeningStatus.ANALYSIS_FAILED
                record_event(
                    session,
                    AuditAction.ANALYSIS_FAILED,
                    "screening",
                    screening.id,
                    {"result_version": version, "error_type": type(exc).__name__},
                )
                session.commit()
                raise AnalysisFailedError(screening.id) from exc

            result_json = result.model_dump(mode="json")
            row = AnalysisResult(
                id=result.result_id,
                screening_id=screening.id,
                version=version,
                created_at=result.created_at,
                overall_risk_level=result.triage.overall_risk_level.value,
                pipeline_version=result.versions.pipeline,
                rule_config_version=result.versions.rule_config.version,
                rule_config_sha256=result.versions.rule_config.sha256,
                synthetic=result.synthetic,
                result=result_json,
            )
            session.add(row)
            screening.status = ScreeningStatus.ANALYSED
            screening.synthetic = bool(screening.synthetic or result.synthetic)
            record_event(
                session,
                AuditAction.REANALYSIS_COMPLETED if reanalysis else AuditAction.ANALYSIS_COMPLETED,
                "screening",
                screening.id,
                {
                    "result_id": row.id,
                    "result_version": version,
                    "supersedes_result_id": supersedes,
                    "overall_risk_level": row.overall_risk_level,
                    "pipeline_version": row.pipeline_version,
                    "rule_config_sha256": row.rule_config_sha256,
                    "artifacts": len(output.artifacts or {}),
                },
            )
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ConflictError(
                    "Another analysis of this screening finished at the same time; "
                    "reload the screening to see the latest version."
                ) from exc
            return row, result

    def reanalyze(
        self, session: Session, screening: Screening
    ) -> tuple[AnalysisResult, ScreeningResult]:
        """Rebuild the inputs from storage and analyse them as a new version."""
        try:
            inp = self.capture_store.rebuild_input(session, screening)
        except StoredInputsError as exc:
            logger.exception("Stored inputs of screening %s are unreadable", screening.id)
            raise ApiError(STORED_INPUTS_UNREADABLE, status_code=500) from exc
        return self.run(session, screening, inp, reanalysis=True)

    # -- helpers -------------------------------------------------------------------------------

    def _discard(self, prefix: str) -> None:
        try:
            self.storage.delete_prefix(prefix)
        except Exception:  # noqa: BLE001 - best effort cleanup
            logger.warning("Could not clean up %s", prefix, exc_info=True)

    @staticmethod
    def _normalise(
        result: Any, screening_id: str, version: int, supersedes: str | None
    ) -> ScreeningResult:
        """Validate the pipeline output and pin the fields the backend owns."""
        if not isinstance(result, ScreeningResult):
            result = ScreeningResult.model_validate(result)
        expected = {
            "screening_id": screening_id,
            "result_version": version,
            "supersedes_result_id": supersedes,
        }
        updates: dict[str, Any] = {
            k: v for k, v in expected.items() if getattr(result, k) != v
        }
        if updates:
            logger.warning("Pipeline result did not carry the expected %s", sorted(updates))
        updates["created_at"] = utcnow()
        result = result.model_copy(update=updates)
        # Round-trip through validation so nothing invalid is ever stored.
        return ScreeningResult.model_validate(result.model_dump(mode="json"))
