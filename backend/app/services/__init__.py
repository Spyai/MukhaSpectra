"""Application services. ``build_services`` wires them together once per app instance."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from app.services.analysis import AnalysisService
from app.services.demo import DemoService
from app.services.ingest import IngestService
from app.services.inputs import CaptureStore
from app.services.pipeline_bridge import DEFAULT_BRIDGE, PipelineBridge
from app.services.reports import AppReportsRenderer, ReportRenderer, ReportService
from app.settings import Settings
from app.storage import LocalStorage, Storage


@dataclass
class Services:
    settings: Settings
    session_factory: sessionmaker[Session]
    storage: Storage
    bridge: PipelineBridge
    capture_store: CaptureStore
    analysis: AnalysisService
    ingest: IngestService
    demo: DemoService
    reports: ReportService


def build_services(
    settings: Settings,
    session_factory: sessionmaker[Session],
    *,
    storage: Storage | None = None,
    bridge: PipelineBridge | None = None,
    report_renderer: ReportRenderer | None = None,
) -> Services:
    storage = storage or LocalStorage(settings.storage_dir)
    bridge = bridge or DEFAULT_BRIDGE
    capture_store = CaptureStore(storage, bridge)
    analysis = AnalysisService(settings, storage, bridge, capture_store)
    return Services(
        settings=settings,
        session_factory=session_factory,
        storage=storage,
        bridge=bridge,
        capture_store=capture_store,
        analysis=analysis,
        ingest=IngestService(storage, bridge, capture_store, analysis),
        demo=DemoService(storage, bridge, capture_store, analysis),
        reports=ReportService(settings, storage, report_renderer or AppReportsRenderer()),
    )


__all__ = ["Services", "build_services"]
