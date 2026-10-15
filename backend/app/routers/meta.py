"""Service metadata: health, versions, the result JSON Schema and the oral site catalogue."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Response

import mukha_pipeline
from app import __version__
from app.deps import ServicesDep
from app.schemas import (
    ConfigInfo,
    HealthResponse,
    SiteCatalogueEntry,
    SiteShape,
    SitesResponse,
    VersionResponse,
)
from app.security import access_notice, authentication_mode
from mukha_pipeline import SCHEMA_VERSION
from mukha_pipeline.config import LoadedConfig, PipelineConfig
from mukha_pipeline.schema.export import SCHEMA_PATH
from mukha_pipeline.schema.models import DISCLAIMER, PROTOTYPE_BANNER
from mukha_pipeline.sites import SITE_MAP_VIEWBOX, SITES

SERVICE_NAME = "MukhaSpectra API"

public_router = APIRouter(tags=["system"])  # no API key needed
router = APIRouter(tags=["system"])


@public_router.get("/health", response_model=HealthResponse, summary="Liveness")
def health() -> HealthResponse:
    return HealthResponse(
        status="ok", service=SERVICE_NAME, version=__version__, notice=PROTOTYPE_BANNER
    )


def _config_info(config: LoadedConfig) -> ConfigInfo:
    status = config.data.get("status")
    return ConfigInfo(**config.ref.model_dump(), status=str(status) if status else None)


@router.get("/version", response_model=VersionResponse, summary="Software, config and models")
def version(services: ServicesDep) -> VersionResponse:
    config = PipelineConfig.default()
    return VersionResponse(
        service=SERVICE_NAME,
        api_version="v1",
        software=services.analysis.software_version,
        build=services.analysis.build_id,
        pipeline=mukha_pipeline.__version__,
        schema_version=SCHEMA_VERSION,
        configs={
            "qc": _config_info(config.qc),
            "analysis": _config_info(config.analysis),
            "triage": _config_info(config.triage),
        },
        models=services.analysis.model_infos(),
        prototype_banner=PROTOTYPE_BANNER,
        disclaimer=DISCLAIMER,
        authentication=authentication_mode(services.settings),  # type: ignore[arg-type]
        access_notice=access_notice(services.settings),
    )


@lru_cache(maxsize=1)
def _schema_bytes() -> bytes:
    return SCHEMA_PATH.read_bytes()


@router.get(
    "/schema/screening-result",
    summary="ScreeningResult v1 JSON Schema",
    response_class=Response,
    responses={200: {"content": {"application/schema+json": {}}}},
)
def screening_result_schema() -> Response:
    return Response(content=_schema_bytes(), media_type="application/schema+json")


@router.get("/sites", response_model=SitesResponse, summary="Oral site catalogue and map geometry")
def sites() -> SitesResponse:
    x0, y0, width, height = SITE_MAP_VIEWBOX
    return SitesResponse(
        viewBox=f"{x0} {y0} {width} {height}",
        width=width,
        height=height,
        orientation=(
            "Examiner's view, facing the patient: the patient's right side is drawn on the "
            "viewer's left."
        ),
        sites=[
            SiteCatalogueEntry(
                site=info.site,
                label=info.label,
                short_label=info.short_label,
                paired_site=info.paired_site,
                pairing=info.pairing,
                keratinized=info.keratinized,
                shape=SiteShape(
                    cx=info.shape.cx, cy=info.shape.cy, rx=info.shape.rx, ry=info.shape.ry
                ),
            )
            for info in SITES.values()
        ],
    )
