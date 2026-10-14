"""Screenings: upload, list, results, re-analysis, artifacts, reports and the site map."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, Request, Response
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.deps import ServicesDep, SessionDep
from app.errors import BadRequestError, InvalidInputError, NotFoundError, PayloadTooLargeError
from app.models import as_utc
from app.schemas import (
    ErrorResponse,
    ScreeningCreated,
    ScreeningDetail,
    ScreeningSummary,
)
from app.services.audit import AuditAction, record_event
from app.services.inputs import FilePart
from app.services.records import (
    get_result_row,
    get_screening,
    list_summaries,
    result_rows,
    version_summary,
)
from app.services.reports import ArtifactReader, report_file_name
from app.storage import ObjectNotFoundError
from mukha_pipeline.schema.models import ScreeningResult

router = APIRouter(tags=["screenings"])

MAX_FORM_FILES = 64
MAX_FORM_FIELDS = 64
MAX_MANIFEST_BYTES = 1024 * 1024

ARTIFACT_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".csv": "text/csv; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".pdf": "application/pdf",
    ".webp": "image/webp",
}
# Stored artifacts and the site map are data, never active content.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; img-src data:; style-src 'unsafe-inline'; font-src data:; sandbox"
)
NO_STORE = "no-store, private"

ERRORS_COMMON = {404: {"model": ErrorResponse, "description": "Unknown screening or version"}}
ERRORS_UPLOAD = {
    413: {"model": ErrorResponse, "description": "A file or the request is too large"},
    422: {"model": ErrorResponse, "description": "Invalid manifest, missing consent or file"},
    500: {"model": ErrorResponse, "description": "Analysis failed; captures were stored"},
}

UPLOAD_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["manifest"],
                    "properties": {
                        "manifest": {
                            "type": "string",
                            "contentMediaType": "application/json",
                            "contentSchema": {"$ref": "#/components/schemas/UploadManifest"},
                            "description": "UploadManifest as JSON. Each wl / af / dark / "
                            "spectral_file value names another multipart file field.",
                        }
                    },
                    "additionalProperties": {"type": "string", "format": "binary"},
                }
            }
        },
    }
}

VersionQuery = Annotated[
    int | None, Query(ge=1, description="Result version (default: the latest).")
]


def _check_request_size(request: Request, limit: int) -> None:
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit:
        raise PayloadTooLargeError(
            f"The upload is larger than the {limit // (1024 * 1024)} MB request limit."
        )


@router.post(
    "/screenings",
    status_code=201,
    response_model=ScreeningCreated,
    responses=ERRORS_UPLOAD,
    summary="Upload a screening (multipart: manifest + files) and analyse it",
    openapi_extra=UPLOAD_REQUEST_BODY,
)
async def create_screening(
    request: Request, services: ServicesDep, session: SessionDep
) -> ScreeningCreated:
    settings = services.settings
    _check_request_size(request, settings.max_request_bytes)
    form = await request.form(
        max_files=MAX_FORM_FILES, max_fields=MAX_FORM_FIELDS, max_part_size=MAX_MANIFEST_BYTES
    )
    try:
        raw = form.get("manifest")
        if isinstance(raw, UploadFile):
            raw = await raw.read()
        manifest = services.ingest.parse_manifest(raw)

        limit = settings.max_upload_bytes
        files: dict[str, FilePart] = {}
        for name, location in manifest.file_fields():
            if name in files:
                continue
            value = form.get(name)
            if value is None:
                raise InvalidInputError(
                    f"The manifest refers to file field '{name}' ({location}), "
                    "but no file with that name was uploaded."
                )
            if not isinstance(value, UploadFile):
                raise InvalidInputError(f"Form field '{name}' ({location}) must be a file.")
            if value.size is not None and value.size > limit:
                raise PayloadTooLargeError(
                    f"File '{name}' is larger than the {settings.max_upload_mb} MB limit."
                )
            data = await value.read()
            if len(data) > limit:
                raise PayloadTooLargeError(
                    f"File '{name}' is larger than the {settings.max_upload_mb} MB limit."
                )
            if not data:
                raise InvalidInputError(f"The file in field '{name}' ({location}) is empty.")
            files[name] = FilePart(
                field=name, filename=value.filename, content_type=value.content_type, data=data
            )
    finally:
        await form.close()

    screening_id, row, result = await run_in_threadpool(
        services.ingest.create_screening, session, manifest, files
    )
    return ScreeningCreated(screening_id=screening_id, result_version=row.version, result=result)


@router.get(
    "/screenings",
    response_model=list[ScreeningSummary],
    summary="List screenings, newest first (total count in X-Total-Count)",
)
def list_screenings(
    response: Response,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ScreeningSummary]:
    items, total = list_summaries(session, limit=limit, offset=offset)
    response.headers["X-Total-Count"] = str(total)
    return items


@router.get(
    "/screenings/{screening_id}",
    response_model=ScreeningDetail,
    responses=ERRORS_COMMON,
    summary="Latest result and the list of result versions",
)
def get_screening_detail(screening_id: str, session: SessionDep) -> ScreeningDetail:
    screening = get_screening(session, screening_id)
    rows = result_rows(session, screening.id)
    latest = ScreeningResult.model_validate(rows[-1].result) if rows else None
    detail = ScreeningDetail(
        screening_id=screening.id,
        created_at=as_utc(screening.created_at),  # type: ignore[arg-type]
        status=screening.status,
        latest=latest,
        versions=[version_summary(row) for row in rows],
    )
    record_event(
        session,
        AuditAction.RESULT_VIEWED,
        "screening",
        screening.id,
        {"result_version": rows[-1].version if rows else None},
    )
    session.commit()
    return detail


@router.get(
    "/screenings/{screening_id}/results/{version}",
    response_model=ScreeningResult,
    responses=ERRORS_COMMON,
    summary="One specific result version",
)
def get_result_version(
    screening_id: str, version: Annotated[int, Path(ge=1)], session: SessionDep
) -> ScreeningResult:
    screening = get_screening(session, screening_id)
    row = get_result_row(session, screening.id, version)
    result = ScreeningResult.model_validate(row.result)
    record_event(
        session, AuditAction.RESULT_VIEWED, "screening", screening.id, {"result_version": version}
    )
    session.commit()
    return result


@router.post(
    "/screenings/{screening_id}/reanalyze",
    status_code=201,
    response_model=ScreeningCreated,
    responses={**ERRORS_COMMON, 500: ERRORS_UPLOAD[500]},
    summary="Re-analyse stored captures as a new result version",
)
def reanalyze_screening(
    screening_id: str, services: ServicesDep, session: SessionDep
) -> ScreeningCreated:
    screening = get_screening(session, screening_id)
    row, result = services.analysis.reanalyze(session, screening)
    return ScreeningCreated(screening_id=screening.id, result_version=row.version, result=result)


@router.get(
    "/screenings/{screening_id}/artifacts/{key:path}",
    responses={
        **ERRORS_COMMON,
        400: {"model": ErrorResponse, "description": "Invalid artifact key"},
        200: {"content": {"image/png": {}, "application/json": {}, "image/svg+xml": {}}},
    },
    response_class=Response,
    summary="A stored analysis artifact (PNG / JPEG / JSON / SVG)",
)
def get_artifact(
    screening_id: str,
    key: str,
    services: ServicesDep,
    session: SessionDep,
    version: VersionQuery = None,
) -> Response:
    screening = get_screening(session, screening_id)
    row = get_result_row(session, screening.id, version)
    reader = ArtifactReader(services.storage, screening.id, row.version)
    try:
        storage_key = reader.key_for(key)
    except ValueError as exc:
        raise BadRequestError("Invalid artifact key.") from exc
    try:
        data = services.storage.get(storage_key)
    except ObjectNotFoundError as exc:
        raise NotFoundError("Artifact not found.") from exc
    suffix = "." + key.rsplit(".", 1)[-1].lower() if "." in key.rsplit("/", 1)[-1] else ""
    return Response(
        content=data,
        media_type=ARTIFACT_MEDIA_TYPES.get(suffix, "application/octet-stream"),
        headers={"Content-Security-Policy": CONTENT_SECURITY_POLICY, "Cache-Control": "private"},
    )


@router.get(
    "/screenings/{screening_id}/report.pdf",
    response_class=Response,
    responses={**ERRORS_COMMON, 200: {"content": {"application/pdf": {}}}},
    summary="Clinician referral report (PDF, stored once per result version)",
)
def get_report_pdf(
    screening_id: str, services: ServicesDep, session: SessionDep, version: VersionQuery = None
) -> Response:
    screening = get_screening(session, screening_id)
    row = get_result_row(session, screening.id, version)
    pdf = services.reports.clinician_pdf(session, screening, row)
    filename = report_file_name(screening.id, row.version)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": NO_STORE,
        },
    )


@router.get(
    "/screenings/{screening_id}/report.html",
    response_class=Response,
    responses={**ERRORS_COMMON, 200: {"content": {"text/html": {}}}},
    summary="Clinician report as HTML (preview)",
)
def get_report_html(
    screening_id: str, services: ServicesDep, session: SessionDep, version: VersionQuery = None
) -> Response:
    screening = get_screening(session, screening_id)
    row = get_result_row(session, screening.id, version)
    html = services.reports.clinician_html(session, screening, row)
    return Response(
        content=html,
        media_type="text/html; charset=utf-8",
        headers={"Content-Security-Policy": CONTENT_SECURITY_POLICY, "Cache-Control": NO_STORE},
    )


@router.get(
    "/screenings/{screening_id}/sitemap.svg",
    response_class=Response,
    responses={**ERRORS_COMMON, 200: {"content": {"image/svg+xml": {}}}},
    summary="Oral site map coloured by risk level",
)
def get_site_map(
    screening_id: str,
    services: ServicesDep,
    session: SessionDep,
    version: VersionQuery = None,
    width: Annotated[int, Query(ge=100, le=2000)] = 400,
    legend: Annotated[bool, Query()] = True,
) -> Response:
    screening = get_screening(session, screening_id)
    row = get_result_row(session, screening.id, version)
    svg = services.reports.site_map_svg(row, width=width, show_legend=legend)
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Content-Security-Policy": CONTENT_SECURITY_POLICY, "Cache-Control": "private"},
    )
