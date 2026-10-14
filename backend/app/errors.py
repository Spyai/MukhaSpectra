"""API error types and handlers. Every error body is ``{"detail": "<plain message>"}``."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("app.errors")

ANALYSIS_FAILED_MESSAGE = "Analysis failed; the captures were stored and can be re-analysed."
INTERNAL_ERROR_MESSAGE = "Internal server error."


class ApiError(Exception):
    status_code: int = 500

    def __init__(
        self, detail: str, *, status_code: int | None = None, extra: dict[str, Any] | None = None
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code
        self.extra = extra or {}


class NotFoundError(ApiError):
    status_code = 404


class BadRequestError(ApiError):
    status_code = 400


class InvalidInputError(ApiError):
    """Request content is well-formed HTTP but cannot be accepted (manifest, images, spectra)."""

    status_code = 422


class PayloadTooLargeError(ApiError):
    status_code = 413


class ConflictError(ApiError):
    status_code = 409


class ServiceUnavailableError(ApiError):
    status_code = 503


class AnalysisFailedError(ApiError):
    status_code = 500

    def __init__(self, screening_id: str) -> None:
        super().__init__(ANALYSIS_FAILED_MESSAGE, extra={"screening_id": screening_id})


class InputDecodeError(ValueError):
    """Raised by input adapters when a capture cannot be decoded. The message is shown to the
    health worker as-is, so it must be plain language. (The pipeline's ``ImageDecodeError`` and
    ``SpectralParseError`` are handled the same way.)"""


def _api_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    if exc.status_code >= 500:
        logger.error("API error %s: %s", exc.status_code, exc.detail)
    return JSONResponse({"detail": exc.detail, **exc.extra}, status_code=exc.status_code)


def _unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error", exc_info=exc)
    return JSONResponse({"detail": INTERNAL_ERROR_MESSAGE}, status_code=500)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(Exception, _unexpected_error_handler)
