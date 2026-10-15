"""Phase 1 access control and request-size guards.

Authentication with user accounts and roles is Phase 5. Until then the API offers one shared key:

* ``MUKHA_API_KEY`` set: every ``/api/v1`` route except ``/health`` needs the ``X-API-Key``
  header (compared in constant time), otherwise 401. A required custom header also forces a
  CORS preflight, so other web pages cannot send simple cross-origin POSTs.
* unset: there is no authentication at all. The server logs :data:`NO_API_KEY_WARNING` at
  startup, ``/version`` and the OpenAPI text say so, and the default docker-compose publishes
  ports on 127.0.0.1 only.

:class:`BodySizeLimitMiddleware` enforces the request byte limit while the body streams in, so a
chunked request without ``Content-Length`` cannot bypass it.
"""

from __future__ import annotations

import hmac
import json
from typing import Annotated

from fastapi import Request, Security
from fastapi.security import APIKeyHeader
from pydantic import SecretStr
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import PayloadTooLargeError, UnauthorizedError
from app.settings import Settings

API_KEY_HEADER = "X-API-Key"
API_KEY_REQUIRED = "A valid API key is required (X-API-Key header)."
NO_API_KEY_WARNING = (
    "No API key configured: do not expose this server beyond localhost "
    "(Phase 5 adds authentication). Anyone who can reach it can read and change every screening."
)
ACCESS_NOTICE_NONE = (
    "Authentication is not yet implemented: no API key is configured, so anyone who can reach "
    "this server can read and change every screening, including patient names in reports. Keep "
    "it on localhost; set MUKHA_API_KEY before exposing it to a network. Per-user accounts and "
    "roles are planned for Phase 5."
)
ACCESS_NOTICE_KEY = (
    "Access needs the shared API key in the X-API-Key header. Per-user accounts, roles and "
    "per-user audit are not yet implemented (Phase 5)."
)

api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="Shared API key (only enforced when the server sets MUKHA_API_KEY).",
)


def authentication_mode(settings: Settings) -> str:
    return "api_key" if settings.api_key is not None else "none"


def access_notice(settings: Settings) -> str:
    return ACCESS_NOTICE_KEY if settings.api_key is not None else ACCESS_NOTICE_NONE


def require_api_key(
    request: Request, key: Annotated[str | None, Security(api_key_scheme)] = None
) -> None:
    """FastAPI dependency: 401 unless the configured key was sent (no-op without a key)."""
    settings: Settings = request.app.state.settings
    expected = settings.api_key
    if expected is None:
        return
    secret = expected.get_secret_value() if isinstance(expected, SecretStr) else str(expected)
    if key is None or not hmac.compare_digest(key.encode("utf-8"), secret.encode("utf-8")):
        raise UnauthorizedError(API_KEY_REQUIRED)


def request_too_large_message(limit_bytes: int) -> str:
    return f"The upload is larger than the {limit_bytes // (1024 * 1024)} MB request limit."


class BodySizeLimitMiddleware:
    """Reject request bodies above ``max_bytes`` with 413, counting bytes as they arrive.

    A declared ``Content-Length`` above the limit is refused before anything is read. Otherwise
    (also for chunked bodies) ``receive`` raises :class:`PayloadTooLargeError` as soon as the
    count passes the limit; the API error handler turns that into a JSON 413.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        message = request_too_large_message(self.max_bytes)
        for name, value in scope.get("headers") or []:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > self.max_bytes:
                    await _send_json(send, 413, {"detail": message})
                    return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > self.max_bytes:
                    raise PayloadTooLargeError(message)
            return msg

        await self.app(scope, limited_receive, send)


async def _send_json(send: Send, status: int, body: dict[str, str]) -> None:
    payload = json.dumps(body).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})
