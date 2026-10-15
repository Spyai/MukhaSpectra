"""Regression tests: access control, request limits, concurrency and deployment settings.

* security.0: optional shared API key (X-API-Key) on every route except /health; a loud startup
  warning without one; the default compose file publishes ports on 127.0.0.1 only.
* honesty.5 (API side): /version and the OpenAPI text say plainly when there is no
  authentication.
* security.3: byte limits are enforced while the body streams in (also for chunked requests),
  per file and per request; an oversized manifest file part and deeply nested JSON are rejected.
* security.6: a bounded number of concurrent analyses; extra requests get 503.
* correctness.3 / correctness.6: the Docker image can talk to PostgreSQL; nginx accepts the
  same request size as the API and answers 413 as JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest
import yaml
from api_fakes import default_files, default_manifest, png_bytes, upload
from pydantic import ValidationError
from sqlalchemy import func, select

from app.models import Capture, Screening
from app.security import API_KEY_HEADER, NO_API_KEY_WARNING
from app.settings import Settings
from mukha_pipeline.positioning import find_positioning_violations

REPO = Path(__file__).resolve().parents[2]
KEY = "camp-key-0123456789abcdef"


# --------------------------------------------------------------------------------------------
# security.0: shared API key
# --------------------------------------------------------------------------------------------


@pytest.fixture
def keyed(make_client):
    return make_client(api_key=KEY)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/v1/screenings"),
        ("get", "/api/v1/version"),
        ("get", "/api/v1/sites"),
        ("get", "/api/v1/schema/screening-result"),
        ("get", "/api/v1/screenings/x/report.html"),
        ("get", "/api/v1/screenings/x/report.pdf"),
        ("post", "/api/v1/screenings/x/reanalyze"),
        ("post", "/api/v1/demo/screenings/high"),
        ("delete", "/api/v1/screenings/x"),
        ("delete", "/api/v1/patients/P-1"),
    ],
)
def test_every_route_but_health_needs_the_key(keyed, method, path):
    res = getattr(keyed, method)(path)
    assert res.status_code == 401, res.text
    assert res.json() == {"detail": "A valid API key is required (X-API-Key header)."}
    wrong = getattr(keyed, method)(path, headers={API_KEY_HEADER: KEY + "x"})
    assert wrong.status_code == 401


def test_health_stays_open_and_the_key_unlocks_the_api(keyed):
    assert keyed.get("/api/v1/health").status_code == 200
    ok = keyed.get("/api/v1/screenings", headers={API_KEY_HEADER: KEY})
    assert ok.status_code == 200


def test_upload_without_key_is_refused_before_anything_is_stored(keyed):
    res = upload(keyed)
    assert res.status_code == 401
    services = keyed.app.state.services
    with services.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Screening)) == 0
    keyed.headers[API_KEY_HEADER] = KEY
    assert upload(keyed).status_code == 201


def test_cors_preflight_allows_the_key_header(keyed):
    res = keyed.options(
        "/api/v1/screenings",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "DELETE",
            "Access-Control-Request-Headers": "x-api-key",
        },
    )
    assert res.status_code == 200
    assert "x-api-key" in res.headers["access-control-allow-headers"].lower()
    assert "DELETE" in res.headers["access-control-allow-methods"]


def test_short_keys_are_rejected():
    with pytest.raises(ValidationError):
        Settings(api_key="short")
    assert Settings(api_key="").api_key is None  # empty means "not configured"


def test_missing_key_logs_a_prominent_warning(make_client, caplog):
    with caplog.at_level(logging.WARNING, logger="app"):
        make_client()
    assert any(NO_API_KEY_WARNING in r.getMessage() for r in caplog.records)
    assert "do not expose this server beyond localhost" in NO_API_KEY_WARNING


def test_no_warning_when_a_key_is_configured(make_client, caplog):
    with caplog.at_level(logging.WARNING, logger="app"):
        make_client(api_key=KEY)
    assert not any(NO_API_KEY_WARNING in r.getMessage() for r in caplog.records)


def test_trusted_hosts_can_be_restricted(make_client):
    client = make_client(allowed_hosts=["localhost", "127.0.0.1"])
    assert client.get("/api/v1/health", headers={"Host": "localhost:8000"}).status_code == 200
    assert client.get("/api/v1/health", headers={"Host": "rebind.evil.example"}).status_code == 400


def test_compose_publishes_ports_on_localhost_only():
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    for name in ("backend", "frontend"):
        for port in compose["services"][name].get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)
    # The key is passed through from the operator's environment, never written in the file.
    for name in ("backend", "frontend"):
        env = compose["services"][name].get("environment") or {}
        key = str(env.get("MUKHA_API_KEY") or "")
        assert key in ("", "${MUKHA_API_KEY:-}", "${MUKHA_API_KEY}"), (name, key)


# --------------------------------------------------------------------------------------------
# honesty.5: say so when there is no authentication
# --------------------------------------------------------------------------------------------


def test_version_states_that_authentication_is_not_implemented(client):
    body = client.get("/api/v1/version").json()
    assert body["authentication"] == "none"
    assert "not yet implemented" in body["access_notice"]
    assert not find_positioning_violations(body["access_notice"])


def test_version_describes_the_shared_key(keyed):
    body = keyed.get("/api/v1/version", headers={API_KEY_HEADER: KEY}).json()
    assert body["authentication"] == "api_key"
    assert "X-API-Key" in body["access_notice"]
    assert "not yet implemented" in body["access_notice"]  # per-user accounts are not


def test_openapi_says_authentication_is_not_implemented(client, keyed):
    text = client.get("/openapi.json").json()["info"]["description"]
    assert "Authentication is not yet implemented" in text
    keyed_spec = keyed.get("/openapi.json").json()
    assert "Authentication is not yet implemented" not in keyed_spec["info"]["description"]
    assert "X-API-Key" in keyed_spec["info"]["description"]
    schemes = keyed_spec["components"]["securitySchemes"]
    assert any(s.get("name") == "X-API-Key" for s in schemes.values())


# --------------------------------------------------------------------------------------------
# security.3: limits enforced while streaming
# --------------------------------------------------------------------------------------------

BOUNDARY = "mukhatestboundary"


def _multipart(parts: list[tuple[str, str | None, bytes]]) -> bytes:
    out = bytearray()
    for name, filename, data in parts:
        out += f"--{BOUNDARY}\r\n".encode()
        disposition = f'form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        out += f"Content-Disposition: {disposition}\r\n".encode()
        if filename is not None:
            out += b"Content-Type: application/octet-stream\r\n"
        out += b"\r\n" + data + b"\r\n"
    out += f"--{BOUNDARY}--\r\n".encode()
    return bytes(out)


def _chunked_post(client, body: bytes, chunk: int = 64 * 1024):
    def gen():
        for i in range(0, len(body), chunk):
            yield body[i : i + chunk]

    return client.post(
        "/api/v1/screenings",
        content=gen(),
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )


def test_chunked_request_over_the_total_limit_is_413(make_client):
    client = make_client(max_request_mb=1)
    body = _multipart([("junk", "junk.bin", b"0" * (3 * 1024 * 1024))])
    res = _chunked_post(client, body)
    assert res.status_code == 413, res.text
    assert "request limit" in res.json()["detail"]


def test_chunked_file_over_the_per_file_limit_is_413(make_client):
    client = make_client(max_upload_mb=1, max_request_mb=10)
    manifest = json.dumps(default_manifest()).encode()
    files = default_files()
    parts = [("manifest", None, manifest)]
    parts += [(name, filename, data) for name, (filename, data, _type) in files.items()]
    parts.append(("extra", "big.png", b"0" * (1024 * 1024 + 10)))  # not even referenced
    res = _chunked_post(client, _multipart(parts))
    assert res.status_code == 413, res.text
    assert "1 MB limit" in res.json()["detail"]


def test_streaming_stops_reading_once_the_limit_is_passed(make_client):
    """At the ASGI level: the app gives up after ~limit bytes of an endless chunked body."""
    client = make_client(max_request_mb=1)
    app = client.app
    pulled = 0
    head = f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="x"; filename="x"\r\n\r\n'

    async def receive():
        nonlocal pulled
        if pulled == 0:
            pulled += len(head)
            return {"type": "http.request", "body": head.encode(), "more_body": True}
        pulled += 65536
        return {"type": "http.request", "body": b"0" * 65536, "more_body": True}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/screenings",
        "raw_path": b"/api/v1/screenings",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", f"multipart/form-data; boundary={BOUNDARY}".encode()),
            (b"transfer-encoding", b"chunked"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "state": {},
    }
    asyncio.run(asyncio.wait_for(app(scope, receive, send), timeout=30))
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413
    assert pulled <= 1024 * 1024 + 2 * 65536


def test_manifest_file_part_is_capped(client):
    padded = json.dumps(default_manifest()) + " " * (2 * 1024 * 1024)
    files = {**default_files(), "manifest": ("manifest.json", padded.encode(), "application/json")}
    res = client.post("/api/v1/screenings", files=files)
    assert res.status_code == 413
    assert "manifest" in res.json()["detail"].lower()


def test_deeply_nested_manifest_is_422_not_500(client, services, db):
    nested = "[" * 100_000 + "]" * 100_000
    res = client.post("/api/v1/screenings", data={"manifest": nested}, files=default_files())
    assert res.status_code in (413, 422), res.text
    assert db.scalar(select(func.count()).select_from(Capture)) == 0


def test_duplicate_form_fields_are_rejected(client):
    body = _multipart(
        [
            ("manifest", None, json.dumps(default_manifest()).encode()),
            ("wl_bl", "a.png", png_bytes()),
            ("wl_bl", "b.png", png_bytes()),
        ]
    )
    res = client.post(
        "/api/v1/screenings",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    assert res.status_code == 422
    assert "more than once" in res.json()["detail"]


def test_malformed_multipart_is_400(client):
    res = client.post(
        "/api/v1/screenings",
        content=b"--nope\r\nthis is not multipart",
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    assert res.status_code in (400, 422)
    assert isinstance(res.json()["detail"], str)


def test_missing_boundary_is_400(client):
    res = client.post(
        "/api/v1/screenings", content=b"x", headers={"Content-Type": "multipart/form-data"}
    )
    assert res.status_code == 400


# --------------------------------------------------------------------------------------------
# security.6: bounded concurrent analyses
# --------------------------------------------------------------------------------------------


def test_extra_analyses_are_refused_with_503(make_client):
    client = make_client(max_concurrent_analyses=1)
    services = client.app.state.services
    sid = upload(client).json()["screening_id"]
    with services.analysis.analysis_slot():
        busy_upload = upload(client)
        busy_demo = client.post("/api/v1/demo/screenings/low")
        busy_reanalysis = client.post(f"/api/v1/screenings/{sid}/reanalyze")
    for res in (busy_upload, busy_demo, busy_reanalysis):
        assert res.status_code == 503, res.text
        assert res.json() == {"detail": "Server busy, try again shortly."}
        assert res.headers.get("retry-after")
    with services.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Screening)) == 1
    assert client.post(f"/api/v1/screenings/{sid}/reanalyze").status_code == 201


def test_concurrency_default():
    assert Settings().max_concurrent_analyses == 2
    assert Settings().enable_demo is True


# --------------------------------------------------------------------------------------------
# correctness.3 / correctness.6: deployment files
# --------------------------------------------------------------------------------------------


def test_docker_image_installs_the_postgres_driver():
    dockerfile = (REPO / "backend" / "Dockerfile").read_text(encoding="utf-8")
    assert "./backend[postgres]" in dockerfile


def test_nginx_body_limit_matches_the_api_and_413_is_json():
    conf = (REPO / "deploy" / "nginx.conf.template").read_text(encoding="utf-8")
    assert f"client_max_body_size {Settings().max_request_mb}m;" in conf
    assert "error_page 413" in conf
    assert "application/json" in conf and '"detail"' in conf
    assert "proxy_read_timeout" in conf and "proxy_send_timeout" in conf
    assert Settings().max_request_mb == 200


def test_413_from_the_api_is_json(make_client):
    client = make_client(max_request_mb=1)
    files = default_files()
    files["wl_bl"] = ("big.png", b"0" * (1024 * 1024 + 10), "image/png")
    res = upload(client, files=files)
    assert res.status_code == 413
    assert res.headers["content-type"].startswith("application/json")


def test_new_messages_follow_the_positioning_rules(services):
    from app.errors import SERVER_BUSY_MESSAGE
    from app.schemas import CONSENT_METHOD_REQUIRED
    from app.security import ACCESS_NOTICE_KEY, ACCESS_NOTICE_NONE, API_KEY_REQUIRED
    from app.services.erasure import ANALYSIS_RUNNING, PATIENT_NOT_FOUND
    from app.services.ingest import MANIFEST_NOT_FINITE, MANIFEST_TOO_DEEP

    messages = [
        NO_API_KEY_WARNING,
        ACCESS_NOTICE_NONE,
        ACCESS_NOTICE_KEY,
        API_KEY_REQUIRED,
        SERVER_BUSY_MESSAGE,
        CONSENT_METHOD_REQUIRED,
        MANIFEST_NOT_FINITE,
        MANIFEST_TOO_DEEP,
        ANALYSIS_RUNNING,
        PATIENT_NOT_FOUND,
        services.ingest._budget_message(150_000_000),
    ]
    for message in messages:
        assert not find_positioning_violations(message), message
        assert not any(w in message.lower() for w in ("accuracy", "sensitivity", "specificity"))
