"""System endpoints: health, version, schema, sites, OpenAPI metadata, headers and CORS."""

from __future__ import annotations

import json

from fastapi import FastAPI

import mukha_pipeline
from app import __version__
from mukha_pipeline.config import PipelineConfig
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.export import SCHEMA_PATH
from mukha_pipeline.schema.models import DISCLAIMER, PROTOTYPE_BANNER
from mukha_pipeline.sites import SITES, OralSite

HONESTY_FORBIDDEN = ("accuracy", "sensitivity", "specificity", "auc")


def test_health(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body == {
        "status": "ok",
        "service": "MukhaSpectra API",
        "version": __version__,
        "notice": PROTOTYPE_BANNER,
    }


def test_version_reports_every_version_and_config_hash(client):
    res = client.get("/api/v1/version")
    assert res.status_code == 200
    body = res.json()
    config = PipelineConfig.default()
    # backend version + build id, exactly what every result records (honesty.4)
    assert body["software"] == f"{__version__}+{body['build']}"
    assert body["build"]
    assert body["pipeline"] == mukha_pipeline.__version__
    assert body["schema_version"] == mukha_pipeline.SCHEMA_VERSION
    assert body["configs"]["triage"]["sha256"] == config.triage.ref.sha256
    assert body["configs"]["qc"]["sha256"] == config.qc.ref.sha256
    assert body["configs"]["analysis"]["sha256"] == config.analysis.ref.sha256
    assert body["configs"]["triage"]["status"] == "baseline_heuristic_unvalidated"
    assert body["prototype_banner"] == PROTOTYPE_BANNER
    assert body["disclaimer"] == DISCLAIMER


def test_version_states_model_status_honestly(client):
    models = {m["component"]: m for m in client.get("/api/v1/version").json()["models"]}
    assert set(models) == {"segmenter", "wl_lesion_classifier", "spectral_classifier", "fusion"}
    assert all(m["validated"] is False for m in models.values())
    wl = models["wl_lesion_classifier"]
    assert wl["status"] == "model_unavailable"
    assert wl["notes"] == "White-light classifier: not available — not used in triage"
    assert models["spectral_classifier"]["status"] == "placeholder"
    assert "not validated" in models["spectral_classifier"]["notes"]
    assert models["fusion"]["status"] == "baseline_rules"
    assert "unvalidated" in models["fusion"]["notes"]


def test_version_text_has_no_performance_claims_or_forbidden_wording(client):
    text = client.get("/api/v1/version").text
    assert not find_positioning_violations(text)
    assert not any(word in text.lower() for word in HONESTY_FORBIDDEN)


def test_version_falls_back_when_pipeline_models_cannot_load(make_client, fake_pipeline):
    client = make_client()
    services = client.app.state.services

    def boom(*_args, **_kwargs):
        raise ImportError("not built yet")

    analysis = services.analysis
    analysis.bridge = analysis.bridge.replace(load_classifier=boom, spectral_classifier_info=boom)
    analysis._classifier_loaded = False
    models = {m["component"]: m for m in client.get("/api/v1/version").json()["models"]}
    assert models["wl_lesion_classifier"]["status"] == "model_unavailable"
    assert "not used in triage" in models["wl_lesion_classifier"]["notes"]
    assert models["spectral_classifier"]["status"] == "placeholder"
    assert models["spectral_classifier"]["notes"].startswith("Spectral classifier: placeholder")


def test_schema_endpoint_serves_the_checked_in_schema(client):
    res = client.get("/api/v1/schema/screening-result")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/schema+json")
    assert res.json() == json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert res.json()["x-schema-version"] == "1.0.0"


def test_sites_catalogue_and_geometry(client):
    body = client.get("/api/v1/sites").json()
    assert body["viewBox"] == "0 0 400 320"
    assert (body["width"], body["height"]) == (400, 320)
    assert "right side is drawn on the viewer's left" in body["orientation"]
    assert [s["site"] for s in body["sites"]] == [s.value for s in SITES]
    assert {s["site"] for s in body["sites"]} == {s.value for s in OralSite}
    left = next(s for s in body["sites"] if s["site"] == "buccal_mucosa_left")
    info = SITES[OralSite.BUCCAL_MUCOSA_LEFT]
    assert left["label"] == info.label
    assert left["paired_site"] == "buccal_mucosa_right"
    assert left["pairing"] == "contralateral"
    assert left["shape"] == {
        "cx": info.shape.cx,
        "cy": info.shape.cy,
        "rx": info.shape.rx,
        "ry": info.shape.ry,
    }


def test_openapi_metadata_and_positioning(client):
    res = client.get("/openapi.json")
    assert res.status_code == 200
    spec = res.json()
    assert spec["info"]["title"] == "MukhaSpectra API"
    assert spec["info"]["version"] == __version__
    assert PROTOTYPE_BANNER in spec["info"]["description"]
    assert "Screening aid — not a diagnosis" in spec["info"]["description"]
    paths = spec["paths"]
    for path in (
        "/api/v1/health",
        "/api/v1/version",
        "/api/v1/schema/screening-result",
        "/api/v1/sites",
        "/api/v1/screenings",
        "/api/v1/demo/screenings/{kind}",
        "/api/v1/screenings/{screening_id}",
        "/api/v1/screenings/{screening_id}/results/{version}",
        "/api/v1/screenings/{screening_id}/reanalyze",
        "/api/v1/screenings/{screening_id}/artifacts/{key}",
        "/api/v1/screenings/{screening_id}/report.pdf",
        "/api/v1/screenings/{screening_id}/report.html",
        "/api/v1/screenings/{screening_id}/sitemap.svg",
    ):
        assert path in paths, path
    assert not find_positioning_violations(res.text)


def test_openapi_documents_the_multipart_upload(client):
    spec = client.get("/openapi.json").json()
    body = spec["paths"]["/api/v1/screenings"]["post"]["requestBody"]
    form = body["content"]["multipart/form-data"]["schema"]
    assert form["required"] == ["manifest"]
    assert form["properties"]["manifest"]["contentSchema"] == {
        "$ref": "#/components/schemas/UploadManifest"
    }
    components = spec["components"]["schemas"]
    manifest = components["UploadManifest"]
    assert {"patient", "consent", "sites", "spectral_file"} <= set(manifest["properties"])
    assert "ManifestSite" in components and "ManifestPatient" in components
    assert client.get("/openapi.json").json() == spec  # cached


def test_security_headers_on_every_response(client):
    res = client.get("/api/v1/health")
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["referrer-policy"] == "no-referrer"
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["x-mukhaspectra-notice"].startswith("Research prototype")


def test_cors_allows_the_configured_console_origins(client):
    res = client.options(
        "/api/v1/screenings",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_module_level_app_instance():
    import app.main

    assert isinstance(app.main.app, FastAPI)
    assert app.main.app.title == "MukhaSpectra API"


def test_unknown_route_is_404(client):
    assert client.get("/api/v1/nope").status_code == 404


def test_creating_the_app_has_no_filesystem_side_effects(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.settings import Settings

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'nested' / 'db.sqlite'}",
        storage_dir=tmp_path / "objects",
    )
    app = create_app(settings)
    assert not (tmp_path / "nested").exists() and not (tmp_path / "objects").exists()
    with TestClient(app) as client:  # startup prepares the folder and migrates
        assert client.get("/api/v1/health").status_code == 200
        assert (tmp_path / "nested" / "db.sqlite").exists()
