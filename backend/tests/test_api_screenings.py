"""Listing, detail, result versions, re-analysis, artifacts, path traversal, PII and audit."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sqlalchemy import select

from api_fakes import PII_MARKERS, default_manifest, upload
from app.models import AnalysisResult, AuditEvent, Capture, PatientIdentity, Screening
from mukha_pipeline.positioning import find_positioning_violations


@pytest.fixture
def screening_id(client) -> str:
    res = upload(client)
    assert res.status_code == 201, res.text
    return res.json()["screening_id"]


# --------------------------------------------------------------------------------------------
# List and detail
# --------------------------------------------------------------------------------------------


def test_list_is_empty_initially(client):
    res = client.get("/api/v1/screenings")
    assert res.status_code == 200
    assert res.json() == []
    assert res.headers["x-total-count"] == "0"


def test_list_newest_first_with_paging(client):
    low = default_manifest()
    low["sites"][0]["visual_flag"] = {"lesion_seen": False}
    ids = [upload(client, m).json()["screening_id"] for m in (default_manifest(), low,
                                                             default_manifest())]
    res = client.get("/api/v1/screenings")
    items = res.json()
    assert res.headers["x-total-count"] == "3"
    assert [i["screening_id"] for i in items] == ids[::-1]
    middle = items[1]
    assert middle["overall_risk_level"] == "LOW"
    assert middle["lesion_count"] == 0
    assert middle["latest_result_version"] == 1
    assert middle["site_count"] == 2
    assert middle["status"] == "analysed"
    assert middle["source"] == "upload"
    assert middle["pseudonymous_id"].startswith("P-")
    assert middle["age_years"] == 52 and middle["sex"] == "female"
    assert middle["screening_site_name"] == "PHC Hoskote camp"
    assert middle["review_status"] == "pending_review"
    assert items[0]["overall_risk_level"] == "HIGH" and items[0]["lesion_count"] == 1

    page = client.get("/api/v1/screenings", params={"limit": 1, "offset": 1}).json()
    assert [i["screening_id"] for i in page] == [ids[1]]
    assert client.get("/api/v1/screenings", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/screenings", params={"limit": 500}).status_code == 422


def test_list_shows_screenings_without_results(client, fake_pipeline):
    fake_pipeline.fail = True
    upload(client)
    (item,) = client.get("/api/v1/screenings").json()
    assert item["status"] == "analysis_failed"
    assert item["latest_result_version"] is None and item["overall_risk_level"] is None


def test_detail(client, screening_id):
    res = client.get(f"/api/v1/screenings/{screening_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["screening_id"] == screening_id
    assert body["status"] == "analysed"
    assert body["created_at"].endswith(("Z", "+00:00"))
    assert body["latest"]["result_version"] == 1
    (version,) = body["versions"]
    assert version["version"] == 1
    assert version["result_id"] == body["latest"]["result_id"]
    assert version["overall_risk_level"] == "HIGH"
    assert version["supersedes_result_id"] is None
    assert not find_positioning_violations(res.text)


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/screenings/{id}",
        "/api/v1/screenings/{id}/results/1",
        "/api/v1/screenings/{id}/artifacts/sites/x.png",
        "/api/v1/screenings/{id}/report.pdf",
        "/api/v1/screenings/{id}/report.html",
        "/api/v1/screenings/{id}/sitemap.svg",
    ],
)
def test_unknown_screening_is_404(client, path):
    res = client.get(path.format(id="00000000-0000-0000-0000-000000000000"))
    assert res.status_code == 404
    assert res.json() == {"detail": "Screening not found."}


def test_unknown_screening_reanalyze_is_404(client):
    res = client.post("/api/v1/screenings/does-not-exist/reanalyze")
    assert res.status_code == 404


def test_result_versions(client, screening_id):
    res = client.get(f"/api/v1/screenings/{screening_id}/results/1")
    assert res.status_code == 200
    assert res.json()["result_version"] == 1
    missing = client.get(f"/api/v1/screenings/{screening_id}/results/9")
    assert missing.status_code == 404
    assert "version 9" in missing.json()["detail"]
    assert client.get(f"/api/v1/screenings/{screening_id}/results/0").status_code == 422


# --------------------------------------------------------------------------------------------
# Re-analysis
# --------------------------------------------------------------------------------------------


def test_reanalyze_creates_a_new_version_and_never_overwrites(client, services, db,
                                                              screening_id, fake_pipeline):
    v1_before = client.get(f"/api/v1/screenings/{screening_id}/results/1").json()
    res = client.post(f"/api/v1/screenings/{screening_id}/reanalyze")
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["result_version"] == 2
    v2 = body["result"]
    assert v2["result_version"] == 2
    assert v2["supersedes_result_id"] == v1_before["result_id"]
    assert v2["result_id"] != v1_before["result_id"]
    # version 1 is untouched
    assert client.get(f"/api/v1/screenings/{screening_id}/results/1").json() == v1_before

    detail = client.get(f"/api/v1/screenings/{screening_id}").json()
    assert [v["version"] for v in detail["versions"]] == [1, 2]
    assert detail["latest"]["result_version"] == 2
    assert detail["versions"][1]["supersedes_result_id"] == v1_before["result_id"]

    for version in (1, 2):
        assert services.storage.exists(
            f"screenings/{screening_id}/results/v{version}/sites/buccal_mucosa_left/wl.png"
        )

    # the re-analysis input was rebuilt from storage: same pixels, metadata and readings
    first, second = fake_pipeline.analyses
    assert second.result_version == 2 and second.supersedes_result_id == v1_before["result_id"]
    for a, b in zip(first.sites, second.sites, strict=True):
        assert a.site == b.site and a.role == b.role and a.visual_flag == b.visual_flag
        for slot in ("wl", "af", "dark"):
            ca, cb = getattr(a, slot), getattr(b, slot)
            assert (ca is None) == (cb is None)
            if ca is not None:
                assert np.array_equal(ca.image, cb.image)
                assert ca.meta == cb.meta
    assert [r.reading_id for r in first.spectral] == [r.reading_id for r in second.spectral]
    assert second.patient == first.patient
    assert second.consent == first.consent
    assert second.risk_factors == first.risk_factors
    assert second.context == first.context
    assert second.now == first.now  # calibration validity is judged at capture time
    assert fake_pipeline.spectral_calls[1]["now"] == fake_pipeline.spectral_calls[0]["now"]

    actions = [e.action for e in db.scalars(
        select(AuditEvent).where(AuditEvent.entity_id == screening_id).order_by(AuditEvent.at))]
    assert "reanalysis_completed" in actions

    third = client.post(f"/api/v1/screenings/{screening_id}/reanalyze").json()
    assert third["result_version"] == 3
    assert third["result"]["supersedes_result_id"] == v2["result_id"]


def test_reanalyze_with_missing_stored_inputs_is_500(client, services, db, screening_id):
    capture = db.scalars(select(Capture).where(Capture.screening_id == screening_id)).first()
    (services.storage.root / capture.storage_key).unlink()
    res = client.post(f"/api/v1/screenings/{screening_id}/reanalyze")
    assert res.status_code == 500
    assert "could not be read" in res.json()["detail"]
    db.expire_all()
    assert len(db.scalars(select(AnalysisResult)).all()) == 1


# --------------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "media_type"),
    [
        ("sites/buccal_mucosa_left/heatmap_overlay.png", "image/png"),
        ("sites/buccal_mucosa_left/relative_af_grid.json", "application/json"),
        ("sites/buccal_mucosa_left/contour.svg", "image/svg+xml"),
    ],
)
def test_artifacts_are_served_with_their_media_type(client, screening_id, key, media_type):
    res = client.get(f"/api/v1/screenings/{screening_id}/artifacts/{key}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith(media_type)
    assert "default-src 'none'" in res.headers["content-security-policy"]
    assert res.headers["x-content-type-options"] == "nosniff"
    if media_type == "application/json":
        assert res.json()["threshold"] == 0.75


def test_artifacts_by_version(client, services, screening_id):
    client.post(f"/api/v1/screenings/{screening_id}/reanalyze")
    key = "sites/buccal_mucosa_left/wl.png"
    services.storage.put(f"screenings/{screening_id}/results/v1/{key}", b"\x89PNG-v1")
    v1 = client.get(f"/api/v1/screenings/{screening_id}/artifacts/{key}", params={"version": 1})
    latest = client.get(f"/api/v1/screenings/{screening_id}/artifacts/{key}")
    assert v1.content == b"\x89PNG-v1"
    assert latest.status_code == 200 and latest.content != b"\x89PNG-v1"
    assert client.get(f"/api/v1/screenings/{screening_id}/artifacts/{key}",
                      params={"version": 7}).status_code == 404


def test_missing_artifact_is_404(client, screening_id):
    res = client.get(f"/api/v1/screenings/{screening_id}/artifacts/sites/nope.png")
    assert res.status_code == 404
    assert res.json() == {"detail": "Artifact not found."}


def test_unknown_extension_is_octet_stream(client, services, screening_id):
    services.storage.put(f"screenings/{screening_id}/results/v1/raw/blob.bin", b"\x00\x01")
    res = client.get(f"/api/v1/screenings/{screening_id}/artifacts/raw/blob.bin")
    assert res.headers["content-type"] == "application/octet-stream"
    services.storage.put(f"screenings/{screening_id}/results/v1/README", b"x")
    res = client.get(f"/api/v1/screenings/{screening_id}/artifacts/README")
    assert res.headers["content-type"] == "application/octet-stream"


TRAVERSAL_KEYS = [
    "..%2F..%2F..%2F..%2Fsecret.txt",
    "..%2f..%2f..%2f..%2f..%2fsecret.txt",
    "%2e%2e/%2e%2e/%2e%2e/%2e%2e/secret.txt",
    "%2E%2E%2F%2E%2E%2Finputs",
    "sites/..%2F..%2F..%2Finputs/{capture}",
    "..%2F..%2Finputs%2F{capture}",
    "%2Fetc%2Fpasswd",
    "%2F{root}%2Fsecret.txt",
    "..%5C..%5Csecret.txt",
    "sites%5C..%5C..%5Csecret.txt",
    "sites/%00/x.png",
    "sites//x.png",
    "./sites/x.png",
    "%252e%252e%252fsecret.txt",
]


@pytest.mark.parametrize("raw_key", TRAVERSAL_KEYS)
def test_path_traversal_is_blocked(client, services, db, screening_id, tmp_path, raw_key):
    secret = "TOP-SECRET-CONTENT"
    (tmp_path / "secret.txt").write_text(secret)
    services.storage.root.joinpath("secret.txt").write_text(secret)
    capture = db.scalars(select(Capture).where(Capture.screening_id == screening_id)).first()
    key = raw_key.format(capture=capture.storage_key.rsplit("/", 1)[-1],
                         root=str(services.storage.root).strip("/").replace("/", "%2F"))
    res = client.get(f"/api/v1/screenings/{screening_id}/artifacts/{key}")
    assert res.status_code in (400, 404), (key, res.status_code)
    assert secret not in res.text
    assert res.content != services.storage.get(capture.storage_key)


def test_invalid_artifact_key_message(client, screening_id):
    res = client.get(f"/api/v1/screenings/{screening_id}/artifacts/..%2Fx.png")
    assert res.status_code == 400
    assert res.json() == {"detail": "Invalid artifact key."}


# --------------------------------------------------------------------------------------------
# Privacy and audit
# --------------------------------------------------------------------------------------------


def _assert_no_pii(text: str) -> None:
    lowered = text.lower()
    for marker in PII_MARKERS:
        assert marker not in lowered, marker


def test_pii_never_leaves_the_identity_table(client, services, db, screening_id):
    created = client.post(f"/api/v1/screenings/{screening_id}/reanalyze").text
    _assert_no_pii(created)
    for path in (f"/api/v1/screenings/{screening_id}", "/api/v1/screenings",
                 f"/api/v1/screenings/{screening_id}/results/1"):
        _assert_no_pii(client.get(path).text)

    # clinical tables, audit log and stored inputs/artifacts hold no PII
    for model in (Screening, Capture, AnalysisResult, AuditEvent):
        for row in db.scalars(select(model)):
            values = {c.key: getattr(row, c.key) for c in model.__mapper__.column_attrs}
            _assert_no_pii(json.dumps(values, default=str))
    for key in services.storage.list_prefix(f"screenings/{screening_id}/"):
        _assert_no_pii(key)
        if not key.endswith((".png", ".pdf")):
            _assert_no_pii(services.storage.get(key).decode("utf-8", errors="replace"))

    # ... and the identity table does hold it, keyed by the pseudonymous id
    identity = db.scalars(select(PatientIdentity)).one()
    screening = db.get(Screening, screening_id)
    assert identity.pseudonymous_id == screening.pseudonymous_id
    assert identity.name == "Asha Ramesh Testpatient"
    assert identity.external_id == "HOSP-EXT-424242"


def test_upload_without_identity_creates_no_identity_row(client, db):
    manifest = default_manifest(patient={"age_years": 30, "sex": "male", "name": "  "})
    assert upload(client, manifest).status_code == 201
    assert db.scalars(select(PatientIdentity)).all() == []


def test_audit_events_are_written(client, db, screening_id):
    client.get(f"/api/v1/screenings/{screening_id}")
    client.get(f"/api/v1/screenings/{screening_id}/results/1")
    client.post(f"/api/v1/screenings/{screening_id}/reanalyze")
    client.get(f"/api/v1/screenings/{screening_id}/report.pdf")
    client.get(f"/api/v1/screenings/{screening_id}/report.pdf")
    client.get(f"/api/v1/screenings/{screening_id}/report.html")
    events = db.scalars(select(AuditEvent).order_by(AuditEvent.at)).all()
    actions = [e.action for e in events]
    assert actions.count("screening_created") == 1
    assert actions.count("analysis_completed") == 1
    assert actions.count("reanalysis_completed") == 1
    assert actions.count("result_viewed") == 2
    assert actions.count("report_generated") == 1
    assert actions.count("report_viewed") == 3
    assert all(e.actor == "anonymous" for e in events)
    created = next(e for e in events if e.action == "screening_created")
    assert created.entity_type == "screening" and created.entity_id == screening_id
    assert created.detail["consent_method"] == "checkbox"
    assert created.detail["captures"] == 6
    completed = next(e for e in events if e.action == "analysis_completed")
    assert completed.detail["result_version"] == 1
    assert len(completed.detail["rule_config_sha256"]) == 64
