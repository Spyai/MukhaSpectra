"""POST /api/v1/screenings: happy path, consent gate, validation, decoding errors, limits,
analysis failure."""

from __future__ import annotations

import json

import numpy as np
import pytest
from api_fakes import default_files, default_manifest, png_bytes, upload
from sqlalchemy import func, select

from app.errors import ANALYSIS_FAILED_MESSAGE
from app.models import (
    AnalysisResult,
    AuditEvent,
    CalibrationRecord,
    Capture,
    Patient,
    PatientIdentity,
    Report,
    Screening,
)
from app.services.ingest import CONSENT_REQUIRED
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import ScreeningResult

ALL_TABLES = (
    Patient,
    PatientIdentity,
    Screening,
    Capture,
    AnalysisResult,
    Report,
    CalibrationRecord,
    AuditEvent,
)


def _row_counts(db) -> dict[str, int]:
    return {m.__tablename__: db.scalar(select(func.count()).select_from(m)) for m in ALL_TABLES}


def _stored_files(services) -> list[str]:
    root = services.storage.root
    return (
        [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
        if root.exists()
        else []
    )


def assert_nothing_stored(services, db) -> None:
    assert all(n == 0 for n in _row_counts(db).values()), _row_counts(db)
    assert _stored_files(services) == []


# --------------------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------------------


def test_upload_happy_path(client, services, db, fake_pipeline):
    res = upload(client)
    assert res.status_code == 201, res.text
    body = res.json()
    sid = body["screening_id"]
    assert body["result_version"] == 1
    result = ScreeningResult.model_validate(body["result"])
    assert result.screening_id == sid
    assert result.result_version == 1 and result.supersedes_result_id is None
    assert result.patient.pseudonymous_id.startswith("P-")
    assert result.patient.age_years == 52
    assert result.triage.overall_risk_level == "HIGH"
    assert not find_positioning_violations(res.text)

    # the pipeline got decoded float images, the spectral readings and the injected clock
    (inp,) = fake_pipeline.analyses
    assert [s.site.value for s in inp.sites] == ["buccal_mucosa_left", "buccal_mucosa_right"]
    assert inp.sites[0].dark is not None and inp.sites[1].dark is None
    assert inp.sites[1].role == "reference_only"
    assert {r.reading_id for r in inp.spectral} == {"s1", "s2"}
    assert inp.now is not None and inp.now.tzinfo is not None
    assert inp.context.device_ids == ["scout-01"]
    assert inp.context.firmware_versions == ["0.1.0"]

    # neutral file names were given to the adapter (never the uploaded ones)
    assert fake_pipeline.adapter_calls[0]["filenames"] == {
        "wl": "buccal_mucosa_left-wl.png",
        "af": "buccal_mucosa_left-af.png",
        "dark": "buccal_mucosa_left-dark.png",
    }
    assert fake_pipeline.spectral_calls[0]["filename"] == "spectral.csv"

    # raw inputs stored byte-for-byte; artifacts stored under results/v1
    files = default_files()
    captures = db.scalars(select(Capture).where(Capture.screening_id == sid)).all()
    assert len(captures) == 6
    stored = {c.kind: [] for c in captures}
    for c in captures:
        assert c.storage_key.startswith(f"screenings/{sid}/inputs/{c.id}")
        stored[c.kind].append(services.storage.get(c.storage_key))
    assert files["wl_bl"][1] in stored["white_light"]
    assert stored["spectral"] == [files["spectrum"][1]]
    assert {c.file_name for c in captures} >= {"buccal_mucosa_left-wl.png", "spectral.csv"}
    assert services.storage.exists(
        f"screenings/{sid}/results/v1/sites/buccal_mucosa_left/heatmap_overlay.png"
    )

    screening = db.get(Screening, sid)
    assert screening.status == "analysed"
    assert screening.source == "upload"
    assert screening.synthetic is False
    assert screening.consent["consent_recorded"] is True
    assert screening.site_inputs[0]["captures"]["dark"] is not None
    row = db.scalars(select(AnalysisResult).where(AnalysisResult.screening_id == sid)).one()
    assert row.id == result.result_id and row.version == 1
    assert row.overall_risk_level == "HIGH"
    assert row.rule_config_sha256 == result.versions.rule_config.sha256
    assert row.pipeline_version == "0.0.0-fake"
    actions = [e.action for e in db.scalars(select(AuditEvent).order_by(AuditEvent.at))]
    assert actions == ["screening_created", "analysis_completed"]


def test_upload_without_dark_or_spectrum(client, db):
    manifest = default_manifest(spectral_file=None)
    manifest["sites"][0]["dark"] = None
    files = {k: v for k, v in default_files().items() if k not in ("dark_bl", "spectrum")}
    res = upload(client, manifest, files)
    assert res.status_code == 201, res.text
    assert db.scalar(select(func.count()).select_from(Capture)) == 4


def test_unreferenced_extra_files_are_not_stored(client, db):
    files = {**default_files(), "unused": ("x.png", png_bytes(), "image/png")}
    assert upload(client, files=files).status_code == 201
    assert db.scalar(select(func.count()).select_from(Capture)) == 6


def test_synthetic_provenance_is_carried_through(client, db, fake_pipeline):
    res = upload(client, default_manifest(data_provenance="synthetic"))
    assert res.status_code == 201
    result = res.json()["result"]
    assert result["synthetic"] is True
    assert result["patient"]["pseudonymous_id"].startswith("SYN-")
    assert fake_pipeline.adapter_calls[0]["synthetic"] is True
    assert db.get(Screening, res.json()["screening_id"]).synthetic is True


def test_manifest_may_be_sent_as_a_file_part(client):
    files = {
        **default_files(),
        "manifest": ("manifest.json", json.dumps(default_manifest()).encode(), "application/json"),
    }
    res = client.post("/api/v1/screenings", files=files)
    assert res.status_code == 201, res.text


# --------------------------------------------------------------------------------------------
# Consent gate: rejected before anything is stored
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "consent",
    [
        {"consent_recorded": False, "method": "checkbox"},
        {"method": "checkbox"},
        {"consent_recorded": "yes"},
        None,
    ],
)
def test_consent_missing_is_rejected_and_nothing_is_stored(
    client, services, db, consent, fake_pipeline
):
    manifest = default_manifest()
    if consent is None:
        manifest.pop("consent")
    else:
        manifest["consent"] = consent
    res = upload(client, manifest)
    assert res.status_code == 422
    assert res.json() == {"detail": CONSENT_REQUIRED}
    assert_nothing_stored(services, db)
    assert fake_pipeline.adapter_calls == [] and fake_pipeline.analyses == []


def test_consent_is_checked_before_other_manifest_errors(client, services, db):
    manifest = default_manifest(consent={"consent_recorded": False})
    manifest["sites"][0]["site"] = "not_a_site"
    res = upload(client, manifest)
    assert res.json()["detail"] == CONSENT_REQUIRED
    assert_nothing_stored(services, db)


# --------------------------------------------------------------------------------------------
# Manifest and file validation -> 422, nothing stored
# --------------------------------------------------------------------------------------------


def _manifest_with(mutator) -> dict:
    manifest = default_manifest()
    mutator(manifest)
    return manifest


@pytest.mark.parametrize(
    ("manifest_text", "expected"),
    [
        (None, "must include a 'manifest' form field"),
        ("", "must include a 'manifest' form field"),
        ("{not json", "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        (
            json.dumps(_manifest_with(lambda m: m["sites"][0].update(site="moon"))),
            "manifest.sites.0.site",
        ),
        (
            json.dumps(_manifest_with(lambda m: m["sites"][1].update(site="buccal_mucosa_left"))),
            "listed more than once",
        ),
        (json.dumps(_manifest_with(lambda m: m.update(sites=[]))), "manifest.sites"),
        (json.dumps(_manifest_with(lambda m: m.update(unexpected=True))), "manifest.unexpected"),
        (
            json.dumps(_manifest_with(lambda m: m["patient"].update(age_years=300))),
            "manifest.patient.age_years",
        ),
        (
            json.dumps(_manifest_with(lambda m: m["sites"][0].update(px_per_mm=-1))),
            "manifest.sites.0.px_per_mm",
        ),
        (
            json.dumps(_manifest_with(lambda m: m["sites"][0].update(wl="../etc"))),
            "manifest.sites.0.wl",
        ),
        (
            json.dumps(_manifest_with(lambda m: m["sites"][0].update(wl="manifest"))),
            "'manifest' cannot be used",
        ),
        (
            json.dumps(
                _manifest_with(lambda m: [s.update(role="reference_only") for s in m["sites"]])
            ),
            "role 'examined'",
        ),
    ],
)
def test_invalid_manifest_is_422_with_a_clear_message(
    client, services, db, manifest_text, expected
):
    data = {} if manifest_text is None else {"manifest": manifest_text}
    res = client.post("/api/v1/screenings", data=data, files=default_files())
    assert res.status_code == 422, res.text
    assert isinstance(res.json()["detail"], str)
    assert expected in res.json()["detail"]
    assert_nothing_stored(services, db)


def test_many_manifest_errors_are_summarised(client):
    manifest = default_manifest()
    manifest["sites"] = [{"site": f"bad{i}"} for i in range(5)]
    detail = upload(client, manifest).json()["detail"]
    assert detail.startswith("Invalid manifest.") and "more)" in detail


def test_json_body_instead_of_multipart_is_422(client):
    res = client.post("/api/v1/screenings", json=default_manifest())
    assert res.status_code == 422
    assert "manifest" in res.json()["detail"]


def test_missing_file_field_is_422_and_nothing_stored(client, services, db):
    files = {k: v for k, v in default_files().items() if k != "af_br"}
    res = upload(client, files=files)
    assert res.status_code == 422
    assert "'af_br'" in res.json()["detail"] and "sites[1].af" in res.json()["detail"]
    assert_nothing_stored(services, db)


def test_missing_spectral_file_is_422(client, services, db):
    files = {k: v for k, v in default_files().items() if k != "spectrum"}
    res = upload(client, files=files)
    assert res.status_code == 422
    assert "spectral_file" in res.json()["detail"]
    assert_nothing_stored(services, db)


def test_file_field_sent_as_text_is_422(client, services, db):
    files = {k: v for k, v in default_files().items() if k != "wl_bl"}
    res = client.post(
        "/api/v1/screenings",
        data={"manifest": json.dumps(default_manifest()), "wl_bl": "not a file"},
        files=files,
    )
    assert res.status_code == 422
    assert "must be a file" in res.json()["detail"]
    assert_nothing_stored(services, db)


def test_empty_file_is_422(client, services, db):
    files = default_files()
    files["af_bl"] = ("af.png", b"", "image/png")
    res = upload(client, files=files)
    assert res.status_code == 422
    assert "is empty" in res.json()["detail"]
    assert_nothing_stored(services, db)


def test_undecodable_image_is_422_and_nothing_stored(client, services, db):
    files = default_files()
    files["af_bl"] = ("af.png", b"this is not an image", "image/png")
    res = upload(client, files=files)
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert detail.startswith("Could not read the images for Left buccal mucosa")
    assert "retake" in detail.lower()
    assert_nothing_stored(services, db)


def test_bad_spectral_file_is_422_and_nothing_stored(client, services, db):
    files = default_files()
    files["spectrum"] = ("s.csv", b"just,some,columns\n1,2,3\n", "text/csv")
    res = upload(client, files=files)
    assert res.status_code == 422
    assert res.json()["detail"].startswith("Could not read the spectral file")
    assert_nothing_stored(services, db)


def test_file_over_the_size_limit_is_413(make_client):
    client = make_client(max_upload_mb=1)
    services = client.app.state.services
    files = default_files()
    files["wl_bl"] = ("big.png", b"\x89PNG" + b"0" * (1024 * 1024 + 10), "image/png")
    res = upload(client, files=files)
    assert res.status_code == 413
    assert "1 MB limit" in res.json()["detail"]
    with services.session_factory() as session:
        assert_nothing_stored(services, session)


def test_request_over_the_total_limit_is_413(make_client):
    client = make_client(max_request_mb=1)
    files = default_files()
    files["wl_bl"] = ("big.png", b"0" * (1024 * 1024 + 10), "image/png")
    res = upload(client, files=files)
    assert res.status_code == 413
    assert "request limit" in res.json()["detail"]


def test_pipeline_not_installed_is_503(make_client, fake_pipeline, db):
    client = make_client()

    def missing():
        raise ImportError("mukha_pipeline.capture.upload")

    services = client.app.state.services
    services.ingest.bridge = services.ingest.bridge.replace(upload_adapter_factory=missing)
    res = upload(client)
    assert res.status_code == 503
    assert res.json()["detail"] == "The analysis pipeline is not available on this server."


# --------------------------------------------------------------------------------------------
# Analysis failure keeps the captures; re-analysis recovers
# --------------------------------------------------------------------------------------------


def test_analysis_failure_keeps_captures_and_can_be_reanalysed(client, services, db, fake_pipeline):
    fake_pipeline.fail = True
    res = upload(client)
    assert res.status_code == 500
    body = res.json()
    assert body["detail"] == ANALYSIS_FAILED_MESSAGE
    sid = body["screening_id"]
    screening = db.get(Screening, sid)
    assert screening.status == "analysis_failed"
    assert db.scalar(select(func.count()).select_from(Capture)) == 6
    assert db.scalar(select(func.count()).select_from(AnalysisResult)) == 0
    assert services.storage.list_prefix(f"screenings/{sid}/results/") == []
    actions = [e.action for e in db.scalars(select(AuditEvent).order_by(AuditEvent.at))]
    assert actions == ["screening_created", "analysis_failed"]

    detail = client.get(f"/api/v1/screenings/{sid}").json()
    assert detail["status"] == "analysis_failed" and detail["latest"] is None
    assert client.get(f"/api/v1/screenings/{sid}/report.pdf").status_code == 404

    fake_pipeline.fail = False
    res = client.post(f"/api/v1/screenings/{sid}/reanalyze")
    assert res.status_code == 201, res.text
    assert res.json()["result_version"] == 1
    assert res.json()["result"]["supersedes_result_id"] is None
    db.expire_all()
    assert db.get(Screening, sid).status == "analysed"
    # re-analysis saw the same pixels as the failed first attempt
    first, second = fake_pipeline.analyses
    for a, b in zip(first.sites, second.sites, strict=True):
        assert np.array_equal(a.wl.image, b.wl.image)
        assert a.wl.meta == b.wl.meta


def test_invalid_artifact_key_from_the_pipeline_fails_the_analysis(client, services, fake_pipeline):
    fake_pipeline.extra_artifacts = {"../../escape.png": b"x"}
    res = upload(client)
    assert res.status_code == 500
    assert not (services.storage.root / "escape.png").exists()
    assert not (services.storage.root / "screenings" / "escape.png").exists()


def test_pipeline_result_fields_owned_by_the_backend_are_pinned(client, fake_pipeline):
    original = fake_pipeline.analyze

    def sloppy(inp, *, classifier=None, software_version=None):
        out = original(inp, classifier=classifier, software_version=software_version)
        out.result = out.result.model_copy(update={"screening_id": "wrong", "result_version": 9})
        return out

    client.app.state.services.analysis.bridge = fake_pipeline.bridge().replace(analyze=sloppy)
    body = upload(client).json()
    assert body["result"]["screening_id"] == body["screening_id"]
    assert body["result"]["result_version"] == 1
