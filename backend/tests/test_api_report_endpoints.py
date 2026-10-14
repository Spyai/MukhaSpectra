"""Report endpoints (PDF / HTML / site map) with a fake renderer: caching, identity join,
versioning, artifact access and error handling."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from api_fakes import upload
from app.models import AuditEvent, Report
from app.services.reports import REPORT_FAILED, ArtifactReader, report_display_id


@pytest.fixture
def screening_id(client) -> str:
    return upload(client).json()["screening_id"]


def test_pdf_is_rendered_once_then_served_from_storage(client, services, db, screening_id,
                                                       fake_renderer):
    url = f"/api/v1/screenings/{screening_id}/report.pdf"
    first = client.get(url)
    assert first.status_code == 200
    assert first.headers["content-type"] == "application/pdf"
    assert first.headers["content-disposition"] == (
        f'inline; filename="MukhaSpectra_{screening_id}_v1.pdf"'
    )
    assert first.headers["cache-control"] == "no-store, private"
    assert first.content.startswith(b"%PDF")
    second = client.get(url)
    assert second.content == first.content
    assert len(fake_renderer.pdf_calls) == 1  # cached

    report = db.scalars(select(Report)).one()
    assert report.kind == "clinician" and report.version == 1
    assert report.storage_key == f"screenings/{screening_id}/reports/v1/clinician.pdf"
    assert services.storage.get(report.storage_key) == first.content
    assert len(report.sha256) == 64 and report.size_bytes == len(first.content)
    actions = [e.action for e in db.scalars(select(AuditEvent).where(
        AuditEvent.entity_type == "report"))]
    assert actions.count("report_generated") == 1
    assert actions.count("report_viewed") == 2


def test_renderer_receives_identity_options_and_an_artifact_reader(client, screening_id,
                                                                   fake_renderer):
    client.get(f"/api/v1/screenings/{screening_id}/report.pdf")
    (call,) = fake_renderer.pdf_calls
    assert call["identity"] == {"name": "Asha Ramesh Testpatient",
                                "external_id": "HOSP-EXT-424242"}
    options = call["options"]
    assert options["record_url"] == f"https://console.example.test/screenings/{screening_id}"
    assert options["report_id"] == report_display_id(screening_id, 1)
    assert options["report_id"].startswith("MSR-") and options["report_id"].endswith("-V1")
    assert options["report_version"] == 1
    assert options["generated_at"].tzinfo is not None
    assert options["facilities"][0]["name"] == "District Hospital (example)"
    assert call["artifact"] is not None and call["artifact"].startswith(b"\x89PNG")
    assert call["traversal"] is None


def test_each_result_version_gets_its_own_pdf(client, db, screening_id, fake_renderer):
    client.post(f"/api/v1/screenings/{screening_id}/reanalyze")
    latest = client.get(f"/api/v1/screenings/{screening_id}/report.pdf")
    assert f"_v2.pdf" in latest.headers["content-disposition"]
    v1 = client.get(f"/api/v1/screenings/{screening_id}/report.pdf", params={"version": 1})
    assert f"MukhaSpectra_{screening_id}_v1.pdf" in v1.headers["content-disposition"]
    assert v1.content != latest.content
    assert len(fake_renderer.pdf_calls) == 2
    assert sorted(r.version for r in db.scalars(select(Report))) == [1, 2]
    assert client.get(f"/api/v1/screenings/{screening_id}/report.pdf",
                      params={"version": 5}).status_code == 404


def test_pdf_is_re_rendered_when_the_stored_file_is_missing(client, services, db, screening_id,
                                                            fake_renderer):
    url = f"/api/v1/screenings/{screening_id}/report.pdf"
    client.get(url)
    report = db.scalars(select(Report)).one()
    (services.storage.root / report.storage_key).unlink()
    res = client.get(url)
    assert res.status_code == 200 and res.content.startswith(b"%PDF")
    assert len(fake_renderer.pdf_calls) == 2
    db.expire_all()
    assert len(db.scalars(select(Report)).all()) == 1


def test_report_html(client, screening_id, fake_renderer):
    res = client.get(f"/api/v1/screenings/{screening_id}/report.html")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert res.headers["cache-control"] == "no-store, private"
    assert "script-src" not in res.headers["content-security-policy"]
    assert "Research prototype — not for clinical use" in res.text
    client.get(f"/api/v1/screenings/{screening_id}/report.html")
    assert fake_renderer.html_calls == 2  # previews are not cached


def test_site_map_svg(client, screening_id, fake_renderer):
    res = client.get(f"/api/v1/screenings/{screening_id}/sitemap.svg")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/svg+xml")
    assert res.text.startswith("<svg")
    assert fake_renderer.svg_calls[-1] == {"width": 400, "show_legend": True}
    client.get(f"/api/v1/screenings/{screening_id}/sitemap.svg",
               params={"width": 640, "legend": "false", "version": 1})
    assert fake_renderer.svg_calls[-1] == {"width": 640, "show_legend": False}
    assert client.get(f"/api/v1/screenings/{screening_id}/sitemap.svg",
                      params={"width": 5}).status_code == 422


def test_renderer_failure_is_a_plain_500(client, db, screening_id, fake_renderer):
    fake_renderer.error = RuntimeError("weasyprint exploded")
    res = client.get(f"/api/v1/screenings/{screening_id}/report.pdf")
    assert res.status_code == 500
    assert res.json() == {"detail": REPORT_FAILED}
    assert db.scalars(select(Report)).all() == []
    assert client.get(f"/api/v1/screenings/{screening_id}/sitemap.svg").status_code == 500


def test_missing_report_engine_is_503(client, screening_id, fake_renderer):
    fake_renderer.error = ImportError("app.reports")
    res = client.get(f"/api/v1/screenings/{screening_id}/report.html")
    assert res.status_code == 503
    assert res.json()["detail"] == "The report engine is not available on this server."


def test_artifact_reader(services, client, screening_id):
    reader = ArtifactReader(services.storage, screening_id, 1)
    assert reader("sites/buccal_mucosa_left/wl.png").startswith(b"\x89PNG")
    assert reader("sites/missing.png") is None
    assert reader("../v2/x.png") is None
    assert reader("/etc/passwd") is None
    with pytest.raises(ValueError):
        reader.key_for("a/../b")
