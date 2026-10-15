"""Regression tests for security.2: erasure (DPDP Act 2023 withdrawal of consent).

DELETE /api/v1/screenings/{id} and DELETE /api/v1/patients/{pseudonymous_id} remove the database
rows (identity included when nothing else uses it), the stored inputs, every result artifact and
the cached identified PDFs, and leave a PII-free audit event behind.
"""

from __future__ import annotations

import json

from api_fakes import PATIENT_NAME, PII_MARKERS, upload
from sqlalchemy import func, select

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

CLINICAL_TABLES = (
    Patient,
    PatientIdentity,
    Screening,
    Capture,
    AnalysisResult,
    Report,
    CalibrationRecord,
)


def _counts(services) -> dict[str, int]:
    with services.session_factory() as session:
        return {
            m.__tablename__: session.scalar(select(func.count()).select_from(m)) or 0
            for m in CLINICAL_TABLES
        }


def _files(services) -> list[str]:
    root = services.storage.root
    if not root.exists():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _events(services, action: str) -> list[AuditEvent]:
    with services.session_factory() as session:
        return list(session.scalars(select(AuditEvent).where(AuditEvent.action == action)))


def _uploaded_with_report(client) -> str:
    sid = upload(client).json()["screening_id"]
    assert client.post(f"/api/v1/screenings/{sid}/reanalyze").status_code == 201
    pdf = client.get(f"/api/v1/screenings/{sid}/report.pdf")
    assert PATIENT_NAME.encode() in pdf.content  # the cached PDF holds the name
    return sid


def test_delete_screening_erases_everything(client, services):
    sid = _uploaded_with_report(client)
    assert any(f.startswith(f"screenings/{sid}/reports/") for f in _files(services))

    res = client.delete(f"/api/v1/screenings/{sid}")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["erased_screening_ids"] == [sid]
    assert body["patient_record_erased"] is True
    assert body["stored_objects_deleted"] > 0

    assert all(n == 0 for n in _counts(services).values()), _counts(services)
    assert _files(services) == []
    assert client.get(f"/api/v1/screenings/{sid}").status_code == 404
    assert client.get(f"/api/v1/screenings/{sid}/report.pdf").status_code == 404
    assert client.delete(f"/api/v1/screenings/{sid}").status_code == 404

    (event,) = _events(services, "screening_erased")
    assert event.entity_id == sid
    detail = json.dumps(event.detail).lower()
    assert not any(marker in detail for marker in PII_MARKERS)
    assert event.detail["result_versions"] == 2 and event.detail["reports"] == 1


def test_identity_is_kept_while_another_screening_uses_it(client, services):
    first = upload(client).json()
    second = upload(client).json()
    with services.session_factory() as session:
        # Two screenings of one patient (Phase 2 follow-ups); simulated by re-pointing.
        shared = first["result"]["patient"]["pseudonymous_id"]
        row = session.get(Screening, second["screening_id"])
        orphan = row.pseudonymous_id
        row.pseudonymous_id = shared
        session.delete(session.get(Patient, orphan))
        session.commit()

    res = client.delete(f"/api/v1/screenings/{first['screening_id']}")
    assert res.status_code == 200
    assert res.json()["patient_record_erased"] is False
    with services.session_factory() as session:
        assert session.get(Patient, shared) is not None
        assert session.scalar(select(func.count()).select_from(PatientIdentity)) == 1
    assert client.get(f"/api/v1/screenings/{second['screening_id']}").status_code == 200

    res = client.delete(f"/api/v1/patients/{shared}")
    assert res.status_code == 200, res.text
    assert res.json()["erased_screening_ids"] == [second["screening_id"]]
    assert res.json()["patient_record_erased"] is True
    assert all(n == 0 for n in _counts(services).values())
    assert _files(services) == []
    assert len(_events(services, "screening_erased")) == 2
    (patient_event,) = _events(services, "patient_erased")
    assert not any(m in json.dumps(patient_event.detail).lower() for m in PII_MARKERS)


def test_delete_patient_erases_all_of_their_screenings(client, services):
    sid = _uploaded_with_report(client)
    with services.session_factory() as session:
        pid = session.get(Screening, sid).pseudonymous_id
    res = client.delete(f"/api/v1/patients/{pid}")
    assert res.status_code == 200
    assert res.json()["erased_screening_ids"] == [sid]
    assert all(n == 0 for n in _counts(services).values())
    assert _files(services) == []
    assert client.delete(f"/api/v1/patients/{pid}").status_code == 404


def test_demo_screening_erasure_removes_its_calibrations(client, services):
    sid = client.post("/api/v1/demo/screenings/high").json()["screening_id"]
    assert any(f.startswith("calibrations/") for f in _files(services))
    assert client.delete(f"/api/v1/screenings/{sid}").status_code == 200
    assert all(n == 0 for n in _counts(services).values())
    assert _files(services) == []


def test_unknown_ids_are_404(client):
    assert client.delete("/api/v1/screenings/nope").status_code == 404
    assert client.delete("/api/v1/patients/P-NOPE").status_code == 404


def test_erasure_is_documented_in_openapi(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "delete" in paths["/api/v1/screenings/{screening_id}"]
    assert "delete" in paths["/api/v1/patients/{pseudonymous_id}"]
