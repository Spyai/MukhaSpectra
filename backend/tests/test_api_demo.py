"""SYNTHETIC demo patients: storage as captures, labelling, re-analysis and reports."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from api_fakes import make_fake_demo_patient
from PIL import Image
from sqlalchemy import select

from app.models import CalibrationRecord, Capture, PatientIdentity, Screening
from app.services.demo import encode_png_rgb, spectral_inputs_to_files


def test_demo_high(client, services, db, fake_pipeline):
    res = client.post("/api/v1/demo/screenings/high")
    assert res.status_code == 201, res.text
    body = res.json()
    sid = body["screening_id"]
    result = body["result"]
    assert body["result_version"] == 1
    assert result["synthetic"] is True
    assert result["data_provenance"] == "synthetic"
    assert result["notices"]["synthetic_notice"].startswith("SYNTHETIC DATA")
    assert result["patient"]["pseudonymous_id"].startswith("SYN-")
    assert result["triage"]["overall_risk_level"] == "HIGH"
    assert fake_pipeline.demo_calls == ["high"]

    screening = db.get(Screening, sid)
    assert screening.source == "demo" and screening.synthetic is True
    identity = db.scalars(select(PatientIdentity)).one()
    assert identity.name == "SYNTHETIC PATIENT HIGH"
    assert identity.pseudonymous_id == screening.pseudonymous_id

    captures = db.scalars(select(Capture).where(Capture.screening_id == sid)).all()
    kinds = sorted(c.kind for c in captures)
    assert kinds.count("white_light") == 2 and kinds.count("dark_frame") == 2
    assert kinds.count("spectral") == 1
    for c in captures:
        data = services.storage.get(c.storage_key)
        if c.kind == "spectral":
            doc = json.loads(data)
            assert doc["format"] == "mukhaspectra.spectral.v1" and doc["synthetic"] is True
            assert {r["reading_id"] for r in doc["readings"]} == {"syn-l", "syn-r"}
        else:
            assert data.startswith(b"\x89PNG") and c.file_name.endswith(".png")
            assert c.meta["capture_metadata"]["synthetic"] is True

    records = db.scalars(select(CalibrationRecord)).all()
    assert sorted(r.kind for r in records) == ["image", "spectral"]
    assert all(r.synthetic for r in records)
    assert sorted(screening.calibration_ids) == sorted(r.id for r in records)

    # version 1 analysed exactly the pixels that were stored
    (inp,) = fake_pipeline.analyses
    stored = services.storage.get(
        next(
            c for c in captures if c.kind == "white_light" and c.site == inp.sites[0].site.value
        ).storage_key
    )
    decoded = np.asarray(Image.open(io.BytesIO(stored)).convert("RGB"), np.float32) / 255.0
    assert np.array_equal(decoded, inp.sites[0].wl.image)


def test_demo_reanalysis_reproduces_the_inputs(client, fake_pipeline):
    sid = client.post("/api/v1/demo/screenings/moderate").json()["screening_id"]
    res = client.post(f"/api/v1/screenings/{sid}/reanalyze")
    assert res.status_code == 201, res.text
    assert res.json()["result_version"] == 2
    assert res.json()["result"]["synthetic"] is True
    first, second = fake_pipeline.analyses
    assert second.data_provenance == "synthetic"
    assert second.now == first.now
    for a, b in zip(first.sites, second.sites, strict=True):
        for slot in ("wl", "af", "dark"):
            assert np.array_equal(getattr(a, slot).image, getattr(b, slot).image)
            assert getattr(a, slot).meta == getattr(b, slot).meta
    # spectral readings survive the v1 JSON round trip
    by_id = {r.reading_id: r for r in second.spectral}
    for r in first.spectral:
        twin = by_id[r.reading_id]
        assert (twin.site, twin.role, twin.reference_reading_id) == (
            r.site,
            r.role,
            r.reference_reading_id,
        )
        assert [f.channels for f in twin.lit] == [f.channels for f in r.lit]
        assert [f.gain for f in twin.lit] == [f.gain for f in r.lit]
        assert twin.meta.source_adapter == "synthetic"
    # calibrations restored from their records
    cal_a, cal_b = first.image_calibrations["scout-syn"], second.image_calibrations["scout-syn"]
    assert np.array_equal(cal_a.flat_field, cal_b.flat_field)
    assert np.allclose(cal_a.color_matrix, cal_b.color_matrix)
    assert cal_b.af_scale == cal_a.af_scale and cal_b.calibration_id == cal_a.calibration_id
    assert cal_b.expires_at == cal_a.expires_at and cal_b.synthetic
    sp_a, sp_b = (
        first.spectral_calibrations["confirm-syn"],
        second.spectral_calibrations["confirm-syn"],
    )
    assert (sp_b.dark, sp_b.white, sp_b.gain) == (sp_a.dark, sp_a.white, sp_a.gain)
    assert sp_b.created_at == sp_a.created_at


def test_demo_report_uses_the_synthetic_identity(client, fake_renderer):
    sid = client.post("/api/v1/demo/screenings/low").json()["screening_id"]
    pdf = client.get(f"/api/v1/screenings/{sid}/report.pdf")
    assert pdf.status_code == 200
    assert fake_renderer.pdf_calls[0]["identity"]["name"] == "SYNTHETIC PATIENT LOW"


def test_unknown_demo_kind_is_422(client, db):
    res = client.post("/api/v1/demo/screenings/extreme")
    assert res.status_code == 422
    assert res.json() == {
        "detail": "Unknown demo patient 'extreme'. Use one of: low, moderate, high."
    }
    assert db.scalars(select(Screening)).all() == []


@pytest.mark.parametrize("kind", ["high", "extreme"])
def test_demo_disabled_is_404(make_client, kind):
    client = make_client(enable_demo=False)
    res = client.post(f"/api/v1/demo/screenings/{kind}")
    assert res.status_code == 404
    assert res.json() == {"detail": "Demo screenings are disabled on this server."}


def test_demo_generator_unavailable_is_503(client):
    services = client.app.state.services

    def missing(kind):
        raise ImportError("mukha_pipeline.synthetic")

    services.demo.bridge = services.demo.bridge.replace(make_demo_patient=missing)
    res = client.post("/api/v1/demo/screenings/low")
    assert res.status_code == 503


def test_encode_png_rgb_variants():
    gray = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    png, decoded = encode_png_rgb(gray)
    assert png.startswith(b"\x89PNG") and decoded.shape == (3, 4, 3)
    rgba = np.zeros((2, 2, 4), np.uint8)
    rgba[..., 0] = 255
    _, decoded = encode_png_rgb(rgba)
    assert decoded.shape == (2, 2, 3) and decoded[0, 0, 0] == 1.0
    single = np.full((2, 2, 1), 0.5, np.float32)
    _, decoded = encode_png_rgb(single)
    assert decoded.shape == (2, 2, 3)
    assert np.allclose(decoded, 128 / 255)


def test_spectral_inputs_to_files_groups_by_device():
    readings = make_fake_demo_patient("high").spectral
    readings[1].meta = readings[1].meta.model_copy(
        update={"device_id": "other-probe", "synthetic": False}
    )
    files = dict(spectral_inputs_to_files(readings))
    assert set(files) == {"confirm-syn", "other-probe"}
    doc = json.loads(files["other-probe"])
    assert "synthetic" not in doc and doc["device_id"] == "other-probe"
    assert doc["readings"][0]["lit"][0]["gain"] == 64.0
