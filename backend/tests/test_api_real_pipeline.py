"""End-to-end tests against the REAL pipeline (and report engine where noted).

Each test skips cleanly while the module it needs is not importable yet, and starts passing
as the pipeline modules land. No fakes are injected into the bridge here.
"""

from __future__ import annotations

import io
import json
import re

import numpy as np
import pytest
from api_fakes import default_files, default_manifest, make_fake_demo_patient, spectral_csv_bytes
from PIL import Image

from app.services.demo import spectral_inputs_to_files
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import RiskLevel, ScreeningResult, SourceAdapter

pytestmark = pytest.mark.real_pipeline


def require(module: str, attr: str | None = None):
    mod = pytest.importorskip(module)
    if attr is not None and not hasattr(mod, attr):
        pytest.skip(f"{module}.{attr} is not available yet")
    return getattr(mod, attr) if attr else mod


def mouth_like_png(seed: int, *, af: bool = False, size: tuple[int, int] = (320, 240)) -> bytes:
    """A textured pink (WL) or green-glow (AF) image; not realistic, but decodable and busy."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    texture = 0.5 + 0.25 * np.sin(xx / 7.0) * np.cos(yy / 5.0) + 0.1 * rng.random((h, w))
    img = np.zeros((h, w, 3), np.float32)
    if af:
        img[..., 1] = 0.55 * texture
        img[..., 0] = 0.08 * texture
        img[..., 2] = 0.05
    else:
        img[..., 0] = 0.80 * texture + 0.15
        img[..., 1] = 0.45 * texture
        img[..., 2] = 0.45 * texture
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def dark_png(size: tuple[int, int] = (320, 240)) -> bytes:
    arr = np.full((size[1], size[0], 3), 3, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def real_files() -> dict[str, tuple[str, bytes, str]]:
    files = default_files()
    for i, name in enumerate(("wl_bl", "af_bl", "wl_br", "af_br")):
        files[name] = (files[name][0], mouth_like_png(i, af=name.startswith("af")), "image/png")
    files["dark_bl"] = (files["dark_bl"][0], dark_png(), "image/png")
    return files


@pytest.fixture
def real_client(make_client):
    require("mukha_pipeline.orchestrator", "analyze_screening")
    require("mukha_pipeline.capture.upload", "UploadAdapter")
    require("mukha_pipeline.io.spectral", "parse_spectral_file")
    return make_client(fakes=False)


def _clean_text(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


# --------------------------------------------------------------------------------------------
# Contract checks that need only the parsers
# --------------------------------------------------------------------------------------------


def test_real_parser_reads_the_test_spectral_fixture():
    parse = require("mukha_pipeline.io.spectral", "parse_spectral_file")
    parsed = parse(
        spectral_csv_bytes(),
        "spectral.csv",
        source_adapter=SourceAdapter.UPLOAD,
        device_id=None,
        synthetic=False,
        now=None,
    )
    by_id = {r.reading_id: r for r in parsed.readings}
    assert set(by_id) == {"s1", "s2"}
    assert by_id["s1"].reference_reading_id == "s2"
    assert len(by_id["s1"].lit) == 3 and len(by_id["s1"].dark) == 3
    assert by_id["s1"].meta.device_id == "confirm-01"


def test_demo_spectral_serialisation_round_trips_through_the_real_parser():
    parse = require("mukha_pipeline.io.spectral", "parse_spectral_file")
    original = make_fake_demo_patient("high").spectral
    ((device, blob),) = spectral_inputs_to_files(original)
    parsed = parse(
        blob,
        "spectral.json",
        source_adapter=SourceAdapter.SYNTHETIC,
        device_id=None,
        synthetic=True,
        now=None,
    )
    assert parsed.calibration is None
    by_id = {r.reading_id: r for r in parsed.readings}
    for r in original:
        twin = by_id[r.reading_id]
        assert (twin.site, twin.role, twin.reference_reading_id) == (
            r.site,
            r.role,
            r.reference_reading_id,
        )
        assert [f.channels for f in twin.lit] == [f.channels for f in r.lit]
        assert [f.channels for f in twin.dark] == [f.channels for f in r.dark]
        assert [f.gain for f in twin.lit] == [f.gain for f in r.lit]
        assert [f.integration_ms for f in twin.lit] == [f.integration_ms for f in r.lit]
        assert twin.meta.device_id == device == "confirm-syn"
        assert twin.meta.synthetic is True


# --------------------------------------------------------------------------------------------
# Full upload -> analysis -> re-analysis with the real pipeline
# --------------------------------------------------------------------------------------------


def test_real_upload_reanalyze_and_artifacts(real_client):
    res = real_client.post(
        "/api/v1/screenings", data={"manifest": json.dumps(default_manifest())}, files=real_files()
    )
    assert res.status_code == 201, res.text
    body = res.json()
    result = ScreeningResult.model_validate(body["result"])
    assert result.triage.overall_risk_level in set(RiskLevel)
    assert result.triage.requires_clinician_review is True
    assert all(m.validated is False for m in result.versions.models)
    assert not find_positioning_violations(res.text)
    assert {s.site.value for s in result.sites} == {"buccal_mucosa_left", "buccal_mucosa_right"}
    assert result.spectral_readings, "the uploaded spectrum should be analysed"

    sid = body["screening_id"]
    keys = [k for s in result.sites for k in s.artifacts.model_dump().values() if k]
    for key in keys:
        art = real_client.get(f"/api/v1/screenings/{sid}/artifacts/{key}")
        assert art.status_code == 200, key
        if key.endswith(".png"):
            assert art.headers["content-type"] == "image/png"
        elif key.endswith(".json"):
            assert art.headers["content-type"].startswith("application/json")

    again = real_client.post(f"/api/v1/screenings/{sid}/reanalyze")
    assert again.status_code == 201, again.text
    v2 = ScreeningResult.model_validate(again.json()["result"])
    assert v2.result_version == 2 and v2.supersedes_result_id == result.result_id
    assert v2.triage.overall_risk_level == result.triage.overall_risk_level


def test_real_undecodable_image_is_422(real_client):
    files = real_files()
    files["wl_bl"] = ("wl.jpg", b"\xff\xd8\xff garbage that is not a jpeg", "image/jpeg")
    res = real_client.post(
        "/api/v1/screenings", data={"manifest": json.dumps(default_manifest())}, files=files
    )
    assert res.status_code == 422, res.text
    assert res.json()["detail"].startswith("Could not read the images for Left buccal mucosa")
    assert real_client.get("/api/v1/screenings").json() == []


def test_real_bad_spectral_file_is_422(real_client):
    files = real_files()
    files["spectrum"] = ("spectrum.csv", b"reading_id,kind\nonly,lit\n", "text/csv")
    res = real_client.post(
        "/api/v1/screenings", data={"manifest": json.dumps(default_manifest())}, files=files
    )
    assert res.status_code == 422, res.text
    assert res.json()["detail"].startswith("Could not read the spectral file")


def test_real_version_endpoint_reports_pipeline_models(real_client):
    models = {m["component"]: m for m in real_client.get("/api/v1/version").json()["models"]}
    assert models["wl_lesion_classifier"]["status"] == "model_unavailable"
    assert "not used in triage" in models["wl_lesion_classifier"]["notes"]
    assert models["spectral_classifier"]["status"] == "placeholder"
    assert all(m["validated"] is False for m in models.values())


# --------------------------------------------------------------------------------------------
# Synthetic demo + real report engine
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["low", "moderate", "high"])
def test_real_demo_end_to_end(real_client, kind):
    require("mukha_pipeline.synthetic", "make_demo_patient")
    res = real_client.post(f"/api/v1/demo/screenings/{kind}")
    assert res.status_code == 201, res.text
    result = ScreeningResult.model_validate(res.json()["result"])
    assert result.synthetic is True and result.data_provenance == "synthetic"
    assert result.notices.synthetic_notice
    assert not find_positioning_violations(res.text)
    sid = res.json()["screening_id"]
    again = real_client.post(f"/api/v1/screenings/{sid}/reanalyze")
    assert again.status_code == 201, again.text
    v2 = ScreeningResult.model_validate(again.json()["result"])
    assert v2.supersedes_result_id == result.result_id
    assert v2.triage.overall_risk_level == result.triage.overall_risk_level
    # The stored calibration (incl. its violet-light flat field) is used again as it was.
    assert [lim.code for lim in v2.limitations] == [lim.code for lim in result.limitations]
    assert "af_flat_field_missing" not in {lim.code for lim in v2.limitations}


def test_real_report_pdf_html_and_site_map(make_client):
    require("mukha_pipeline.orchestrator", "analyze_screening")
    require("mukha_pipeline.capture.upload", "UploadAdapter")
    reports = require("app.reports")
    from app.services.reports import AppReportsRenderer

    client = make_client(fakes=False)
    client.app.state.services.reports.renderer = AppReportsRenderer()
    assert reports.render_clinician_report_pdf
    res = client.post(
        "/api/v1/screenings", data={"manifest": json.dumps(default_manifest())}, files=real_files()
    )
    assert res.status_code == 201, res.text
    sid = res.json()["screening_id"]

    pdf = client.get(f"/api/v1/screenings/{sid}/report.pdf")
    assert pdf.status_code == 200, pdf.text
    assert pdf.content.startswith(b"%PDF")
    assert client.get(f"/api/v1/screenings/{sid}/report.pdf").content == pdf.content

    html = client.get(f"/api/v1/screenings/{sid}/report.html")
    assert html.status_code == 200
    text = _clean_text(html.text)
    assert "Research prototype" in text
    assert not find_positioning_violations(text)

    svg = client.get(f"/api/v1/screenings/{sid}/sitemap.svg")
    assert svg.status_code == 200 and "<svg" in svg.text
