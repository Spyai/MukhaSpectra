"""Regression tests for review findings in the upload / ingest path.

* security.1 / correctness.2: uploads are decoded at 2048 px and a per-request decoded-pixel
  budget is enforced (413).
* security.4: GPS / owner EXIF, XMP and PNG text are stripped from the stored JPEG / PNG.
* security.5: free-text manifest fields have length limits (422, not a database error), and the
  engine never logs bound parameters.
* security.7 / honesty.10: recorded consent needs a real method; the server stamps the time.
* correctness.4: naive times are camp-local (settings.default_timezone) and stored as UTC.
* correctness.5: NaN / Infinity in the manifest are rejected.
* honesty.1: SYNTHETIC files are detected even when the manifest says "real".
* honesty.4: every result records the build that produced it.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from api_fakes import (
    PII_MARKERS,
    default_files,
    default_manifest,
    png_bytes,
    upload,
)
from PIL import Image, PngImagePlugin, TiffImagePlugin
from pydantic import ValidationError
from sqlalchemy import func, select

from app import __version__
from app.buildinfo import resolve_build_id, software_version
from app.db import create_db_engine
from app.models import AuditEvent, Capture, Screening
from app.services.ingest import CONSENT_METHOD_REQUIRED, CONSENT_REQUIRED
from app.services.pipeline_bridge import (
    default_analyze,
    default_load_image,
    default_upload_adapter,
)
from app.settings import Settings
from mukha_pipeline.schema.models import ScreeningResult

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "sample_data" / "synthetic" / "high"


def _count(db, model) -> int:
    return db.scalar(select(func.count()).select_from(model)) or 0


def _stored_files(services) -> list[str]:
    root = services.storage.root
    return [p.as_posix() for p in root.rglob("*") if p.is_file()] if root.exists() else []


# --------------------------------------------------------------------------------------------
# security.1 / correctness.2: decode budget
# --------------------------------------------------------------------------------------------


def test_default_upload_adapter_and_reanalysis_decode_at_2048_px():
    assert default_upload_adapter().max_side == 2048
    big = io.BytesIO()
    Image.new("RGB", (3000, 2000), (120, 60, 60)).save(big, "PNG")
    image = default_load_image(big.getvalue(), "x.png")
    assert image.shape == (1365, 2048, 3)


def test_decoded_pixel_budget_is_413_and_nothing_is_stored(make_client):
    client = make_client(max_decoded_megapixels=1)
    services = client.app.state.services
    files = default_files()
    for name in ("wl_bl", "af_bl", "dark_bl", "wl_br", "af_br"):
        files[name] = (f"{name}.png", png_bytes(size=(600, 600)), "image/png")
    res = upload(client, files=files)
    assert res.status_code == 413, res.text
    assert "1 MP" in res.json()["detail"]
    with services.session_factory() as session:
        assert _count(session, Screening) == 0 and _count(session, Capture) == 0
    assert _stored_files(services) == []


def test_budget_default_fits_the_largest_legitimate_upload():
    """Every catalogued site with WL + AF + dark frame, each decoded at the full 2048 px square,
    must fit: a full-mouth screening is legitimate and must not get 413."""
    from mukha_pipeline.capture.upload import UPLOAD_DECODE_MAX_SIDE
    from mukha_pipeline.sites import OralSite

    largest = len(OralSite) * 3 * UPLOAD_DECODE_MAX_SIDE**2
    budget = Settings().max_decoded_pixels
    assert budget >= largest
    assert budget <= 1.1 * largest  # still a bound: about 2.2 GB of float32 RGB per upload


# --------------------------------------------------------------------------------------------
# security.4: identifying metadata is not stored
# --------------------------------------------------------------------------------------------


def _jpeg_with_gps() -> bytes:
    exif = Image.Exif()
    exif[315] = "Asha Ramesh (patient)"
    exif[274] = 6
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = tuple(TiffImagePlugin.IFDRational(v, 1) for v in (12, 58, 30))
    buf = io.BytesIO()
    Image.new("RGB", (48, 32), (200, 90, 90)).save(buf, "JPEG", exif=exif, quality=95)
    return buf.getvalue()


def _png_with_text() -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "Asha Ramesh")
    buf = io.BytesIO()
    Image.new("RGB", (48, 32), (40, 160, 40)).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def test_gps_and_owner_metadata_are_stripped_before_storage(client, services, db):
    files = default_files()
    files["wl_bl"] = ("wl.jpg", _jpeg_with_gps(), "image/jpeg")
    files["af_bl"] = ("af.png", _png_with_text(), "image/png")
    res = upload(client, files=files)
    assert res.status_code == 201, res.text
    sid = res.json()["screening_id"]
    captures = db.scalars(select(Capture).where(Capture.screening_id == sid)).all()
    wl = next(c for c in captures if c.site == "buccal_mucosa_left" and c.kind == "white_light")
    af = next(
        c for c in captures if c.site == "buccal_mucosa_left" and c.kind == "autofluorescence"
    )
    for capture, original in ((wl, files["wl_bl"][1]), (af, files["af_bl"][1])):
        stored = services.storage.get(capture.storage_key)
        assert b"asha" not in stored.lower() and b"ramesh" not in stored.lower()
        assert capture.sha256 != capture.meta["capture_metadata"]["file_sha256"]
        assert capture.meta["capture_metadata"]["file_sha256"] == _sha(original)
        assert capture.meta["metadata_removed"]
        with Image.open(io.BytesIO(stored)) as img:
            assert not img.getexif().get_ifd(0x8825)
            assert 315 not in img.getexif()
    stored_wl = services.storage.get(wl.storage_key)
    with Image.open(io.BytesIO(stored_wl)) as img:
        assert img.getexif().get(274) == 6  # orientation kept, so re-analysis sees the same view
    # the compressed image data is untouched: identical pixels before and after
    before = np.asarray(Image.open(io.BytesIO(files["wl_bl"][1])).convert("RGB"))
    after = np.asarray(Image.open(io.BytesIO(stored_wl)).convert("RGB"))
    assert np.array_equal(before, after)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------------------------
# security.5: lengths and logging
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "field", "length", "location"),
    [
        ("context", "screening_site_name", 201, "manifest.context.screening_site_name"),
        ("context", "camp_id", 101, "manifest.context.camp_id"),
        ("context", "health_worker_id", 101, "manifest.context.health_worker_id"),
        ("context", "health_worker_name", 201, "manifest.context.health_worker_name"),
        ("consent", "consent_id", 101, "manifest.consent.consent_id"),
    ],
)
def test_long_free_text_is_422(client, services, db, section, field, length, location):
    manifest = default_manifest()
    manifest[section][field] = "x" * length
    res = upload(client, manifest)
    assert res.status_code == 422, res.text
    assert location in res.json()["detail"]
    assert _count(db, Screening) == 0


def test_long_device_lists_are_422(client):
    manifest = default_manifest()
    manifest["context"]["device_ids"] = ["d" * 101]
    assert upload(client, manifest).status_code == 422
    manifest["context"]["device_ids"] = [f"d{i}" for i in range(100)]
    assert upload(client, manifest).status_code == 422


def test_limits_match_the_database_columns(client, db):
    manifest = default_manifest()
    manifest["context"].update(
        screening_site_name="s" * 200,
        camp_id="c" * 100,
        health_worker_id="h" * 100,
        health_worker_name="n" * 200,
    )
    assert upload(client, manifest).status_code == 201


def test_engines_hide_bound_parameters(monkeypatch):
    assert create_db_engine("sqlite://").hide_parameters is True
    import app.db as db_module

    seen = {}
    monkeypatch.setattr(db_module, "create_engine", lambda url, **kw: seen.update(kw))
    create_db_engine("postgresql+psycopg://u:p@localhost/x")  # driver not needed here
    assert seen["hide_parameters"] is True


# --------------------------------------------------------------------------------------------
# security.7 / honesty.10: consent
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "consent",
    [
        {"consent_recorded": True},
        {"consent_recorded": True, "method": "none"},
        {"consent_recorded": True, "method": "verbal"},
    ],
)
def test_recorded_consent_needs_a_method(client, services, db, consent, fake_pipeline):
    res = upload(client, default_manifest(consent=consent))
    assert res.status_code == 422
    assert res.json() == {"detail": CONSENT_METHOD_REQUIRED}
    assert _count(db, Screening) == 0 and _stored_files(services) == []
    assert fake_pipeline.adapter_calls == []


def test_missing_consent_still_wins(client):
    res = upload(client, default_manifest(consent={"consent_recorded": False, "method": "none"}))
    assert res.json() == {"detail": CONSENT_REQUIRED}


def test_server_stamps_the_consent_time(client, db):
    before = datetime.now(UTC)
    res = upload(client)
    assert res.status_code == 201
    stamped = datetime.fromisoformat(res.json()["result"]["consent"]["recorded_at"])
    assert stamped.tzinfo is not None and stamped >= before.replace(microsecond=0)
    screening = db.get(Screening, res.json()["screening_id"])
    assert screening.consent["recorded_at"] is not None
    created = db.scalars(select(AuditEvent).where(AuditEvent.action == "screening_created")).one()
    assert created.detail["consent_method"] == "checkbox"
    assert created.detail["consent_recorded_at_source"] == "server"


def test_client_consent_time_is_kept_and_normalised(client):
    consent = {**default_manifest()["consent"], "recorded_at": "2026-10-14T09:00:00"}
    res = upload(client, default_manifest(consent=consent))
    assert res.json()["result"]["consent"]["recorded_at"] in (
        "2026-10-14T03:30:00Z",
        "2026-10-14T03:30:00+00:00",
    )


def test_service_rechecks_consent(services, db):
    from app.errors import InvalidInputError
    from app.schemas import UploadManifest

    manifest = UploadManifest.model_validate(default_manifest())
    manifest.consent = manifest.consent.model_copy(update={"method": "none"})
    with pytest.raises(InvalidInputError):
        services.ingest.create_screening(db, manifest, {})


# --------------------------------------------------------------------------------------------
# correctness.4: camp-local time
# --------------------------------------------------------------------------------------------


def test_naive_capture_time_is_camp_local_and_stored_as_utc(client, db, fake_pipeline):
    manifest = default_manifest()
    manifest["context"]["captured_at"] = "2026-10-14T10:00:00"
    res = upload(client, manifest)
    assert res.status_code == 201, res.text
    expected = datetime(2026, 10, 14, 4, 30, tzinfo=UTC)
    result = ScreeningResult.model_validate(res.json()["result"])
    assert result.context.captured_at == expected
    (inp,) = fake_pipeline.analyses
    assert inp.now == expected and inp.context.captured_at == expected
    screening = db.get(Screening, res.json()["screening_id"])
    assert datetime.fromisoformat(screening.context["captured_at"]) == expected


def test_aware_capture_time_is_converted_to_utc(client):
    manifest = default_manifest()
    manifest["context"]["captured_at"] = "2026-10-14T10:00:00+05:30"
    res = upload(client, manifest)
    result = ScreeningResult.model_validate(res.json()["result"])
    assert result.context.captured_at == datetime(2026, 10, 14, 4, 30, tzinfo=UTC)


def test_default_timezone_is_configurable_and_validated(make_client, fake_pipeline):
    client = make_client(default_timezone="UTC")
    manifest = default_manifest()
    manifest["context"]["captured_at"] = "2026-10-14T10:00:00"
    upload(client, manifest)
    assert fake_pipeline.analyses[-1].now == datetime(2026, 10, 14, 10, 0, tzinfo=UTC)
    assert Settings().default_timezone == "Asia/Kolkata"
    with pytest.raises(ValidationError):
        Settings(default_timezone="Mars/Olympus_Mons")


# --------------------------------------------------------------------------------------------
# correctness.5: non-finite numbers
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["Infinity", "-Infinity", "NaN", "1e999"])
def test_non_finite_numbers_are_rejected(client, services, db, token):
    text = json.dumps(default_manifest()).replace('"years": 12', f'"years": {token}')
    assert token in text
    res = client.post("/api/v1/screenings", data={"manifest": text}, files=default_files())
    assert res.status_code == 422, res.text
    assert "finite" in res.json()["detail"]
    assert _count(db, Screening) == 0


# --------------------------------------------------------------------------------------------
# honesty.1: SYNTHETIC uploads
# --------------------------------------------------------------------------------------------


def test_synthetic_file_name_marks_the_screening(client, db, fake_pipeline):
    files = default_files()
    files["wl_bl"] = ("SYNTHETIC_buccal_mucosa_left_wl.png", files["wl_bl"][1], "image/png")
    res = upload(client, files=files)
    assert res.status_code == 201, res.text
    result = res.json()["result"]
    assert result["synthetic"] is True
    assert result["data_provenance"] == "mixed"
    screening = db.get(Screening, res.json()["screening_id"])
    assert screening.synthetic is True and screening.data_provenance == "mixed"
    # the uploaded name is used for detection only and never stored
    names = {c.file_name for c in db.scalars(select(Capture))}
    assert "SYNTHETIC_buccal_mucosa_left_wl.png" not in names
    assert all("asha" not in (n or "").lower() for n in names)


def test_sample_data_without_provenance_is_labelled_synthetic(make_client, fake_pipeline):
    """The review's failure scenario, with the REAL upload adapter: the shipped synthetic
    sample uploaded with ``data_provenance`` left out."""
    if not SAMPLE.is_dir():
        pytest.skip("sample data not generated")
    client = make_client()
    services = client.app.state.services
    services.ingest.bridge = services.ingest.bridge.replace(
        upload_adapter_factory=default_upload_adapter
    )
    sites = []
    files = {}
    for site in ("buccal_mucosa_left", "buccal_mucosa_right"):
        entry = {"site": site, "wl": f"{site}_wl", "af": f"{site}_af"}
        for slot in ("wl", "af"):
            path = SAMPLE / f"SYNTHETIC_{site}_{slot}.jpg"
            # Renamed: detection must not depend on the file name alone.
            files[f"{site}_{slot}"] = (f"IMG_{slot}.jpg", path.read_bytes(), "image/jpeg")
        sites.append(entry)
    body = {
        "consent": {"consent_recorded": True, "method": "checkbox"},
        "sites": sites,
    }
    assert "data_provenance" not in body
    res = client.post("/api/v1/screenings", data={"manifest": json.dumps(body)}, files=files)
    assert res.status_code == 201, res.text
    result = res.json()["result"]
    assert result["synthetic"] is True and result["data_provenance"] == "synthetic"
    assert result["patient"]["pseudonymous_id"].startswith("SYN-")
    for site in result["sites"]:
        for capture in site["captures"]:
            assert capture["metadata"]["synthetic"] is True
            assert capture["metadata"]["extra"]["synthetic_marker"]


def test_synthetic_report_file_name_is_labelled(client):
    sid = client.post("/api/v1/demo/screenings/low").json()["screening_id"]
    res = client.get(f"/api/v1/screenings/{sid}/report.pdf")
    assert res.status_code == 200
    assert 'filename="SYNTHETIC_MukhaSpectra_' in res.headers["content-disposition"]


# --------------------------------------------------------------------------------------------
# honesty.4: build identifier
# --------------------------------------------------------------------------------------------


def test_results_record_the_build(make_client, fake_pipeline):
    client = make_client(build_id="abc1234")
    res = upload(client)
    assert res.json()["result"]["versions"]["software"] == f"{__version__}+abc1234"
    assert fake_pipeline.software_versions == [f"{__version__}+abc1234"]
    body = client.get("/api/v1/version").json()
    assert body["software"] == f"{__version__}+abc1234"
    assert body["build"] == "abc1234"


def test_build_id_comes_from_env_then_git_then_unknown(monkeypatch):
    resolve_build_id.cache_clear()
    try:
        assert resolve_build_id("given") == "given"

        def no_git(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", no_git)
        resolve_build_id.cache_clear()
        assert resolve_build_id(None) == "unknown"

        def fake_git(cmd, **kwargs):
            out = "abc1234\n" if "rev-parse" in cmd else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_git)
        resolve_build_id.cache_clear()
        assert resolve_build_id(None) == "abc1234"
        assert software_version("abc1234") == f"{__version__}+abc1234"
    finally:
        resolve_build_id.cache_clear()


def test_invalid_build_ids_are_rejected():
    with pytest.raises(ValidationError):
        Settings(build_id="has spaces; rm -rf")


def test_default_bridge_passes_the_software_version(monkeypatch):
    import mukha_pipeline.orchestrator as orchestrator

    seen = {}

    def fake(inp, *, classifier=None, software_version=None):
        seen["software_version"] = software_version
        return "out"

    monkeypatch.setattr(orchestrator, "analyze_screening", fake)
    assert default_analyze(object(), software_version="0.1.0+abc") == "out"  # type: ignore[arg-type]
    assert seen["software_version"] == "0.1.0+abc"


def test_audit_events_carry_no_pii_after_new_fields(client, db):
    upload(client)
    for event in db.scalars(select(AuditEvent)):
        text = json.dumps(event.detail).lower()
        assert not any(marker in text for marker in PII_MARKERS)


def test_reanalysis_decodes_like_the_upload_and_keeps_px_per_mm(make_client, fake_pipeline):
    """Real decoder on both paths: v1 and v2 see the same 2048 px pixels and the same scale."""
    client = make_client()
    services = client.app.state.services
    services.ingest.bridge = services.ingest.bridge.replace(
        upload_adapter_factory=default_upload_adapter
    )
    services.capture_store.bridge = services.capture_store.bridge.replace(
        load_image=default_load_image
    )
    big = io.BytesIO()
    yy, xx = np.mgrid[0:2000, 0:3000]
    arr = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    Image.fromarray(arr).save(big, "JPEG", quality=90)
    files = default_files()
    for name in ("wl_bl", "af_bl", "dark_bl", "wl_br", "af_br"):
        files[name] = (f"{name}.jpg", big.getvalue(), "image/jpeg")
    manifest = default_manifest()
    manifest["sites"][0]["px_per_mm"] = 30.0
    res = upload(client, manifest, files)
    assert res.status_code == 201, res.text
    sid = res.json()["screening_id"]
    assert client.post(f"/api/v1/screenings/{sid}/reanalyze").status_code == 201
    first, second = fake_pipeline.analyses
    assert first.sites[0].wl.image.shape == (1365, 2048, 3)
    assert first.sites[0].px_per_mm == pytest.approx(30.0 * 2048 / 3000)
    assert second.sites[0].px_per_mm == pytest.approx(first.sites[0].px_per_mm)
    for a, b in zip(first.sites, second.sites, strict=True):
        assert np.array_equal(a.wl.image, b.wl.image)
        assert np.array_equal(a.af.image, b.af.image)
    meta = first.sites[0].wl.meta
    assert (meta.width_px, meta.height_px) == (3000, 2000)
    assert meta.extra["decode_scale"] == pytest.approx(2048 / 3000, abs=1e-6)
