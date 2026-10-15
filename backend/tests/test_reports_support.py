"""Facilities config, wording helpers and view-model rules of the report engine."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from report_fixtures import cached_case, default_options

from app.reports import DEFAULT_FACILITIES_PATH, Facility, load_facilities
from app.reports import labels as lb
from app.reports.context import build_context
from app.reports.ranking import lesion_numbers, ordered_lesions
from mukha_pipeline.schema.models import (
    CalibrationInfo,
    CalibrationStatus,
    ModelComponent,
    ModelInfo,
    ModelStatus,
    NormPoint,
    QCIssue,
    QCSeverity,
    RiskFactors,
    WLAnalysis,
)

# --------------------------------------------------------------------------------------------
# Facilities
# --------------------------------------------------------------------------------------------


def test_default_facilities_are_clearly_marked_examples():
    facilities = load_facilities()
    assert 2 <= len(facilities) <= 3
    assert DEFAULT_FACILITIES_PATH.name == "facilities.yaml"
    for f in facilities:
        assert isinstance(f, Facility)
        assert f.example is True
        assert f.name.startswith("Example")
        assert f.phone == "+91-XXXXX-XXXXX"
        assert not re.search(r"\d{5}", f.phone or "")  # no real-looking numbers


def test_custom_facilities_file(tmp_path: Path):
    path = tmp_path / "f.yaml"
    path.write_text(
        "facilities:\n"
        "  - name: District Hospital\n"
        "    address: Main Road\n"
        "    hours: 9-1\n"
        "  - {name: Clinic B, address: Market St, phone: '+91-XXXXX-XXXXX', example: true}\n",
        encoding="utf-8",
    )
    first, second = load_facilities(path)
    assert first == Facility(name="District Hospital", address="Main Road", hours="9-1")
    assert second.example and second.phone == "+91-XXXXX-XXXXX"


def test_facilities_list_form_and_empty(tmp_path: Path):
    path = tmp_path / "f.yaml"
    path.write_text("- {name: A, address: B}\n", encoding="utf-8")
    assert load_facilities(path) == [Facility(name="A", address="B")]
    path.write_text("", encoding="utf-8")
    assert load_facilities(path) == []


@pytest.mark.parametrize(
    "content",
    [
        "facilities:\n  - {name: A}\n",
        "facilities:\n  - {name: A, address: B, fax: C}\n",
        "facilities: nope\n",
        "facilities:\n  - just-a-string\n",
        "42\n",
        "facilities:\n  - {name: A, address: [1, 2]}\n",
    ],
)
def test_bad_facilities_are_rejected(tmp_path: Path, content: str):
    path = tmp_path / "f.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_facilities(path)


def test_facility_levels_and_capabilities(tmp_path: Path):
    path = tmp_path / "f.yaml"
    path.write_text(
        "facilities:\n"
        "  - {name: A, address: B, levels: [high, MODERATE], capabilities: [biopsy]}\n"
        "  - {name: C, address: D, levels: HIGH, capabilities: ENT}\n",
        encoding="utf-8",
    )
    a, c = load_facilities(path)
    assert a.levels == ["HIGH", "MODERATE"] and a.capabilities == ["biopsy"]
    assert c.levels == ["HIGH"] and c.capabilities == ["ENT"]
    for bad in ("levels: [URGENT]", "levels: [1, 2]", "capabilities: {x: 1}"):
        path.write_text(f"facilities:\n  - {{name: A, address: B, {bad}}}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_facilities(path)


def test_default_facilities_route_by_level():
    by_name = {f.name: f for f in load_facilities()}
    centre = next(f for name, f in by_name.items() if "Cancer Centre" in name)
    district = next(f for name, f in by_name.items() if "District Hospital" in name)
    assert centre.levels == ["HIGH"]
    assert district.levels == []  # every level


def test_missing_explicit_facilities_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_facilities(tmp_path / "missing.yaml")


# --------------------------------------------------------------------------------------------
# Wording and numbers
# --------------------------------------------------------------------------------------------


def test_number_formatting():
    assert lb.num(0.5234) == "0.52"
    assert lb.num(-0.088, 3) == "−0.088"
    assert lb.num(None) == "—" and lb.num(math.nan) == "—"
    assert lb.signed(1.5) == "+1.5" and lb.signed(-0.9) == "−0.9" and lb.signed(0.0) == "0.0"
    assert lb.signed(None) == "—"
    assert lb.pct(0.072) == "7.2 %"
    assert lb.short_sha("abcdef1234567890") == "abcdef12"
    assert lb.short_sha(None) == "—"


def test_dates_and_timelines():
    ist = timezone(timedelta(hours=5, minutes=30), "IST")
    assert lb.fmt_datetime(datetime(2026, 10, 14, 9, 45, tzinfo=ist)) == "14 Oct 2026, 09:45 IST"
    assert lb.fmt_datetime(datetime(2026, 10, 14, 4, 5, tzinfo=UTC)).endswith("UTC")
    other = timezone(timedelta(hours=2))
    assert lb.fmt_datetime(datetime(2026, 1, 2, 3, 4, tzinfo=other)).endswith("(UTC+02:00)")
    assert lb.fmt_datetime(None) == "—"
    assert lb.fmt_date(datetime(2026, 1, 2)) == "02 Jan 2026"
    assert lb.timeline_text(0) == "Same day (at the screening visit)"
    assert lb.timeline_text(14) == "Within 2 weeks (14 days)"
    start = datetime(2026, 10, 14, 9, 35)
    assert lb.timeline_text(14, start=start) == "Within 2 weeks — by 28 Oct 2026"
    assert lb.timeline_text(365, start=start) == "In 12 months — by 14 Oct 2027"
    assert lb.timeline_text(3, start=start) == "Within 3 days — by 17 Oct 2026"
    assert lb.timeline_text(0, start=start) == "Same day (at the screening visit)"
    assert lb.timeline_text(365) == "In 12 months"
    assert lb.timeline_text(3) == "Within 3 days"
    assert lb.timeline_text(1) == "Within 1 day"
    assert lb.timeline_text(None) == "No fixed timeline"


def test_position_text():
    assert lb.position_text(NormPoint(x=0.5, y=0.5)) == "In photo: centre"
    assert lb.position_text(NormPoint(x=0.1, y=0.1)) == "In photo: upper left"
    assert lb.position_text(NormPoint(x=0.9, y=0.9)) == "In photo: lower right"
    assert lb.position_text(None) == "—"


def test_model_status_wording_is_honest():
    def info(component, status):
        return ModelInfo(component=component, name="m", version="1", status=status)

    assert lb.model_status_text(
        info(ModelComponent.WL_LESION_CLASSIFIER, ModelStatus.MODEL_UNAVAILABLE)
    ) == ("Not available — not used in triage")
    assert (
        lb.model_status_text(info(ModelComponent.SEGMENTER, ModelStatus.MODEL_UNAVAILABLE))
        == "Not available — not used"
    )
    assert lb.model_status_text(
        info(ModelComponent.SPECTRAL_CLASSIFIER, ModelStatus.PLACEHOLDER)
    ) == ("Placeholder heuristic, not validated")
    assert lb.model_status_text(info(ModelComponent.FUSION, ModelStatus.BASELINE_RULES)) == (
        "Baseline heuristic, unvalidated"
    )
    assert "not clinically validated" in lb.model_status_text(
        info(ModelComponent.WL_LESION_CLASSIFIER, ModelStatus.AVAILABLE)
    )


def test_risk_factor_summary_variants():
    assert lb.risk_factor_summary(RiskFactors()) == "No risk factors or symptoms reported"
    rf = RiskFactors.model_validate(
        {
            "smokeless_tobacco": {"former": True, "types": ["khaini"], "years": 10},
            "smoking": {"uses": True, "types": ["chutta_reverse"]},
            "areca_nut": {"uses": True, "years": 5, "per_day": 3},
            "alcohol": {"uses": True, "frequency": "occasional"},
            "prior_oral_cancer": True,
            "prior_oral_lesion": True,
            "symptoms": {"neck_lump": True, "difficulty_opening_mouth": True},
            "irritants": {"sharp_tooth": True, "ill_fitting_denture": True},
        }
    )
    summary = lb.risk_factor_summary(rf)
    assert summary.startswith("Former khaini 10 y")
    for part in [
        "chutta (reverse smoking)",
        "areca nut (supari) 5 y (3/day)",
        "alcohol occasionally",
        "lump in the neck",
        "difficulty opening the mouth",
        "previous oral cancer",
        "previous oral lesion",
        "sharp tooth",
        "ill-fitting denture",
    ]:
        assert part in summary, part
    assert "; " in summary and " · " in summary  # groups by "; ", items by " · "
    groups = lb.risk_factor_groups(rf)
    assert set(groups) == {"Habits", "Symptoms", "History", "Irritants"}
    only_per_day = RiskFactors.model_validate(
        {"smokeless_tobacco": {"uses": True, "types": ["gutka"], "per_day": 4}}
    )
    assert lb.risk_factor_summary(only_per_day) == "Gutka (4/day)"


@pytest.mark.parametrize(
    ("raw", "masked"),
    [
        ("91-2345-6789-0123", "XX-XXXX-XXXX-0123"),
        ("ABHA 91-2345-6789-0123", "XXXX XX-XXXX-XXXX-0123"),
        ("HSK-0042", "XXX-0042"),
        ("123456789012", "XXXXXXXX9012"),
        ("A12", "XX2"),
        ("", ""),
        (None, None),
    ],
)
def test_mask_id_keeps_last_four(raw, masked):
    assert lb.mask_id(raw) == masked


def test_wording_tables_cover_every_enum_value():
    from mukha_pipeline.schema.models import (
        CalibrationStatus,
        ConfounderType,
        Diascopy,
        FactorSource,
        LesionOrigin,
        ReferenceType,
        ReferralUrgency,
        RiskLevel,
        WLLabel,
    )

    for enum, table in [
        (CalibrationStatus, lb.CALIBRATION_TEXT), (ConfounderType, lb.CONFOUNDER_TEXT),
        (Diascopy, lb.DIASCOPY_TEXT), (FactorSource, lb.SOURCE_TEXT),
        (LesionOrigin, lb.ORIGIN_TEXT), (ReferenceType, lb.REFERENCE_TEXT),
        (ReferralUrgency, lb.URGENCY_TEXT), (RiskLevel, lb.RISK_TITLE), (RiskLevel, lb.RISK_COLORS),
        (ModelComponent, lb.COMPONENT_TEXT), (WLLabel, lb.WL_LABEL_TEXT),
    ]:  # fmt: skip
        assert set(table) == set(enum), enum


# --------------------------------------------------------------------------------------------
# Ranking and view model
# --------------------------------------------------------------------------------------------


def test_lesions_ordered_by_severity_then_score():
    result = cached_case("high").result
    assert [les.lesion_id for les in result.lesions] == ["les-002", "les-001"]
    assert [les.lesion_id for les in ordered_lesions(result)] == ["les-001", "les-002"]
    assert lesion_numbers(result) == {"les-001": 1, "les-002": 2}


def test_context_wl_model_available_branch():
    case = cached_case("high")
    site = case.result.sites[0]
    model = site.wl.model.model_dump() | {"status": "available", "version": "0.3.0"}
    wl = WLAnalysis.model_validate(
        {
            "status": "ok",
            "model": model,
            "predictions": [
                {"label": "white_patch", "probability": 0.62},
                {"label": "normal", "probability": 0.30},
            ],
            "top_label": "white_patch",
            "saliency_artifact": "sites/buccal_mucosa_left/heatmap_overlay.png",
        }
    )
    sites = [site.model_copy(update={"wl": wl}), *case.result.sites[1:]]
    result = case.result.model_copy(update={"sites": sites})
    ctx = build_context(result, case.reader, identity=None, options=default_options(case))
    view = ctx["lesions"][0].wl
    assert view.available
    assert view.predictions[0] == ("White patch", "0.62")
    assert "not clinically validated" in view.status_text
    assert view.saliency is not None and view.saliency.available


@pytest.mark.parametrize("status", ["skipped_qc_failed", "error"])
def test_context_wl_not_used_branches(status: str):
    case = cached_case("high")
    site = case.result.sites[0]
    model = site.wl.model.model_copy(update={"status": ModelStatus.AVAILABLE})
    wl = site.wl.model_copy(update={"status": status, "model": model})
    sites = [site.model_copy(update={"wl": wl}), *case.result.sites[1:]]
    result = case.result.model_copy(update={"sites": sites})
    ctx = build_context(result, case.reader, identity=None, options=default_options(case))
    view = ctx["lesions"][0].wl
    assert not view.available
    assert "not used" in view.status_text


def test_context_spectral_notes_and_unlinked_readings():
    case = cached_case("high")
    lesion_reading, reference = case.result.spectral_readings
    failed_ref = reference.model_copy(
        update={
            "qc": reference.qc.model_copy(update={"passed": False}),
            "signal_type": "relative_signal",
        }
    )
    orphan = lesion_reading.model_copy(
        update={"reading_id": "s9", "reference_reading_id": None, "classifier": None}
    )
    result = case.result.model_copy(
        update={"spectral_readings": [lesion_reading, failed_ref, orphan]}
    )
    ctx = build_context(result, case.reader, identity=None, options=default_options(case))
    notes = ctx["lesions"][0].spectral[0].notes
    assert any("different signal types" in n for n in notes)
    assert any("failed QC" in n for n in notes)
    unlinked = ctx["unlinked_spectral"]
    assert [s.reading_id for s in unlinked] == ["s9"]
    assert unlinked[0].classifier_status == "Not run"
    assert any("No reference reading" in n for n in unlinked[0].notes)


def test_context_level_note_when_red_flag_raises_level():
    case = cached_case("high")
    triage = case.result.triage.model_copy(update={"score": 4.0})
    result = case.result.model_copy(update={"triage": triage})
    ctx = build_context(result, case.reader, identity=None, options=default_options(case))
    assert ctx["triage"]["level_note"] == (
        "The rule score alone gives moderate risk; a clinical red flag raises the level to "
        "high risk."
    )


def test_context_visual_only_lesion_has_region_note():
    case = cached_case("high")
    lesion = case.result.lesions[1].model_copy(
        update={"origins": ["visual"], "region_ids": [], "spectral_reading_ids": []}
    )
    result = case.result.model_copy(update={"lesions": [lesion]})
    ctx = build_context(result, case.reader, identity=None, options=default_options(case))
    view = ctx["lesions"][0]
    assert view.regions == []
    assert "raised by health-worker observation" in view.region_note


def test_context_edge_cases_degrade_gracefully():
    """Unregistered AF, failed + uncalibrated spectra, no WL record, sparse context."""
    case = cached_case("high")
    r = case.result
    site0 = r.sites[0]
    arts = site0.artifacts.model_copy(update={"af_registered": None})
    sites = [site0.model_copy(update={"artifacts": arts, "wl": None}), *r.sites[1:]]
    issue = QCIssue(
        code="spectral_saturated",
        severity=QCSeverity.ERROR,
        message="Probe reading saturated — reduce gain and repeat.",
        message_key="qc.spectral_saturated",
    )
    lesion_reading, reference = r.spectral_readings
    bad = lesion_reading.model_copy(
        update={
            "qc": lesion_reading.qc.model_copy(update={"passed": False, "issues": [issue]}),
            "signal_type": "relative_signal",
            "calibration": CalibrationInfo(status=CalibrationStatus.UNCALIBRATED),
            "delta_vs_reference": None,
        }
    )
    orphan_issue = QCIssue(
        code="other",
        severity=QCSeverity.WARNING,
        message="Session note from the device.",
        message_key="qc.other",
    )
    result = r.model_copy(
        update={
            "sites": sites,
            "spectral_readings": [bad, reference],
            "qc_summary": r.qc_summary.model_copy(
                update={"issues": [*r.qc_summary.issues, orphan_issue]}
            ),
            "context": r.context.model_copy(update={"health_worker_name": None}),
            "consent": r.consent.model_copy(update={"consent_recorded": False}),
            "versions": r.versions.model_copy(
                update={
                    "models": [
                        m
                        for m in r.versions.models
                        if m.component != ModelComponent.WL_LESION_CLASSIFIER
                    ]
                }
            ),
        }
    )
    ctx = build_context(result, case.reader, identity=None, options=default_options(case))
    first = ctx["lesions"][0]
    assert first.panels[1].note == "405 nm, unregistered"
    assert first.panels[1].image.available  # raw AF still shown, without region brackets
    assert "not run for this site" in first.wl.status_text
    spectral = first.spectral[0]
    assert spectral.qc_text.startswith("Reading failed QC: Probe reading saturated")
    assert "relative signal (uncalibrated" in spectral.calibration_text
    # No pipeline delta: no Δ (a plain lesion − reference difference depends on the probe).
    assert spectral.features[0].delta == "—"
    messages = [i.message for i in ctx["qc"]["issues"]]
    assert "Probe reading saturated — reduce gain and repeat." in messages
    assert "Session note from the device." in messages
    assert ctx["header"]["health_worker"] == "hw-17"
    assert ctx["patient"]["consent"] == "Not recorded"
    assert ctx["models_line"] == (
        "all unvalidated · White-light classifier: not available — not used in triage (see §5)"
    )
    assert ctx["model_rows"][0].component == "White-light classifier"
    assert ctx["model_rows"][0].status == "Not available — not used in triage"


def test_rgba_and_missing_key_images():
    import io

    from PIL import Image

    from app.reports.images import embed_artifact

    buf = io.BytesIO()
    Image.new("RGBA", (40, 30), (255, 0, 0, 128)).save(buf, format="PNG")
    img = embed_artifact(lambda key: buf.getvalue(), "x.png")
    assert img.available and img.data_uri.startswith("data:image/jpeg;base64,")
    assert embed_artifact(lambda key: b"", "x.png").reason == "Image not available"
    assert embed_artifact(lambda key: None, None).reason == "Not captured"


def test_preview_cli_fixture_and_result_file(tmp_path: Path):
    from app.reports.preview import main

    out = tmp_path / "high.pdf"
    sitemap = tmp_path / "map.svg"
    assert main(["--fixture", "high", "--out", str(out), "--html", "--sitemap", str(sitemap)]) == 0
    assert out.read_bytes().startswith(b"%PDF")
    assert "Clinician review" in out.with_suffix(".html").read_text(encoding="utf-8")
    assert sitemap.read_text(encoding="utf-8").startswith("<svg")

    case = cached_case("low")
    run_dir = tmp_path / "result"
    for key, payload in case.artifacts.items():
        target = run_dir / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    result_json = run_dir / "result.json"
    result_json.write_text(case.result.model_dump_json(), encoding="utf-8")
    html_out = tmp_path / "low.html"
    assert (
        main(
            [
                "--result",
                str(result_json),
                "--out",
                str(html_out),
                "--record-url",
                "https://example.org/r/1",
                "--report-id",
                "R-LOW",
            ]
        )
        == 0
    )
    html = html_out.read_text(encoding="utf-8")
    assert "R-LOW" in html and "https://example.org/r/1" in html
    assert html.count("data:image/jpeg;base64,") == 0  # LOW has no findings, so no panels


# --------------------------------------------------------------------------------------------
# Image crops, time zones, coverage wording, preview defaults
# --------------------------------------------------------------------------------------------


def test_crop_window_fills_the_frame_and_follows_the_region():
    from app.reports.images import crop_window, union_box
    from mukha_pipeline.schema.models import NormBox

    # 4:3 image into a 58:30 frame: full width, height cropped, centred on the region.
    x0, y0, x1, y1 = crop_window((480, 360), (58, 30), NormBox(x=0.4, y=0.05, w=0.2, h=0.1))
    assert (x0, x1) == (0, 480)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(58 / 30, rel=0.01)
    assert y0 == 0  # clamped to the top edge
    _, y0c, _, y1c = crop_window((480, 360), (58, 30), None)
    assert y0c == pytest.approx(360 - y1c, abs=1)  # centred without a focus
    # Portrait images are cropped in height too; a wide image is cropped in width.
    wx0, _, wx1, _ = crop_window((1000, 200), (58, 30), NormBox(x=0.9, y=0.4, w=0.1, h=0.2))
    assert wx1 == 1000 and (wx1 - wx0) == pytest.approx(200 * 58 / 30, abs=1)
    box = union_box([NormBox(x=0.1, y=0.2, w=0.1, h=0.1), NormBox(x=0.5, y=0.6, w=0.2, h=0.3)])
    assert (box.x, box.y) == (0.1, 0.2)
    assert box.w == pytest.approx(0.6) and box.h == pytest.approx(0.7)
    assert union_box([]) is None


def test_embed_artifact_crops_and_moves_brackets():
    import base64
    import io

    from PIL import Image

    from app.reports.images import embed_artifact
    from mukha_pipeline.schema.models import NormBox

    buf = io.BytesIO()
    Image.new("RGB", (480, 360), (200, 120, 120)).save(buf, format="PNG")
    box = NormBox(x=0.45, y=0.45, w=0.1, h=0.1)
    img = embed_artifact(lambda key: buf.getvalue(), "x.png", boxes=[box], aspect=(58, 30))
    assert img.available
    raw = base64.b64decode(img.data_uri.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as out:
        assert out.width / out.height == pytest.approx(58 / 30, rel=0.02)
        # Brackets are drawn around the (re-mapped) region: dark pixels near its corner.
        cx, cy = round(0.43 * out.width), round(out.height * (0.43 * 360 - 56) / 248)
        patch = out.crop((cx - 4, cy - 4, cx + 4, cy + 4)).convert("L")
        assert patch.getextrema()[0] < 80


def test_display_timezone_fallbacks():
    from app.reports.context import IST_FALLBACK, display_timezone

    ist = display_timezone("Asia/Kolkata")
    assert datetime(2026, 10, 14, 4, 5, tzinfo=UTC).astimezone(ist).strftime("%H:%M") == "09:35"
    assert display_timezone(None) is UTC
    assert display_timezone("Not/AZone") is UTC
    assert IST_FALLBACK.utcoffset(None) == timedelta(hours=5, minutes=30)


def test_naive_datetimes_are_treated_as_utc():
    case = cached_case("high")
    naive = case.result.context.model_copy(update={"captured_at": datetime(2026, 10, 14, 4, 5)})
    result = case.result.model_copy(update={"context": naive})
    ctx = build_context(result, case.reader, identity=None, options=default_options(case))
    assert ctx["header"]["captured_at"] == "14 Oct 2026, 09:35 IST"


def test_site_list_text_merges_pairs():
    from app.reports.context import site_list_text
    from mukha_pipeline.sites import OralSite as S

    assert site_list_text([S.HARD_PALATE, S.SOFT_PALATE, S.RETROMOLAR_LEFT]) == [
        "hard and soft palate",
        "left retromolar area",
    ]
    assert site_list_text([S.BUCCAL_MUCOSA_LEFT, S.BUCCAL_MUCOSA_RIGHT]) == ["both buccal mucosae"]
    assert site_list_text([]) == []


def test_reference_site_shows_reference_not_low_pill():
    case = cached_case("high")
    ctx = build_context(case.result, case.reader, identity=None, options=default_options(case))
    rows = {r.site: r for r in ctx["qc"]["sites"]}
    assert not rows["Right buccal mucosa"].show_level
    assert rows["Left buccal mucosa"].show_level
    assert all(r.qc in ("Passed", "Failed") for r in rows.values())  # no QC scores


def test_preview_cli_inconclusive_has_no_qr(tmp_path: Path):
    from app.reports.preview import main

    out = tmp_path / "inc.html"
    assert main(["--fixture", "inconclusive", "--out", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "QR code not included" in html
    out = tmp_path / "stress.html"
    assert main(["--fixture", "stress", "--out", str(out)]) == 0
    assert "XXXX XX-XXXX-XXXX-0123" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("agreed", "modified", "expected"),
    [
        (
            True,
            None,
            "Clinician-reviewed: agrees with the device level (HIGH) — Dr. B, 20 Oct 2026",
        ),
        (False, "HIGH", "Clinician-reviewed: agrees with the device level (HIGH) — Dr. B, 20 Oct"),
        (False, None, "Clinician-reviewed: see the reviewer's notes (device level HIGH) — Dr. B"),
    ],
)
def test_review_lines_without_a_level_change(agreed, modified, expected):
    from mukha_pipeline.schema.models import ClinicianReview

    case = cached_case("high")
    review = ClinicianReview(
        reviewer_name="Dr. B",
        agreed=agreed,
        modified_risk_level=modified,
        signed_at=datetime(2026, 10, 20, 4, 0, tzinfo=UTC),
    )
    result = case.result.model_copy(update={"review": review})
    ctx = build_context(result, case.reader, identity=None, options=default_options(case))
    assert ctx["triage"]["reviewed_badge"] is None
    assert ctx["triage"]["reviewer_line"].startswith(expected)
    assert ctx["header"]["review_status"] == "Clinician-reviewed"
