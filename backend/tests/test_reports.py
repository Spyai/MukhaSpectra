"""Clinician report engine: HTML structure, PDF output, positioning and honesty rules."""

from __future__ import annotations

import io
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import pypdf
import pytest
from report_fixtures import ReportCase, cached_case, default_options

from app.reports import (
    Facility,
    PatientIdentity,
    ProgrammeContact,
    ReportOptions,
    render_clinician_report_html,
    render_clinician_report_pdf,
)
from app.reports.render import STATIC_DIR, css_string, visible_text
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import (
    PROTOTYPE_BANNER,
    ClinicianReview,
    RiskLevel,
    ScreeningResult,
)

SECTION_HEADINGS = [
    "Oral screening — clinician referral report",  # 1 header
    "Screening aid — not a diagnosis",  # 2 disclaimer
    "Patient and risk factors",  # 3
    "Overall risk level and recommended action",  # 4
    "Oral site map",  # 5 (printed beside the risk card)
    "Why this level — contributing factors",  # 4, continued under the map row
    "Suspicious findings",  # 6
    "Quality control and limitations",  # 7
    "Clinician review",  # 8
    "Referral details",  # 9
]
FORBIDDEN_CLAIMS = re.compile(r"accuracy|sensitivity|specificity", re.IGNORECASE)


@dataclass
class Rendered:
    case: ReportCase
    html: str
    pdf: bytes
    pages: list[str]
    seconds: float

    @property
    def text(self) -> str:
        return visible_text(self.html)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _pdf_pages(pdf: bytes) -> list[str]:
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    return [_norm(page.extract_text() or "") for page in reader.pages]


def _render(case: ReportCase, *, identity: PatientIdentity | None = None, **opts) -> Rendered:
    options = default_options(case, **opts)
    started = time.perf_counter()
    pdf = render_clinician_report_pdf(case.result, case.reader, identity=identity, options=options)
    seconds = time.perf_counter() - started
    html = render_clinician_report_html(
        case.result, case.reader, identity=identity, options=options
    )
    return Rendered(case, html, pdf, _pdf_pages(pdf), seconds)


@pytest.fixture(scope="module")
def high() -> Rendered:
    return _render(
        cached_case("high"), identity=PatientIdentity(name="Ravi K.", external_id="HSK-0042")
    )


@pytest.fixture(scope="module")
def low() -> Rendered:
    return _render(cached_case("low"))


@pytest.fixture(scope="module")
def inconclusive() -> Rendered:
    return _render(cached_case("inconclusive"), record_url=None)


@pytest.fixture(scope="module")
def high_not_synthetic() -> Rendered:
    return _render(cached_case("high", synthetic=False))


@pytest.fixture(scope="module")
def stress() -> Rendered:
    case = cached_case("stress")
    identity = PatientIdentity(name=case.identity_name, external_id=case.identity_external_id)
    return _render(case, identity=identity)


@pytest.fixture(scope="module", params=["high", "low", "inconclusive", "stress"])
def any_report(request, high, low, inconclusive, stress) -> Rendered:
    return {"high": high, "low": low, "inconclusive": inconclusive, "stress": stress}[request.param]


# --------------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------------


def test_html_contains_every_section_heading(any_report: Rendered):
    for heading in SECTION_HEADINGS:
        assert heading in any_report.text, heading


def test_sections_follow_brief_order(high: Rendered):
    positions = [high.text.index(h) for h in SECTION_HEADINGS]
    assert positions == sorted(positions)


def test_header_records_versions_devices_and_calibration(high: Rendered):
    r = high.case.result
    v = r.versions
    text = high.text
    for needle in [
        f"MukhaSpectra {v.software}",
        v.pipeline,
        v.schema_version,
        v.rule_config.name,
        v.rule_config.sha256[:8],
        v.qc_config.sha256[:8],
        v.analysis_config.sha256[:8],
        "scout-dev-01",
        "confirm-dev-01",
        "scout 0.1.0",
        "Calibrated",
        "PHC Hoskote screening camp",
        "CHW Lakshmi R. (hw-17)",
        "MS-2026-10-14-0042-R1",
        r.result_id,
    ]:
        assert needle in text, needle
    # The full rule-config hash is printed once for traceability.
    assert v.rule_config.sha256 in text


def test_model_status_is_honest(high: Rendered):
    text = high.text
    assert "White-light classifier: not available — not used in triage" in text
    assert "Placeholder heuristic, not validated" in text
    assert "Baseline heuristic, unvalidated" in text
    assert "no performance figures are claimed" in text


def test_patient_details_are_minimal(high: Rendered, low: Rendered):
    assert "Ravi K." in high.text
    # External IDs (ABHA / Aadhaar-linked) are printed masked, last 4 characters only.
    assert "HSK-0042" not in high.text and "HSK-0042" not in " ".join(high.pages)
    assert "Patient ID (masked) XXX-0042" in high.text
    assert "P-7F3K-92QD" in high.text
    assert "52 y · Male" in high.text
    assert "Name not shown (privacy)" in low.text  # no identity passed
    assert "Recorded · research: no" in high.text


def test_risk_factor_summary_in_plain_words(high: Rendered):
    assert (
        "Gutka 20 y (6/day) · bidi 15 y · alcohol daily; non-healing ulcer > 2 weeks · "
        "burning sensation" in high.text
    )


def test_risk_badge_has_icon_and_text_not_colour_alone(high: Rendered, inconclusive: Rendered):
    assert re.search(r'class="risk-badge risk-high"><svg[^>]*class="risk-icon"', high.html)
    # The header carries a compact copy of the badge (10-second test) besides the main one.
    assert re.search(r'class="risk-badge sm risk-high"><svg[^>]*class="risk-icon"', high.html)
    assert "HIGH Overall risk level" in high.text  # subtitle no longer repeats the level
    assert "High risk High risk" not in high.text
    assert 'class="risk-badge risk-inconclusive"' in inconclusive.html
    assert "INCONCLUSIVE Overall risk level" in inconclusive.text


def test_action_steps_and_timeline(high: Rendered, low: Rendered, inconclusive: Rendered):
    assert "Urgent referral for specialist examination and biopsy." in high.text
    # Captured 14 Oct 2026, 09:35 IST: the due date is printed with the timeline.
    assert "Within 2 weeks — by 28 Oct 2026" in high.text
    assert "In 12 months — by 14 Oct 2027" in low.text
    assert "This does not rule out disease" in low.text
    assert "Same day (at the screening visit)" in inconclusive.text
    assert "before the patient leaves" not in inconclusive.text


def test_contributing_factors_with_score_and_thresholds(high: Rendered):
    text = high.text
    assert "Rule score 12.8 (Moderate ≥ 3.0, High ≥ 6.0)" in text
    assert "Highest finding score (left buccal mucosa)" in text
    assert "+10.3" in text
    # The highest finding score names its finding, not "Site".
    assert re.search(
        r"Highest finding score \(left buccal mucosa\)</td>\s*<td>Finding 1</td>", high.html
    )
    # Overrides are tagged "Red flag (floor)" with one footnote under the table.
    assert high.html.count('<span class="tag">Red flag (floor)</span>') == 2
    assert (
        text.count("a clinical red flag sets a minimum risk level; the device cannot lower it") == 1
    )
    assert "Non-healing ulcer for more than 2 weeks" in text
    # A reported symptom has no site: the clinician is told where to look.
    assert (
        "Floor: HIGH · patient-reported; site not recorded — examine the whole mouth and neck"
        in text
    )


def test_inconclusive_without_score_still_explains(inconclusive: Rendered):
    assert "No overall rule score was computed" in inconclusive.text
    assert "Left buccal mucosa could not be analysed (QC failed)" in inconclusive.text


def test_lesion_blocks(high: Rendered):
    text = high.text
    assert "Finding 1 · Left buccal mucosa" in text
    assert "Finding 2 · Left lateral tongue" in text
    assert text.index("Finding 1 · Left buccal mucosa") < text.index("Finding 2 · Left lateral")
    # Triptych panels with embedded images (no external references).
    assert high.html.count('<figure class="panel">') == 6
    for title in ("White light", "Autofluorescence", "AF-loss heatmap"):
        assert title in text
    # Measurements, reference and confounders.
    for needle in [
        "38.6 mm²",
        "0.52",
        "0.38",
        "1.71×",
        "Contralateral — Right buccal mucosa",
        "16.0 % of mucosa",
        "In-image (automatic)",
        "Porphyrin / plaque (red fluorescence) 34 %",
    ]:
        assert needle in text, needle
    assert "Diascopy: AF loss persists under pressure" in text
    assert "Thick white patch, does not scrape off" in text
    assert "r1 · In photo: centre" in text
    assert "Visual: no lesion recorded" in text  # finding 2: nothing seen, said explicitly
    assert "Border irreg. 0 = smooth round edge, 1 = very irregular" in text
    # Spectral plot + features (2 decimals, cited ones bold) + classifier status.
    assert 'alt="Spectral plot, lesion versus reference"' in high.html
    assert '<tr class="cited"><td>Hb dip (540–580 nm)</td><td class="num">0.38</td>' in high.html
    assert '<tr class="cited"><td>555 / 630 nm ratio</td>' in high.html
    assert '<tr class=""><td>590 / 680 nm ratio</td>' in high.html
    assert "Suspicious (heuristic)" in text
    # A placeholder heuristic's score is not printed (it invites over-reading).
    assert "score 0.68" not in text
    assert "Placeholder heuristic, not validated" in text


def test_qc_and_limitations(high: Rendered, inconclusive: Rendered):
    assert "Image blurry — hold the device steady and retake." in high.text
    assert "Retake needed" in high.text
    assert "Right lateral tongue could not be analysed" in high.text
    for lim in high.case.result.limitations:
        assert lim.message in high.text
    assert "Uncalibrated" in inconclusive.text
    assert "Fluorescence too weak — check the violet light is on and retake." in inconclusive.text
    assert "Some glare" in inconclusive.text  # warnings shown too


def test_review_block_blank_by_default(high: Rendered):
    assert "Agree with device risk level (HIGH)" in high.text
    assert "Modify risk to:" in high.text
    for level in ("LOW", "MODERATE"):
        assert f'<span class="box"></span>{level}</span>' in high.html
    assert '<span class="box"></span>HIGH</span>' not in high.html  # current level excluded
    for field in (
        "Clinical findings on examination",
        "Biopsy:",
        "taken (site",
        "planned",
        "not indicated",
        "Follow-up date:",
        "Signature",
        "Date",
    ):
        assert field in high.text, field
    assert "not a digital signature" not in high.text  # only when a review was recorded


def test_review_block_prefilled_when_signed():
    case = cached_case("low")
    review = ClinicianReview(
        reviewer_name="Dr. A. Rao",
        reviewer_id="KDC-12345",
        agreed=False,
        modified_risk_level=RiskLevel.MODERATE,
        notes="Re-examine in 2 weeks.",
        signed_at=datetime(2026, 10, 15, 10, 0, tzinfo=UTC),
    )
    result = case.result.model_copy(update={"review": review})
    html = render_clinician_report_html(result, case.reader, options=default_options(case))
    text = visible_text(html)
    assert "Dr. A. Rao" in text and "KDC-12345" in text
    assert "Re-examine in 2 weeks." in text
    assert '<span class="box">✓</span>MODERATE' in html
    assert "(signed electronically)" not in text
    assert "(recorded in MukhaSpectra — not a digital signature)" in text
    # The clinician's level reaches the headline and the referral, with the device level.
    line = "Clinician-reviewed level: MODERATE (device: LOW) — Dr. A. Rao, 15 Oct 2026"
    assert text.count(line) >= 3  # header, overall card, referral
    assert "15 Oct 2026, 15:30 IST" in text  # review time shown in IST


def test_referral_with_qr_and_example_facilities(high: Rendered):
    url = "https://mukhaspectra.example.org/r/scr-2026-10-14-0042"
    assert 'alt="QR code linking to this screening record"' in high.html
    # No access-control claim until the record is behind a login (Phase 5).
    assert "Scan to open this screening record (access control arrives in a later release)" in (
        high.text
    )
    assert "secure" not in high.text.lower() and "login" not in high.text.lower()
    assert re.search(r'<img src="data:image/svg\+xml;base64,[^"]+" alt="QR code', high.html)
    assert url in high.text
    assert "Example facility — configure facilities.yaml" in high.text
    assert "First-choice facility" in high.text and "Nearest facility" not in high.text
    assert "+91-XXXXX-XXXXX" in high.text
    assert "Urgent referral · Within 2 weeks — by 28 Oct 2026" in high.text
    # The referral stands alone: who, and why.
    assert "Patient Ravi K. · P-7F3K-92QD · 52 y · Male" in high.text
    assert (
        "Reason HIGH risk level · Finding 1, left buccal mucosa (HIGH) + 1 more finding · "
        "red flags: non-healing ulcer for more than 2 weeks; white patch seen by the health "
        "worker" in high.text
    )
    # The URL is also printed in the PDF (may wrap across lines).
    assert url.replace("/", "") in "".join(high.pages).replace(" ", "").replace("/", "")


def test_no_qr_without_record_url(inconclusive: Rendered):
    assert "QR code not included" in inconclusive.text
    assert "QR code linking" not in inconclusive.html


def test_no_facilities_configured(caplog):
    case = cached_case("high")
    options = ReportOptions(report_id="R-1", generated_at=datetime(2026, 10, 14, tzinfo=UTC))
    with caplog.at_level(logging.WARNING, logger="app.reports.context"):
        html = render_clinician_report_html(case.result, case.reader, options=options)
    text = visible_text(html)
    assert "Referral facility not configured — contact the screening programme." in text
    assert '<div class="facility missing">' in html
    assert "facilities.yaml" not in text
    assert any("no referral facility configured" in r.message for r in caplog.records)


def test_programme_contact_is_printed():
    case = cached_case("high")
    options = ReportOptions(
        report_id="R-1",
        generated_at=datetime(2026, 10, 14, tzinfo=UTC),
        programme_contact=ProgrammeContact(name="District NCD cell", phone="+91-XXXXX-XXXXX"),
    )
    text = visible_text(render_clinician_report_html(case.result, case.reader, options=options))
    assert (
        "Referral facility not configured — contact the screening programme: "
        "District NCD cell · +91-XXXXX-XXXXX." in text
    )
    assert "Programme: District NCD cell · +91-XXXXX-XXXXX." in text


def test_example_facilities_never_reach_real_results(high_not_synthetic: Rendered):
    text = high_not_synthetic.text
    assert "Example" not in text
    assert "Referral facility not configured — contact the screening programme" in text


def test_low_and_inconclusive_show_one_facility_line(low: Rendered, inconclusive: Rendered):
    assert "If a symptom appears or a patch persists > 2 weeks: Example District Hospital" in (
        low.text
    )
    assert "Regional Cancer Centre" not in low.text  # HIGH-only facility
    assert "Government Dental College" not in low.text  # single line only
    assert "First-choice facility" not in low.text
    assert "If a retake is not possible or a lesion is visible: Example District Hospital" in (
        inconclusive.text
    )
    assert "Patient P-9Q4C-77LM · 61 y · Female" in inconclusive.text
    assert "Reason INCONCLUSIVE screen · Left buccal mucosa could not be assessed" in (
        inconclusive.text
    )


def test_facility_levels_filter():
    case = cached_case("low")
    options = ReportOptions(
        report_id="R-1",
        generated_at=datetime(2026, 10, 14, tzinfo=UTC),
        facilities=[
            Facility(name="Cancer Centre", address="A", levels=["HIGH"]),
            Facility(name="Local Clinic", address="B", capabilities=["oral medicine"]),
        ],
    )
    text = visible_text(render_clinician_report_html(case.result, case.reader, options=options))
    assert "Local Clinic" in text and "Cancer Centre" not in text


def test_real_facility_has_no_example_label():
    case = cached_case("low")
    options = ReportOptions(
        report_id="R-1",
        generated_at=datetime(2026, 10, 14, tzinfo=UTC),
        facilities=[Facility(name="Test Clinic", address="1 Test Road", phone="+91-XXXXX-XXXXX")],
    )
    text = visible_text(render_clinician_report_html(case.result, case.reader, options=options))
    assert "Test Clinic" in text
    assert "Example facility" not in text


def test_html_is_self_contained(any_report: Rendered):
    html = any_report.html
    assert not re.search(r'(src|href)="(?!data:)', html)
    assert "<link" not in html
    assert "@import" not in html


# --------------------------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------------------------


def test_pdf_renders_with_expected_page_count(any_report: Rendered):
    assert any_report.pdf.startswith(b"%PDF")
    assert 2 <= len(any_report.pages) <= 4


def test_high_fixture_is_two_to_four_pages(high: Rendered):
    assert 2 <= len(high.pages) <= 4


def test_banner_and_report_id_on_every_page(any_report: Rendered):
    report_id = default_options(any_report.case).report_id
    for i, page in enumerate(any_report.pages, start=1):
        assert PROTOTYPE_BANNER in page, f"banner missing on page {i}"
        assert report_id in page, f"report id missing on page {i}"
        assert f"Page {i} of {len(any_report.pages)}" in page
        rules = any_report.case.result.versions.rule_config.version
        assert f"rules {rules}" in page  # footer versions


def test_running_header_identifies_the_patient(any_report: Rendered):
    """Continuation pages carry report id + pseudonymous id + age/sex (never the name)."""
    p = any_report.case.result.patient
    sex = {"male": "M", "female": "F"}[p.sex.value]
    expected = f"{p.pseudonymous_id} · {p.age_years} y {sex}"
    for i, page in enumerate(any_report.pages, start=1):
        assert expected in page, f"patient missing from running header on page {i}"
    top_right = re.search(r"@top-right \{ content: ([^;]*);", any_report.html)
    assert top_right is not None and p.pseudonymous_id in top_right.group(1)
    if any_report.case.identity_name:  # the name stays out of the running header
        assert any_report.case.identity_name.split()[0] not in top_right.group(1)


def test_five_findings_fit_in_four_pages(stress: Rendered):
    r = stress.case.result
    assert len(r.lesions) == 5 and r.review is not None
    assert 2 <= len(stress.pages) <= 4
    assert stress.html.count('<article class="lesion') == 2
    text = stress.text
    for n in (3, 4, 5):
        assert re.search(rf"<td class=\"num\">{n}</td><td>", stress.html)
    # A spectral reading linked to two detailed findings is plotted once.
    assert stress.html.count('alt="Spectral plot, lesion versus reference"') == 1
    assert "plot and features under finding 1" in text
    # White-light classifier available: output shown, still not used in triage.
    assert "White-light classifier output (model output, not clinically validated)" in text
    assert "White patch 0.71" in text
    assert "XXXX XX-XXXX-XXXX-0123" in text and "2345-6789" not in text


def test_clinician_modified_level_reaches_headline(stress: Rendered):
    line = "Clinician-reviewed level: MODERATE (device: HIGH) — Dr. A. Rao, 20 Oct 2026"
    assert stress.text.count(line) == 3  # header, overall card, referral
    assert re.search(r'class="reviewed"><span class="pill risk-moderate">', stress.html)
    assert "Urgent referral · device recommendation" in stress.text
    assert "The line above is the device recommendation before review" in stress.text
    assert "Clinician-reviewed" in stress.text.split("Patient and risk factors")[0]
    assert (
        "Reason MODERATE risk level (clinician-reviewed; device HIGH) · Finding 1, left buccal "
        "mucosa (HIGH) + 4 more findings" in stress.text
    )
    # Facilities follow the reviewed level: the HIGH-only cancer centre is not listed.
    assert "Regional Cancer Centre" not in stress.text


def test_device_coverage_is_stated(high: Rendered, low: Rendered, inconclusive: Rendered):
    assert "Device coverage: 5 of 14 sites (1 used only as a comparison reference)." in high.text
    assert (
        "Not examined (no device information): upper and lower lip mucosa, ventral tongue / "
        "floor of mouth, hard and soft palate, upper and lower gingiva, both retromolar areas."
        in high.text
    )
    assert "Not assessed (QC failed): Right lateral tongue — examine clinically." in high.text
    assert "Device coverage: 6 of 14 sites." in low.text
    assert "Device coverage: 2 of 14 sites." in inconclusive.text


def test_inconclusive_speaks_to_the_clinician(inconclusive: Rendered):
    assert (
        "If this patient reaches you: the device could not assess left buccal mucosa; "
        "examine them clinically." in inconclusive.text
    )
    assert "At the screening visit:" in inconclusive.text


def test_times_in_display_time_zone(high: Rendered):
    assert "14 Oct 2026, 09:35 IST" in high.text  # captured 04:05 UTC
    assert "UTC" not in high.text
    case = high.case
    utc = render_clinician_report_html(
        case.result, case.reader, options=default_options(case, display_tz="UTC")
    )
    assert "14 Oct 2026, 04:05 UTC" in visible_text(utc)
    unknown = render_clinician_report_html(
        case.result, case.reader, options=default_options(case, display_tz="Mars/Olympus")
    )
    assert "14 Oct 2026, 04:05 UTC" in visible_text(unknown)


def test_heatmap_has_colour_scale(high: Rendered):
    assert high.html.count('class="hm-scale"') == 2
    assert "relative AF 0.85 → 0.20 · no colour = no loss · grey = not analysed" in high.text
    assert re.search(r'class="bar" style="background: linear-gradient\(to right, #', high.html)


def test_heatmap_scale_omitted_without_af_module(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mukha_pipeline.af":
            raise ImportError("af module not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    case = cached_case("high")
    html = render_clinician_report_html(case.result, case.reader, options=default_options(case))
    assert 'class="hm-scale"' not in html
    assert "AF-loss heatmap" in visible_text(html)


def test_image_panels_fill_the_frame(high: Rendered):
    import base64

    from PIL import Image

    uris = re.findall(r'<div class="frame"><img src="data:image/jpeg;base64,([^"]+)"', high.html)
    assert len(uris) == 6
    for b64 in uris:
        with Image.open(io.BytesIO(base64.b64decode(b64))) as img:
            assert img.width / img.height == pytest.approx(58 / 30, rel=0.02)


def test_non_patient_data_is_flagged(high_not_synthetic: Rendered):
    assert "PHANTOM DATA — captured from a bench test object, not a patient." in (
        high_not_synthetic.text
    )
    for page in high_not_synthetic.pages:
        assert "PHANTOM DATA" in page


def test_stylesheet_rules():
    css = (STATIC_DIR / "report.css").read_text(encoding="utf-8")
    html_rule = re.search(r"\nhtml \{([^}]*)\}", css)
    assert html_rule is not None
    # Global tabular figures widen Inter's hyphen ("Non - healing"): numeric cells only.
    assert "tnum" not in html_rule.group(1) and "tabular" not in html_rule.group(1)
    assert "print-color-adjust: exact" in html_rule.group(1)
    assert re.search(r"\.cols-main-side \{[^}]*grid-template-columns: 1fr 74mm", css)
    assert re.search(r"\.panel img \{[^}]*object-fit: cover", css)
    for dead in (".review-chip", ".rf-groups .grp", ".status-chip.neutral", ".clf ul",
                 ".nowrap", ".map-col .orient"):  # fmt: skip
        assert dead not in css, dead


def test_no_hyphen_spacing_feature_on_ids(high: Rendered):
    """IDs must keep normal hyphens: no tabular-figure rule may apply to ID-bearing blocks."""
    css = (STATIC_DIR / "report.css").read_text(encoding="utf-8")
    for selector in (".report-meta", ".kv .v", ".mono", ".meta-id"):
        for rule in re.findall(r"([^{}]+)\{([^}]*)\}", css):
            selectors = [x.strip() for x in rule[0].split(",")]
            if selector in selectors:
                assert "tnum" not in rule[1], selector


@pytest.mark.parametrize("name", ["high", "high_not_synthetic", "stress"])
def test_findings_heading_is_never_stranded(name: str, request):
    rendered: Rendered = request.getfixturevalue(name)
    pages = [i for i, page in enumerate(rendered.pages) if "4 Suspicious findings" in page]
    assert len(pages) == 1
    assert "Finding 1 ·" in rendered.pages[pages[0]]


@pytest.mark.parametrize("name", ["high", "low", "inconclusive", "stress"])
def test_risk_level_in_top_third_of_page_one(name: str, request):
    """10-second test: level and timeline are printed within the top 35 % of page 1."""
    import fitz

    rendered: Rendered = request.getfixturevalue(name)
    level = rendered.case.result.triage.overall_risk_level.value
    with fitz.open(stream=rendered.pdf, filetype="pdf") as doc:
        page = doc[0]
        hits = page.search_for(level)
        assert hits
        assert min(h.y0 for h in hits) < 0.35 * page.rect.height


def test_synthetic_watermark_on_every_page(any_report: Rendered):
    assert any_report.case.result.synthetic
    for i, page in enumerate(any_report.pages, start=1):
        assert "SYNTHETIC DATA" in page, f"watermark missing on page {i}"
    assert 'class="watermark"' in any_report.html


def test_no_synthetic_marking_for_real_data(high_not_synthetic: Rendered):
    assert not high_not_synthetic.case.result.synthetic
    assert 'class="watermark"' not in high_not_synthetic.html
    assert "synthetic" not in high_not_synthetic.text.lower()
    for page in high_not_synthetic.pages:
        assert "synthetic" not in page.lower()
        assert PROTOTYPE_BANNER in page


def test_render_time_is_reasonable(high: Rendered):
    # The fixture's first render pays one-off imports (WeasyPrint, matplotlib, fonts); time a
    # warm render, which is what the API sees after start-up.
    case = high.case
    started = time.perf_counter()
    render_clinician_report_pdf(case.result, case.reader, options=default_options(case))
    assert time.perf_counter() - started < 5.0


def test_pdf_metadata_title(high: Rendered):
    meta = pypdf.PdfReader(io.BytesIO(high.pdf)).metadata
    assert meta is not None and "MS-2026-10-14-0042-R1" in (meta.title or "")


# --------------------------------------------------------------------------------------------
# Positioning and honesty
# --------------------------------------------------------------------------------------------


def test_positioning_clean_in_html_and_pdf(any_report: Rendered):
    assert find_positioning_violations(any_report.text) == []
    assert find_positioning_violations(" ".join(any_report.pages)) == []


def test_no_accuracy_claims(any_report: Rendered):
    assert not FORBIDDEN_CLAIMS.search(any_report.text)
    assert not FORBIDDEN_CLAIMS.search(" ".join(any_report.pages))


def test_disclaimer_prominent(any_report: Rendered):
    assert 'class="disclaimer"' in any_report.html
    assert "Final diagnosis requires examination by a clinician and histopathology" in (
        any_report.text
    )


def test_positioning_guard_logs_pipeline_wording(caplog):
    case = cached_case("low")
    bad_triage = case.result.triage.model_copy(update={"explanation": "Lesion looks malignant."})
    result = case.result.model_copy(update={"triage": bad_triage})
    with caplog.at_level(logging.WARNING, logger="app.reports.render"):
        render_clinician_report_html(result, case.reader, options=default_options(case))
    assert any("positioning violations" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------------------------


def test_missing_artifacts_render_placeholders():
    case = cached_case("high")
    options = default_options(case)
    html = render_clinician_report_html(case.result, lambda key: None, options=options)
    assert html.count("Image not available") >= 6
    pdf = render_clinician_report_pdf(case.result, lambda key: None, options=options)
    assert pdf.startswith(b"%PDF")
    assert 2 <= len(_pdf_pages(pdf)) <= 4


def test_failing_or_corrupt_artifacts_never_crash():
    case = cached_case("high")

    def broken(key: str) -> bytes | None:
        if key.endswith("wl.png"):
            raise OSError("storage offline")
        if key.endswith("heatmap_overlay.png"):
            return b"not an image"
        return case.reader(key)

    html = render_clinician_report_html(case.result, broken, options=default_options(case))
    assert "Image could not be read" in html
    assert "Image not available" in html
    assert html.count("data:image/jpeg;base64,") == 2  # the two AF panels still render


def test_lesion_detail_is_capped_at_two():
    case = cached_case("high")
    base = case.result.lesions
    extra = [
        base[0].model_copy(
            update={
                "lesion_id": f"les-x{i}",
                "risk_level": RiskLevel.MODERATE,
                "score": 3.0 + i / 10,
                "summary": f"Extra finding {i}.",
            }
        )
        for i in range(3)
    ]
    result: ScreeningResult = case.result.model_copy(update={"lesions": [*base, *extra]})
    html = render_clinician_report_html(result, case.reader, options=default_options(case))
    text = visible_text(html)
    assert html.count('<article class="lesion') == 2
    assert "Other findings" in text
    assert all(f"Extra finding {i}." in text for i in range(3))
    assert "5 findings by risk level; full evidence for the top 2" in text


def test_css_string_escapes_quotes_and_tags():
    out = css_string('Report "A" </style><script>')
    assert out.startswith('"') and out.endswith('"')
    inner = out[1:-1]
    assert '"' not in inner and "<" not in inner and ">" not in inner


def test_autoescape_of_free_text():
    case = cached_case("high")
    site = case.result.sites[0]
    flag = site.visual_flag.model_copy(update={"description": "<script>alert(1)</script>"})
    sites = [site.model_copy(update={"visual_flag": flag}), *case.result.sites[1:]]
    result = case.result.model_copy(update={"sites": sites})
    html = render_clinician_report_html(result, case.reader, options=default_options(case))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
