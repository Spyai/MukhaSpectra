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
    ReportOptions,
    render_clinician_report_html,
    render_clinician_report_pdf,
)
from app.reports.render import css_string, visible_text
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


@pytest.fixture(scope="module", params=["high", "low", "inconclusive"])
def any_report(request, high, low, inconclusive) -> Rendered:
    return {"high": high, "low": low, "inconclusive": inconclusive}[request.param]


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
    assert "Ravi K." in high.text and "HSK-0042" in high.text
    assert "P-7F3K-92QD" in high.text
    assert "52 y · Male" in high.text
    assert "Not shown (privacy)" in low.text  # no identity passed


def test_risk_factor_summary_in_plain_words(high: Rendered):
    assert "Gutka 20 y, 6/day, bidi 15 y, alcohol daily; non-healing ulcer > 2 weeks" in high.text


def test_risk_badge_has_icon_and_text_not_colour_alone(high: Rendered, inconclusive: Rendered):
    assert re.search(r'class="risk-badge risk-high"><svg[^>]*class="risk-icon"', high.html)
    assert "High risk" in high.text
    assert 'class="risk-badge risk-inconclusive"' in inconclusive.html
    assert "Inconclusive" in inconclusive.text


def test_action_steps_and_timeline(high: Rendered, low: Rendered, inconclusive: Rendered):
    assert "Urgent referral for specialist examination and biopsy." in high.text
    assert "Within 2 weeks (14 days)" in high.text
    assert "In 12 months" in low.text
    assert "This does not rule out disease" in low.text
    assert "Today — before the patient leaves" in inconclusive.text


def test_contributing_factors_with_score_and_thresholds(high: Rendered):
    text = high.text
    assert "Rule score 12.8 (Moderate ≥ 3.0, High ≥ 6.0)" in text
    assert "Highest finding score (left buccal mucosa)" in text
    assert "+10.3" in text
    # Overrides are marked as red flags the device cannot lower.
    assert text.count("Clinical red flag — the device cannot lower this") == 2
    assert "Non-healing ulcer for more than 2 weeks" in text


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
    # Spectral plot + features + classifier status.
    assert 'alt="Spectral plot, lesion versus reference"' in high.html
    assert "Hb dip (540–580 nm)" in text
    assert "Suspicious (heuristic)" in text
    assert "not a probability" in text


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
    for level in ("LOW", "MODERATE", "HIGH"):
        assert f'<span class="box"></span>{level}' in high.html
    assert "Signature" in high.text and "Date" in high.text


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
    assert "(signed electronically)" in text


def test_referral_with_qr_and_example_facilities(high: Rendered):
    url = "https://mukhaspectra.example.org/r/scr-2026-10-14-0042"
    assert 'alt="QR code linking to the secure record"' in high.html
    assert re.search(r'<img src="data:image/svg\+xml;base64,[^"]+" alt="QR code', high.html)
    assert url in high.text
    assert "Example facility — configure facilities.yaml" in high.text
    assert "Nearest facility" in high.text
    assert "+91-XXXXX-XXXXX" in high.text
    assert "Urgent referral · Within 2 weeks (14 days)" in high.text
    # The URL is also printed in the PDF (may wrap across lines).
    assert url.replace("/", "") in "".join(high.pages).replace(" ", "").replace("/", "")


def test_no_qr_without_record_url(inconclusive: Rendered):
    assert "QR code not included" in inconclusive.text
    assert "QR code linking" not in inconclusive.html


def test_no_facilities_configured():
    case = cached_case("low")
    options = ReportOptions(report_id="R-1", generated_at=datetime(2026, 10, 14, tzinfo=UTC))
    text = visible_text(render_clinician_report_html(case.result, case.reader, options=options))
    assert "No referral facility configured" in text


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
        assert "rules 1.0.0" in page  # footer versions


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


def test_lesion_detail_is_capped_at_three():
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
    assert html.count('<article class="lesion') == 3
    assert "Other findings" in text
    assert "Extra finding 0." in text and "Extra finding 1." in text
    assert "5 findings by risk level" in text


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
