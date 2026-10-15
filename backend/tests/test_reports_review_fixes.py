"""Regression tests for review findings on the clinician report (wording and honesty).

* clinical.7: a finding below the overall level says why the overall level is higher.
* clinical.11 / honesty.7 / honesty.11: a reference-only site with a visual flag or a finding is
  never presented as healthy; a recorded appearance is never printed as "no lesion recorded".
* honesty.3: the spectral classifier is labelled a heuristic placeholder; "not suspicious" is
  printed with "does not rule out disease".
* honesty.5: the QR caption does not claim a secure, login-protected record.
* not_analysed sites (software error) are named as such and sent for clinical examination.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from report_fixtures import cached_case, default_options

from app.reports import labels as lb
from app.reports import render_clinician_report_html, render_site_map_svg
from app.reports.context import build_context
from app.reports.render import visible_text
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import (
    ContributingFactor,
    FactorSource,
    LesionOrigin,
    RiskLevel,
    ScreeningResult,
    VisualAppearance,
    VisualFlag,
)
from mukha_pipeline.sites import OralSite

NS = {"svg": "http://www.w3.org/2000/svg"}
S = OralSite


def _ctx(result: ScreeningResult, name: str = "high") -> dict:
    case = cached_case(name)
    return build_context(result, case.reader, identity=None, options=default_options(case))


def _text(result: ScreeningResult, name: str = "high") -> str:
    case = cached_case(name)
    return visible_text(
        render_clinician_report_html(result, case.reader, options=default_options(case))
    )


def _factor(code: str, label: str, source: FactorSource, points: float, detail: str | None = None):
    return ContributingFactor(code=code, label=label, source=source, points=points, detail=detail)


def _replace_site(result: ScreeningResult, site: OralSite, **update) -> ScreeningResult:
    sites = [s.model_copy(update=update) if s.site == site else s for s in result.sites]
    return result.model_copy(update={"sites": sites})


# --------------------------------------------------------------------------------------------
# clinical.7 — overall level above a finding's own level is explained on the finding
# --------------------------------------------------------------------------------------------


def _low_finding_raised(*, red_flag: bool) -> ScreeningResult:
    """One LOW finding (score 2.9); risk factors give MODERATE 5.4; optionally a HIGH floor."""
    r = cached_case("high").result
    les = next(x for x in r.lesions if x.lesion_id == "les-001")
    low_les = les.model_copy(update={"risk_level": RiskLevel.LOW, "score": 2.9})
    factors = [
        _factor("lesion.max_score", "Highest finding score (left buccal mucosa)",
                FactorSource.SITE, 2.9),
        _factor("risk.smokeless_tobacco", "Smokeless tobacco (gutka, 20 years)",
                FactorSource.CLINICAL_HISTORY, 1.0),
        _factor("risk.areca_nut", "Areca nut", FactorSource.CLINICAL_HISTORY, 0.5),
        _factor("risk.alcohol_daily", "Daily alcohol", FactorSource.CLINICAL_HISTORY, 0.5),
        _factor("risk.age_40_plus", "Age 40 or over", FactorSource.CLINICAL_HISTORY, 0.5),
    ]  # fmt: skip
    overrides = (
        [
            _factor(
                "symptom.non_healing_ulcer",
                "Non-healing ulcer for more than 2 weeks",
                FactorSource.OVERRIDE,
                0.0,
                "Clinical red flag: overall level is at least HIGH.",
            )
        ]  # fmt: skip
        if red_flag
        else []
    )
    overall = RiskLevel.HIGH if red_flag else RiskLevel.MODERATE
    triage = r.triage.model_copy(
        update={
            "overall_risk_level": overall,
            "score": 5.4,
            "contributing_factors": factors,
            "overrides_applied": overrides,
            "highest_risk_lesion_id": "les-001",
        }
    )
    sites = [
        s.model_copy(update={"site_risk_level": RiskLevel.LOW})
        if s.site == S.BUCCAL_MUCOSA_LEFT
        else s
        for s in r.sites
    ]
    return r.model_copy(update={"lesions": [low_les], "triage": triage, "sites": sites})


def test_low_finding_under_moderate_overall_names_the_risk_factors():
    result = _low_finding_raised(red_flag=False)
    ctx = _ctx(result)
    view = ctx["lesions"][0]
    assert view.badge.level == "LOW"
    assert view.level_context == (
        "Finding level: LOW · Overall level MODERATE because of the patient's risk factors "
        "(see “Why this level”)"
    )
    text = _text(result)
    assert view.level_context in text
    # Page 1 and the map: the finding / site colours are the finding's own level.
    assert ctx["findings_note"] is not None and "own level" in ctx["findings_note"]
    assert ctx["findings_note"] in text
    # The referral slip does not read as "MODERATE … finding (LOW)" without a reason.
    slip = ctx["referral"]["slip_reason"]
    assert (
        "Finding 1, left buccal mucosa (finding level LOW; overall level raised by the "
        "patient's risk factors)" in slip
    )
    assert find_positioning_violations(text) == []


def test_low_finding_under_high_red_flag_names_the_red_flag():
    result = _low_finding_raised(red_flag=True)
    view = _ctx(result)["lesions"][0]
    assert view.level_context == (
        "Finding level: LOW · Overall level HIGH because of a clinical red flag "
        "(non-healing ulcer for more than 2 weeks)"
    )


def test_lower_finding_under_a_higher_finding_says_so():
    ctx = _ctx(cached_case("high").result)
    first, second = ctx["lesions"]
    assert first.level_context is None  # finding 1 is at the overall level
    assert second.level_context == (
        "Finding level: MODERATE · Overall level HIGH because of finding 1 and a clinical red "
        "flag (non-healing ulcer for more than 2 weeks)"
    )
    assert ctx["findings_note"] is None  # finding 1 explains the overall level


def test_low_finding_under_inconclusive_overall():
    r = _low_finding_raised(red_flag=False)
    triage = r.triage.model_copy(
        update={"overall_risk_level": RiskLevel.INCONCLUSIVE, "score": None}
    )
    ctx = _ctx(r.model_copy(update={"triage": triage}))
    assert ctx["lesions"][0].level_context == (
        "Finding level: LOW · Overall level INCONCLUSIVE because of sites the device could not "
        "assess"
    )


def test_finding_at_overall_level_has_no_context_line():
    ctx = _ctx(cached_case("stress").result, "stress")
    assert ctx["lesions"][0].level_context is None


# --------------------------------------------------------------------------------------------
# clinical.11 / honesty.11 — a flagged reference-only site is never "healthy"
# --------------------------------------------------------------------------------------------


def _flagged_reference(level: RiskLevel = RiskLevel.HIGH) -> ScreeningResult:
    r = cached_case("high").result
    flag = VisualFlag(
        lesion_seen=True, appearance=[VisualAppearance.RED_PATCH], description="velvety red area"
    )
    les = next(x for x in r.lesions if x.lesion_id == "les-002")
    ref_les = les.model_copy(
        update={
            "lesion_id": "les-ref",
            "site": S.BUCCAL_MUCOSA_RIGHT,
            "origins": [LesionOrigin.VISUAL],
            "region_ids": [],
            "risk_level": level,
            "score": 6.5,
            "summary": "Red patch seen by the health worker at the reference side.",
        }
    )
    result = _replace_site(r, S.BUCCAL_MUCOSA_RIGHT, visual_flag=flag, site_risk_level=level)
    return result.model_copy(update={"lesions": [*result.lesions, ref_les]})


def _groups(svg: str) -> dict[str, ET.Element]:
    root = ET.fromstring(svg)
    return {g.get("data-site"): g for g in root.iterfind(".//svg:g[@data-site]", NS)}


def test_flagged_reference_site_on_the_map_shows_its_level_not_healthy():
    svg = render_site_map_svg(_flagged_reference())
    assert "healthy" not in svg.lower()
    g = _groups(svg)[S.BUCCAL_MUCOSA_RIGHT.value]
    assert g.get("data-state") == "reference_flagged"
    assert g.get("data-risk") == "HIGH"
    title = g.find("svg:title", NS)
    assert title is not None
    assert title.text == (
        "Right buccal mucosa: reference side (comparison), high risk; 1 flagged finding"
    )
    ellipse = g.find("svg:ellipse", NS)
    assert ellipse is not None
    assert ellipse.get("fill") == lb.RISK_COLORS[RiskLevel.HIGH]["fill"]
    assert ellipse.get("stroke-dasharray")  # still marked as the reference side
    icon = g.find("svg:g[@class='level-icon']", NS)
    assert icon is not None and icon.get("data-level") == "HIGH"


def test_visual_flag_alone_flags_a_reference_site():
    r = cached_case("high").result
    flag = VisualFlag(lesion_seen=False, appearance=[VisualAppearance.WHITE_PATCH])
    result = _replace_site(r, S.BUCCAL_MUCOSA_RIGHT, visual_flag=flag)
    g = _groups(render_site_map_svg(result))[S.BUCCAL_MUCOSA_RIGHT.value]
    assert g.get("data-state") == "reference_flagged"
    assert "reference side (comparison)" in (g.find("svg:title", NS).text or "")


def test_plain_reference_site_is_not_called_healthy():
    svg = render_site_map_svg(cached_case("high").result)
    assert "healthy" not in svg.lower()
    g = _groups(svg)[S.BUCCAL_MUCOSA_RIGHT.value]
    assert g.get("data-state") == "reference_only"
    assert (g.find("svg:title", NS).text or "").startswith(
        "Right buccal mucosa: reference only (comparison side)"
    )


def test_flagged_reference_site_in_the_report():
    result = _flagged_reference()
    ctx = _ctx(result)
    rows = {r.site: r for r in ctx["qc"]["sites"]}
    ref = rows["Right buccal mucosa"]
    assert ref.role == "Reference side (comparison)"
    assert ref.show_level and ref.badge.level == "HIGH"
    assert ctx["coverage"].summary == (
        "Device coverage: 5 of 14 sites (1 used only as a comparison reference; it has a "
        "flagged finding — see the findings)."
    )
    text = _text(result)
    assert "healthy" not in text.lower()
    assert find_positioning_violations(text) == []


@pytest.mark.parametrize("name", ["high", "low", "inconclusive", "stress"])
def test_reference_tissue_is_never_called_healthy(name: str):
    case = cached_case(name)
    assert "healthy" not in _text(case.result, name).lower()
    assert "healthy" not in render_site_map_svg(case.result).lower()


def test_coverage_names_the_reference_as_comparison():
    ctx = _ctx(cached_case("high").result)
    assert ctx["coverage"].summary == (
        "Device coverage: 5 of 14 sites (1 used only as a comparison reference)."
    )


# --------------------------------------------------------------------------------------------
# honesty.7 — a recorded appearance is printed, never "no lesion recorded"
# --------------------------------------------------------------------------------------------


def test_appearance_without_checkbox_is_printed():
    r = cached_case("high").result
    flag = VisualFlag(
        lesion_seen=False,
        appearance=[VisualAppearance.RED_PATCH],
        description="velvety red area, 1 cm",
    )
    result = _replace_site(r, S.LATERAL_TONGUE_LEFT, visual_flag=flag)
    views = {v.site: v for v in _ctx(result)["lesions"]}
    visual = views["Left lateral tongue"].visual
    assert visual == "Seen by health worker: red patch — “velvety red area, 1 cm”"
    assert "no lesion recorded" not in _text(result).split("Finding 2 ·")[1]


def test_description_without_appearance_is_still_printed():
    r = cached_case("high").result
    flag = VisualFlag(lesion_seen=False, description="slightly rough area")
    result = _replace_site(r, S.LATERAL_TONGUE_LEFT, visual_flag=flag)
    views = {v.site: v for v in _ctx(result)["lesions"]}
    assert views["Left lateral tongue"].visual == (
        "no lesion recorded · health-worker note: “slightly rough area”"
    )


def test_visual_seen_matches_the_pipeline_rule():
    from app.reports.ranking import visual_seen
    from mukha_pipeline.fusion.lesions import visual_seen as pipeline_visual_seen

    for flag in (
        VisualFlag(),
        VisualFlag(lesion_seen=True),
        VisualFlag(appearance=[VisualAppearance.WHITE_PATCH]),
        VisualFlag(description="note only"),
    ):
        assert visual_seen(flag) == pipeline_visual_seen(flag), flag


# --------------------------------------------------------------------------------------------
# honesty.3 — spectral classifier wording
# --------------------------------------------------------------------------------------------

PLACEHOLDER_STATUS = "Heuristic placeholder (not a trained model, not validated)"
NOT_SUSPICIOUS = "No spectral difference detected (does not rule out disease)"


def _with_classifier(status: str, label: str) -> ScreeningResult:
    r = cached_case("high").result
    readings = []
    for reading in r.spectral_readings:
        if reading.classifier is not None:
            clf = reading.classifier.model_copy(update={"status": status, "label": label})
            reading = reading.model_copy(update={"classifier": clf})
        readings.append(reading)
    return r.model_copy(update={"spectral_readings": readings})


def test_placeholder_classifier_reporting_ok_is_not_model_output():
    result = _with_classifier("ok", "suspicious")
    spectral = _ctx(result)["lesions"][0].spectral[0]
    assert spectral.classifier_status == PLACEHOLDER_STATUS
    assert spectral.classifier_score is None  # placeholder scores stay hidden
    text = _text(result)
    assert "Model output" not in text and "model output" not in text
    assert PLACEHOLDER_STATUS in text


def test_placeholder_status_wording():
    spectral = _ctx(cached_case("high").result)["lesions"][0].spectral[0]
    assert spectral.classifier_status == PLACEHOLDER_STATUS


def test_not_suspicious_label_does_not_reassure():
    result = _with_classifier("ok", "not_suspicious")
    spectral = _ctx(result)["lesions"][0].spectral[0]
    assert spectral.classifier_label == NOT_SUSPICIOUS
    text = _text(result)
    assert NOT_SUSPICIOUS in text
    assert "Not suspicious" not in text
    assert find_positioning_violations(text) == []


# --------------------------------------------------------------------------------------------
# honesty.5 — the QR caption makes no access-control claim
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["high", "inconclusive"])
def test_qr_caption_makes_no_security_claim(name: str):
    case = cached_case(name)
    url = None if name == "inconclusive" else "default"
    html = render_clinician_report_html(
        case.result, case.reader, options=default_options(case, record_url=url)
    )
    text = visible_text(html)
    for claim in ("secure", "login"):
        assert claim not in text.lower(), claim
        assert claim not in html.lower(), claim
    if url:
        assert (
            "Scan to open this screening record (access control arrives in a later release)" in text
        )
        assert 'alt="QR code linking to this screening record"' in html
    else:
        assert "QR code not included: no record link was configured for this report." in text


def test_compact_findings_note_has_no_security_claim():
    text = _text(cached_case("stress").result, "stress")
    assert "(all findings are in the screening record)" in text
    assert "secure" not in text.lower()


# --------------------------------------------------------------------------------------------
# analysis_status "not_analysed" (software error)
# --------------------------------------------------------------------------------------------


def test_not_analysed_site_is_named_as_a_software_error():
    r = cached_case("high").result
    result = _replace_site(r, S.TONGUE_DORSUM, analysis_status="not_analysed")
    ctx = _ctx(result)
    rows = {row.site: row for row in ctx["qc"]["sites"]}
    assert rows["Tongue dorsum"].analysis == "Not analysed (software error) — examine clinically"
    assert lb.ANALYSIS_STATUS_TEXT["not_analysed"] == (
        "Not analysed (software error) — examine clinically"
    )
    text = _text(result)
    assert "Not analysed (software error) — examine clinically" in text
    # The coverage line does not blame QC for a software error.
    assert (
        "Not assessed (QC failed or software error): Tongue dorsum, Right lateral tongue "
        "— examine clinically." in text
    )
    assert "Not assessed (QC failed): " not in text


def test_reference_role_text_when_unflagged():
    rows = {r.site: r for r in _ctx(cached_case("high").result)["qc"]["sites"]}
    ref = rows["Right buccal mucosa"]
    assert ref.role == "Reference only"
    assert not ref.show_level
    assert ref.reference_only and not ref.reference_flagged
    assert rows["Left buccal mucosa"].role == "Examined"


# --------------------------------------------------------------------------------------------
# Integration: wording and spectral-delta semantics agreed across pipeline and report
# --------------------------------------------------------------------------------------------


def test_low_urgency_text_never_says_no_referral_is_needed():
    from mukha_pipeline.schema.models import ReferralUrgency

    text = lb.URGENCY_TEXT[ReferralUrgency.NONE]
    assert "no referral" not in text.lower()
    assert "clinician review" in text.lower()
    report = _text(cached_case("low").result, "low")
    assert "No referral needed" not in report
    assert find_positioning_violations(report) == []


def test_spectral_delta_is_never_a_plain_probe_dependent_difference():
    """Without a pipeline delta (quotient-based, probe-independent) the Δ column shows a dash:
    a plain lesion-minus-reference difference depends on the probe's LED and filters."""
    r = cached_case("high").result
    readings = [
        x.model_copy(update={"delta_vs_reference": None}) if x.role == "lesion" else x
        for x in r.spectral_readings
    ]
    view = _ctx(r.model_copy(update={"spectral_readings": readings}))["lesions"][0].spectral[0]
    assert view.features and all(row.delta == lb.DASH for row in view.features)


def test_spectral_ratio_delta_is_shown_as_a_relative_change():
    r = cached_case("high").result
    lesion = next(x for x in r.spectral_readings if x.role == "lesion")
    delta = lesion.delta_vs_reference.model_copy(
        update={"ratio_555_630": -0.064, "hb_absorption_proxy": 0.031}
    )
    readings = [
        x.model_copy(update={"delta_vs_reference": delta}) if x.role == "lesion" else x
        for x in r.spectral_readings
    ]
    view = _ctx(r.model_copy(update={"spectral_readings": readings}))["lesions"][0].spectral[0]
    rows = {row.name: row for row in view.features}
    assert rows[lb.SPECTRAL_FEATURE_TEXT["ratio_555_630"]].delta == "−6.4 %"
    assert rows[lb.SPECTRAL_FEATURE_TEXT["hb_absorption_proxy"]].delta == "+0.031"
    assert view.delta_note and "÷" in view.delta_note
