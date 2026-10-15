"""Any detected loss of fluorescence at a high-risk site is re-examined, never routine.

Regression for the integration re-check of science.0: a mild, small FVL region on the lateral
tongue of a 58-year-old gutka user scored 2.84 points (LOW, re-screen in 12 months).
"""

from __future__ import annotations

import pytest
from fusion_helpers import (
    S,
    assert_invariants,
    default_rules,
    parse_rules_default,
    patient,
    region,
    risk,
    rules_data,
    site,
    write_rules,
)

from mukha_pipeline.fusion import TriageRulesError, parse_triage_rules, run_triage
from mukha_pipeline.schema.models import RiskLevel


def _triage(oral_site, rules=None):
    mild = region(mean_rel=0.68, fraction=0.036)
    out = run_triage([site(oral_site, [mild])], [], patient(58), risk(), rules or default_rules())
    assert_invariants(out)
    return out


def test_mild_fvl_at_every_high_risk_site_is_at_least_moderate():
    tr = parse_rules_default()
    for oral_site in tr.site.high_risk_sites:
        out = _triage(oral_site)
        lesion = out.lesions[0]
        assert lesion.score < 3.0, oral_site  # points alone would be LOW
        assert lesion.risk_level == RiskLevel.MODERATE, oral_site
        assert out.triage.overall_risk_level == RiskLevel.MODERATE, oral_site
        codes = [f.code for f in out.triage.overrides_applied]
        assert "af.high_risk_site_fvl_floor" in codes, oral_site


def test_mild_fvl_at_a_low_risk_site_keeps_the_score_path():
    out = _triage(S.SOFT_PALATE)
    assert out.lesions[0].risk_level == RiskLevel.LOW
    assert out.triage.overall_risk_level == RiskLevel.LOW


def test_floor_is_optional(tmp_path):
    def drop(d):
        del d["lesion"]["af"]["high_risk_site_fvl_floor"]

    out = _triage(S.LATERAL_TONGUE_RIGHT, write_rules(tmp_path, drop))
    assert out.triage.overall_risk_level == RiskLevel.LOW


@pytest.mark.parametrize(
    "bad",
    [
        "yes",
        {"floor": "LOW", "label": "x"},
        {"floor": "MODERATE"},
        {"floor": "MODERATE", "label": "Diagnosis: cancer"},
        {"floor": "MODERATE", "label": "ok", "extra": 1},
    ],
)
def test_floor_validation(bad):
    data = rules_data()
    data["lesion"]["af"]["high_risk_site_fvl_floor"] = bad
    with pytest.raises(TriageRulesError):
        parse_triage_rules(data)
