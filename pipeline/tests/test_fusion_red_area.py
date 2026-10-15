"""A deep-red area with fluorescence loss may be a red patch (erythroplakia): never discounted.

Inflammation and blood also look red and dark under violet light, but the device cannot tell
them from erythroplakia, the oral lesion most likely to be dysplastic. The rules therefore keep
such a lesion at least MODERATE (re-examine in 2 weeks) rather than discounting its AF points.
"""

from __future__ import annotations

import pytest
from fusion_helpers import (
    S,
    assert_invariants,
    codes,
    default_rules,
    patient,
    region,
    risk,
    rules_data,
    site,
    write_rules,
)

from mukha_pipeline.fusion import TriageRulesError, parse_triage_rules, run_triage
from mukha_pipeline.schema.models import ConfounderType, FactorSource, RiskLevel

C = ConfounderType


def _triage(regions, rules=None):
    out = run_triage(
        [site(S.TONGUE_DORSUM, regions)], [], patient(30), risk(), rules or default_rules()
    )
    assert_invariants(out)
    return out


def test_blood_overlap_is_not_discounted():
    mild = region(mean_rel=0.7, fraction=0.02)
    plain = _triage([mild]).lesions[0]
    red = _triage([region(mean_rel=0.7, fraction=0.02, confounders=[(C.BLOOD, 0.8)])]).lesions[0]
    assert red.score == pytest.approx(plain.score)
    assert "af.confounders" not in codes(red.contributing_factors)


def test_red_area_with_fvl_is_at_least_moderate_for_lesion_and_patient():
    out = _triage([region(mean_rel=0.72, fraction=0.01, confounders=[(C.BLOOD, 0.5)])])
    lesion = out.lesions[0]
    assert lesion.score < 3.0  # points alone would be LOW
    assert lesion.risk_level == RiskLevel.MODERATE
    assert out.triage.overall_risk_level == RiskLevel.MODERATE
    floor = [f for f in out.triage.overrides_applied if f.code == "af.red_area_floor"]
    assert floor and floor[0].source == FactorSource.OVERRIDE
    assert "red patch" in floor[0].label


def test_small_red_overlap_does_not_trigger_floor():
    out = _triage([region(mean_rel=0.72, fraction=0.01, confounders=[(C.BLOOD, 0.05)])])
    assert out.lesions[0].risk_level == RiskLevel.LOW
    assert out.triage.overall_risk_level == RiskLevel.LOW


def test_red_area_floor_is_optional(tmp_path):
    def drop(d):
        del d["lesion"]["af"]["red_area_floor"]

    rules = write_rules(tmp_path, drop)
    out = _triage([region(mean_rel=0.72, fraction=0.01, confounders=[(C.BLOOD, 0.5)])], rules)
    assert out.triage.overall_risk_level == RiskLevel.LOW


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("floor", "LOW"),
        ("floor", "BOGUS"),
        ("confounder", "nope"),
        ("min_overlap", 2.0),
        ("label", "Diagnosis: erythroplakia"),
    ],
)
def test_red_area_floor_validation(key, value):
    data = rules_data()
    data["lesion"]["af"]["red_area_floor"][key] = value
    with pytest.raises(TriageRulesError):
        parse_triage_rules(data)


def test_red_area_floor_must_be_mapping():
    data = rules_data()
    data["lesion"]["af"]["red_area_floor"] = "yes"
    with pytest.raises(TriageRulesError):
        parse_triage_rules(data)
