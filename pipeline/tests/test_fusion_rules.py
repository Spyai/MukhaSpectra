"""Triage rule parsing: the bundled YAML is valid and every malformed edit fails loudly."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fusion_helpers import rules_data, write_rules

from mukha_pipeline.fusion import TriageRulesError, parse_triage_rules
from mukha_pipeline.fusion.rules import Tier, TierRule
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import ActionCode, ReferralUrgency, RiskLevel
from mukha_pipeline.sites import OralSite


def test_bundled_rules_parse():
    rules = parse_triage_rules(rules_data())
    assert rules.status == "baseline_heuristic_unvalidated"
    assert (rules.moderate_min_score, rules.high_min_score) == (3.0, 6.0)
    assert rules.actions[RiskLevel.HIGH].code == ActionCode.URGENT_REFERRAL
    assert rules.actions[RiskLevel.INCONCLUSIVE].referral_urgency == ReferralUrgency.RETAKE
    assert OralSite.LOWER_GINGIVA in rules.site.high_risk_sites
    # Gingivobuccal complex: high-risk, and not discounted as a low-specificity AF site.
    assert OralSite.LOWER_GINGIVA not in rules.site.af_low_specificity_sites
    assert rules.version == "1.2.0"
    assert {o.code for o in rules.overrides} >= {"visual.red_patch", "symptom.non_healing_ulcer"}


def test_bundled_rule_text_is_positioning_clean():
    rules = parse_triage_rules(rules_data())
    texts = [o.label for o in rules.overrides]
    for action in rules.actions.values():
        texts.extend([action.summary, *action.steps])
    assert [v for t in texts for v in find_positioning_violations(t)] == []
    assert "does not rule out" in rules.actions[RiskLevel.LOW].summary


def test_level_for_score_boundaries():
    rules = parse_triage_rules(rules_data())
    assert rules.level_for_score(0.0) == RiskLevel.LOW
    assert rules.level_for_score(2.9999) == RiskLevel.LOW
    assert rules.level_for_score(3.0) == RiskLevel.MODERATE
    assert rules.level_for_score(5.99) == RiskLevel.MODERATE
    assert rules.level_for_score(6.0) == RiskLevel.HIGH


def test_tiers_take_highest_met_not_sum():
    up = TierRule("min", (Tier(0.03, 0.5), Tier(0.06, 1.0)))
    assert up.best(0.01) is None
    assert up.best(0.03).points == 0.5  # type: ignore[union-attr]
    assert up.best(0.2).points == 1.0  # type: ignore[union-attr]
    down = TierRule("max", (Tier(-0.05, 0.5), Tier(-0.10, 1.0)))
    assert down.best(0.0) is None
    assert down.best(-0.07).points == 0.5  # type: ignore[union-attr]
    assert down.best(-0.10000000001).points == 1.0  # type: ignore[union-attr]


def test_single_string_visual_appearance_is_accepted():
    data = rules_data()
    data["overrides"][1]["when"]["visual_appearance"] = "red_patch"
    rules = parse_triage_rules(data)
    assert rules.overrides[1].visual_appearance == frozenset({"red_patch"})


def test_modified_yaml_loads_from_path(tmp_path):
    loaded = write_rules(tmp_path, lambda d: d["levels"].update(moderate_min_score=1.0))
    assert parse_triage_rules(loaded.data).moderate_min_score == 1.0
    assert loaded.ref.name == "mukhaspectra-triage-rules"


def _set(path: str, value: Any) -> Callable[[dict], None]:
    def mutate(d: dict) -> None:
        keys = path.split(".")
        node: Any = d
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value

    return mutate


def _delete(path: str) -> Callable[[dict], None]:
    def mutate(d: dict) -> None:
        keys = path.split(".")
        node: Any = d
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
        del node[keys[-1]]

    return mutate


BAD_EDITS: dict[str, Callable[[dict], None]] = {
    "missing_levels": _delete("levels"),
    "high_not_above_moderate": _set("levels.high_min_score", 3.0),
    "threshold_not_number": _set("levels.moderate_min_score", "three"),
    "threshold_bool": _set("levels.moderate_min_score", True),
    "threshold_zero": _set("levels.moderate_min_score", 0),
    "unknown_override_key": _set("overrides.0.when", {"symptomz": "neck_lump"}),
    "unknown_symptom": _set("overrides.0.when", {"symptom": "headache"}),
    "unknown_history": _set("overrides.5.when", {"history": "prior_stroke"}),
    "any_lesion_not_bool": _set("overrides.4.when", {"symptom": "lump", "any_lesion": "yes"}),
    "empty_when": _set("overrides.0.when", {}),
    "when_not_mapping": _set("overrides.0.when", ["symptom"]),
    "override_not_mapping": _set("overrides.0", "neck_lump"),
    "overrides_not_list": _set("overrides", {"a": 1}),
    "floor_low": _set("overrides.0.floor", "LOW"),
    "floor_unknown": _set("overrides.0.floor", "SEVERE"),
    "bad_appearance": _set("overrides.1.when", {"visual_appearance": ["purple_patch"]}),
    "empty_appearance": _set("overrides.1.when", {"visual_appearance": []}),
    "override_label_positioning": _set("overrides.0.label", "Diagnosis of oral cancer"),
    "duplicate_override_code": _set("overrides.1.code", "symptom.non_healing_ulcer"),
    "action_step_positioning": _set("actions.HIGH.steps", ["Lesion looks malignant."]),
    "action_summary_empty": _set("actions.LOW.summary", "  "),
    "action_unknown_code": _set("actions.LOW.code", "do_nothing"),
    "action_unknown_urgency": _set("actions.LOW.referral_urgency", "whenever"),
    "action_steps_not_list": _set("actions.LOW.steps", "Re-screen"),
    "action_negative_timeline": _set("actions.LOW.timeline_days", -1),
    "action_missing_level": _delete("actions.INCONCLUSIVE"),
    "unknown_high_risk_site": _set("lesion.site.high_risk_sites", ["left_ear"]),
    "sites_not_list": _set("lesion.site.high_risk_sites", "buccal_mucosa_left"),
    "tier_missing_points": _set("lesion.af.area_mm2", [{"min": 10.0}]),
    "tier_wrong_direction": _set("lesion.spectral.ratio_555_630_delta", [{"min": 1, "points": 1}]),
    "tiers_empty": _set("lesion.af.area_mm2", []),
    "reference_multiplier_missing": _delete("lesion.af.multipliers.reference.none"),
    "reference_multiplier_unknown": _set(
        "lesion.af.multipliers.reference",
        {**rules_data()["lesion"]["af"]["multipliers"]["reference"], "telepathy": 1.0},
    ),
    "reference_multiplier_zero": _set("lesion.af.multipliers.reference.contralateral", 0),
    "multipliers_not_mapping": _set("lesion.af.multipliers.diascopy", [1.0]),
    "min_multiplier_above_one": _set("lesion.af.confounder_discount.min_multiplier", 1.5),
    "discount_above_one": _set("lesion.af.confounder_discount.glare", 1.5),
    "wl_min_probability": _set("lesion.wl_model.min_probability", 1.5),
    "wl_points_missing_label": _delete("lesion.wl_model.points.other"),
    "unknown_risk_factor": _set("patient.risk_factor_points.coffee", 1.0),
    "missing_risk_factor": _delete("patient.risk_factor_points.age_40_plus"),
    "min_sites_zero": _set("inconclusive.min_sites_analysed", 0),
    "min_sites_bool": _set("inconclusive.min_sites_analysed", True),
}


@pytest.mark.parametrize("name", sorted(BAD_EDITS))
def test_malformed_rules_fail_loudly(name):
    data = rules_data()
    BAD_EDITS[name](data)
    with pytest.raises(TriageRulesError):
        parse_triage_rules(data)


def test_top_level_must_be_mapping():
    with pytest.raises(TriageRulesError):
        parse_triage_rules(["not", "a", "mapping"])  # type: ignore[arg-type]
