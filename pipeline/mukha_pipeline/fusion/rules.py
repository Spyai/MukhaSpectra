"""Typed, validated view of ``triage_rules.v1.yaml``.

The YAML is edited by clinicians, so every key is checked when it is parsed: a typo must fail
loudly at load time rather than silently drop a rule (a dropped red-flag floor is exactly the kind
of false reassurance this system exists to prevent). User-facing text in the rules (override
labels, action summaries and steps) is also checked against the positioning guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import (
    ActionCode,
    ConfounderType,
    Diascopy,
    ReferenceType,
    ReferralUrgency,
    RiskLevel,
    Symptoms,
    VisualAppearance,
    WLLabel,
)
from mukha_pipeline.sites import OralSite

HISTORY_KEYS: tuple[str, ...] = ("prior_oral_cancer", "prior_oral_lesion")
RISK_FACTOR_KEYS: tuple[str, ...] = (
    "smokeless_tobacco",
    "smoking",
    "areca_nut",
    "alcohol_daily",
    "prior_oral_lesion",
    "prior_oral_cancer",
    "age_40_plus",
)
OVERRIDE_KEYS: frozenset[str] = frozenset({"symptom", "visual_appearance", "history", "any_lesion"})
FLOOR_LEVELS: frozenset[RiskLevel] = frozenset({RiskLevel.MODERATE, RiskLevel.HIGH})
_EPS = 1e-9


class TriageRulesError(ValueError):
    """The triage rule file is malformed or would produce non-compliant text."""


@dataclass(frozen=True)
class Tier:
    bound: float
    points: float


@dataclass(frozen=True)
class TierRule:
    """Stepped points. ``min`` tiers are met when value >= bound, ``max`` tiers when value <=
    bound. The highest-scoring tier met wins, so tiers never add up."""

    direction: Literal["min", "max"]
    tiers: tuple[Tier, ...]

    def best(self, value: float) -> Tier | None:
        if self.direction == "min":
            met = [t for t in self.tiers if value >= t.bound - _EPS]
        else:
            met = [t for t in self.tiers if value <= t.bound + _EPS]
        return max(met, key=lambda t: t.points, default=None)


@dataclass(frozen=True)
class MinPoints:
    minimum: float
    points: float


@dataclass(frozen=True)
class AFRules:
    fvl_present: float
    depth_points_per_0_1: float
    depth_max: float
    area_fraction: TierRule
    area_mm2: TierRule
    border_irregularity: MinPoints
    contrast_vs_surround: MinPoints
    reference_multipliers: dict[ReferenceType, float]
    diascopy_multipliers: dict[Diascopy, float]
    confounder_discount: dict[ConfounderType, float]
    min_confounder_multiplier: float


@dataclass(frozen=True)
class WLRules:
    min_probability: float
    points: dict[WLLabel, float]


@dataclass(frozen=True)
class SpectralRules:
    hb_absorption_proxy_delta: TierRule
    ratio_555_630_delta: TierRule
    no_reference_multiplier: float


@dataclass(frozen=True)
class SiteRules:
    high_risk_sites: frozenset[OralSite]
    high_risk_multiplier: float
    af_low_specificity_sites: frozenset[OralSite]
    af_low_specificity_multiplier: float


@dataclass(frozen=True)
class PatientRules:
    risk_factor_points: dict[str, float]
    risk_factor_cap: float
    former_use_fraction: float


@dataclass(frozen=True)
class Override:
    """A clinical red-flag floor. Every condition that is set must hold."""

    code: str
    label: str
    floor: RiskLevel
    symptom: str | None = None
    visual_appearance: frozenset[VisualAppearance] | None = None
    history: str | None = None
    any_lesion: bool | None = None


@dataclass(frozen=True)
class InconclusiveRules:
    low_requires_all_examined_sites_analysed: bool
    min_sites_analysed: int


@dataclass(frozen=True)
class ActionTemplate:
    code: ActionCode
    summary: str
    steps: tuple[str, ...]
    timeline_days: int | None
    referral_urgency: ReferralUrgency


@dataclass(frozen=True)
class TriageRules:
    name: str
    version: str
    status: str
    moderate_min_score: float
    high_min_score: float
    af: AFRules
    wl: WLRules
    spectral: SpectralRules
    visual_points: dict[VisualAppearance, float]
    site: SiteRules
    patient: PatientRules
    overrides: tuple[Override, ...]
    inconclusive: InconclusiveRules
    actions: dict[RiskLevel, ActionTemplate]

    def level_for_score(self, score: float) -> RiskLevel:
        """Map a rule score onto LOW / MODERATE / HIGH using the configured thresholds."""
        if score >= self.high_min_score - _EPS:
            return RiskLevel.HIGH
        if score >= self.moderate_min_score - _EPS:
            return RiskLevel.MODERATE
        return RiskLevel.LOW


# --------------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------------


def _get(data: Any, path: str) -> Any:
    node = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise TriageRulesError(f"triage rules: missing key '{path}'")
        node = node[key]
    return node


def _num(data: Any, path: str, *, positive: bool = False) -> float:
    value = _get(data, path)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TriageRulesError(f"triage rules: '{path}' must be a number, got {value!r}")
    if positive and value <= 0:
        raise TriageRulesError(f"triage rules: '{path}' must be > 0, got {value!r}")
    return float(value)


def _enum_map(data: Any, path: str, enum: type, *, positive: bool = False) -> dict:
    raw = _get(data, path)
    if not isinstance(raw, dict):
        raise TriageRulesError(f"triage rules: '{path}' must be a mapping")
    out: dict = {}
    for key in raw:
        try:
            member = enum(key)
        except ValueError as exc:
            raise TriageRulesError(f"triage rules: unknown key '{key}' in '{path}'") from exc
        out[member] = _num(raw, str(key), positive=positive)
    missing = [m.value for m in enum if m not in out]
    if missing:
        raise TriageRulesError(f"triage rules: '{path}' lacks entries for {missing}")
    return out


def _tiers(data: Any, path: str, direction: Literal["min", "max"]) -> TierRule:
    raw = _get(data, path)
    if not isinstance(raw, list) or not raw:
        raise TriageRulesError(f"triage rules: '{path}' must be a non-empty list of tiers")
    tiers = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {direction, "points"}:
            raise TriageRulesError(
                f"triage rules: tier {i} of '{path}' needs exactly '{direction}' and 'points'"
            )
        tiers.append(Tier(bound=_num(item, direction), points=_num(item, "points")))
    return TierRule(direction=direction, tiers=tuple(tiers))


def _min_points(data: Any, path: str) -> MinPoints:
    return MinPoints(minimum=_num(data, f"{path}.min"), points=_num(data, f"{path}.points"))


def _sites(data: Any, path: str) -> frozenset[OralSite]:
    raw = _get(data, path)
    if not isinstance(raw, list):
        raise TriageRulesError(f"triage rules: '{path}' must be a list of sites")
    try:
        return frozenset(OralSite(s) for s in raw)
    except ValueError as exc:
        raise TriageRulesError(f"triage rules: unknown site in '{path}': {exc}") from exc


def _check_text(text: Any, where: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise TriageRulesError(f"triage rules: '{where}' must be non-empty text")
    violations = find_positioning_violations(text)
    if violations:
        raise TriageRulesError(f"triage rules: '{where}' breaks positioning rules: {violations}")
    return text


# --------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------


def _af_rules(af: dict) -> AFRules:
    min_mult = _num(af, "confounder_discount.min_multiplier", positive=True)
    if min_mult > 1.0:
        raise TriageRulesError("triage rules: 'confounder_discount.min_multiplier' must be <= 1")
    discounts_raw = {
        k: v for k, v in _get(af, "confounder_discount").items() if k != "min_multiplier"
    }
    discounts = _enum_map({"d": discounts_raw}, "d", ConfounderType)
    if any(not 0.0 <= d <= 1.0 for d in discounts.values()):
        raise TriageRulesError("triage rules: confounder discounts must lie in [0, 1]")
    return AFRules(
        fvl_present=_num(af, "fvl_present"),
        depth_points_per_0_1=_num(af, "depth.points_per_0_1"),
        depth_max=_num(af, "depth.max"),
        area_fraction=_tiers(af, "area_fraction_of_mucosa", "min"),
        area_mm2=_tiers(af, "area_mm2", "min"),
        border_irregularity=_min_points(af, "border_irregularity"),
        contrast_vs_surround=_min_points(af, "contrast_vs_surround"),
        reference_multipliers=_enum_map(af, "multipliers.reference", ReferenceType, positive=True),
        diascopy_multipliers=_enum_map(af, "multipliers.diascopy", Diascopy, positive=True),
        confounder_discount=discounts,
        min_confounder_multiplier=min_mult,
    )


def _override(raw: Any, index: int) -> Override:
    where = f"overrides[{index}]"
    if not isinstance(raw, dict):
        raise TriageRulesError(f"triage rules: '{where}' must be a mapping")
    code = _get(raw, "code")
    label = _check_text(_get(raw, "label"), f"{where}.label")
    try:
        floor = RiskLevel(_get(raw, "floor"))
    except ValueError as exc:
        raise TriageRulesError(f"triage rules: '{where}.floor' is not a risk level") from exc
    if floor not in FLOOR_LEVELS:
        raise TriageRulesError(f"triage rules: '{where}.floor' must be MODERATE or HIGH")
    when = _get(raw, "when")
    if not isinstance(when, dict) or not when:
        raise TriageRulesError(f"triage rules: '{where}.when' must be a non-empty mapping")
    unknown = set(when) - OVERRIDE_KEYS
    if unknown:
        raise TriageRulesError(f"triage rules: '{where}.when' has unknown keys {sorted(unknown)}")
    symptom = when.get("symptom")
    if symptom is not None and symptom not in Symptoms.model_fields:
        raise TriageRulesError(f"triage rules: '{where}' unknown symptom '{symptom}'")
    history = when.get("history")
    if history is not None and history not in HISTORY_KEYS:
        raise TriageRulesError(f"triage rules: '{where}' unknown history '{history}'")
    appearance = when.get("visual_appearance")
    visual: frozenset[VisualAppearance] | None = None
    if appearance is not None:
        items = [appearance] if isinstance(appearance, str) else appearance
        try:
            visual = frozenset(VisualAppearance(a) for a in items)
        except (ValueError, TypeError) as exc:
            raise TriageRulesError(f"triage rules: '{where}' bad visual_appearance") from exc
        if not visual:
            raise TriageRulesError(f"triage rules: '{where}' visual_appearance is empty")
    any_lesion = when.get("any_lesion")
    if any_lesion is not None and not isinstance(any_lesion, bool):
        raise TriageRulesError(f"triage rules: '{where}.when.any_lesion' must be true or false")
    return Override(
        code=str(code),
        label=label,
        floor=floor,
        symptom=symptom,
        visual_appearance=visual,
        history=history,
        any_lesion=any_lesion,
    )


def _action(raw: Any, level: RiskLevel) -> ActionTemplate:
    where = f"actions.{level.value}"
    try:
        code = ActionCode(_get(raw, "code"))
        urgency = ReferralUrgency(_get(raw, "referral_urgency"))
    except ValueError as exc:
        raise TriageRulesError(f"triage rules: '{where}' has an unknown code or urgency") from exc
    steps = _get(raw, "steps")
    if not isinstance(steps, list):
        raise TriageRulesError(f"triage rules: '{where}.steps' must be a list")
    timeline = raw.get("timeline_days")
    if timeline is not None and (not isinstance(timeline, int) or timeline < 0):
        raise TriageRulesError(f"triage rules: '{where}.timeline_days' must be an int >= 0")
    return ActionTemplate(
        code=code,
        summary=_check_text(_get(raw, "summary"), f"{where}.summary"),
        steps=tuple(_check_text(s, f"{where}.steps") for s in steps),
        timeline_days=timeline,
        referral_urgency=urgency,
    )


def parse_triage_rules(data: dict[str, Any]) -> TriageRules:
    """Validate a loaded ``triage_rules`` YAML dict and return its typed form."""
    if not isinstance(data, dict):
        raise TriageRulesError("triage rules: top level must be a mapping")
    moderate = _num(data, "levels.moderate_min_score", positive=True)
    high = _num(data, "levels.high_min_score", positive=True)
    if high <= moderate:
        raise TriageRulesError("triage rules: high_min_score must be above moderate_min_score")

    lesion = _get(data, "lesion")
    wl_points = _enum_map(lesion, "wl_model.points", WLLabel)
    min_prob = _num(lesion, "wl_model.min_probability")
    if not 0.0 <= min_prob <= 1.0:
        raise TriageRulesError("triage rules: 'wl_model.min_probability' must lie in [0, 1]")

    rf_raw = _get(data, "patient.risk_factor_points")
    unknown_rf = set(rf_raw) - set(RISK_FACTOR_KEYS) if isinstance(rf_raw, dict) else set()
    if unknown_rf:
        raise TriageRulesError(f"triage rules: unknown risk factors {sorted(unknown_rf)}")
    rf_points = {k: _num(data, f"patient.risk_factor_points.{k}") for k in RISK_FACTOR_KEYS}

    overrides_raw = _get(data, "overrides")
    if not isinstance(overrides_raw, list):
        raise TriageRulesError("triage rules: 'overrides' must be a list")
    overrides = tuple(_override(o, i) for i, o in enumerate(overrides_raw))
    codes = [o.code for o in overrides]
    if len(set(codes)) != len(codes):
        raise TriageRulesError("triage rules: override codes must be unique")

    actions_raw = _get(data, "actions")
    actions = {lvl: _action(_get(actions_raw, lvl.value), lvl) for lvl in RiskLevel}

    min_sites = _get(data, "inconclusive.min_sites_analysed")
    if not isinstance(min_sites, int) or isinstance(min_sites, bool) or min_sites < 1:
        raise TriageRulesError("triage rules: 'inconclusive.min_sites_analysed' must be >= 1")

    return TriageRules(
        name=str(data.get("name", "triage_rules")),
        version=str(data.get("version", "0")),
        status=str(data.get("status", "unknown")),
        moderate_min_score=moderate,
        high_min_score=high,
        af=_af_rules(_get(lesion, "af")),
        wl=WLRules(min_probability=min_prob, points=wl_points),
        spectral=SpectralRules(
            hb_absorption_proxy_delta=_tiers(lesion, "spectral.hb_absorption_proxy_delta", "min"),
            ratio_555_630_delta=_tiers(lesion, "spectral.ratio_555_630_delta", "max"),
            no_reference_multiplier=_num(lesion, "spectral.no_reference_multiplier"),
        ),
        visual_points=_enum_map(lesion, "clinical_visual.points", VisualAppearance),
        site=SiteRules(
            high_risk_sites=_sites(lesion, "site.high_risk_sites"),
            high_risk_multiplier=_num(lesion, "site.high_risk_multiplier", positive=True),
            af_low_specificity_sites=_sites(lesion, "site.af_low_specificity.sites"),
            af_low_specificity_multiplier=_num(
                lesion, "site.af_low_specificity.af_multiplier", positive=True
            ),
        ),
        patient=PatientRules(
            risk_factor_points=rf_points,
            risk_factor_cap=_num(data, "patient.risk_factor_cap"),
            former_use_fraction=_num(data, "patient.former_use_fraction"),
        ),
        overrides=overrides,
        inconclusive=InconclusiveRules(
            low_requires_all_examined_sites_analysed=bool(
                _get(data, "inconclusive.low_requires_all_examined_sites_analysed")
            ),
            min_sites_analysed=min_sites,
        ),
        actions=actions,
    )
