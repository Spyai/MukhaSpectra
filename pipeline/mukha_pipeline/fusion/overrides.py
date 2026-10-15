"""Clinical red-flag floors: conditions under which the device may never report a lower level."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mukha_pipeline.fusion.factors import make_factor
from mukha_pipeline.fusion.lesions import visual_seen
from mukha_pipeline.fusion.rules import Override
from mukha_pipeline.schema.models import (
    RISK_ORDER,
    ContributingFactor,
    FactorSource,
    RiskFactors,
    RiskLevel,
    SiteResult,
    VisualAppearance,
)


@dataclass(frozen=True)
class ClinicalContext:
    """What an override condition can look at."""

    risk_factors: RiskFactors
    appearances: frozenset[VisualAppearance]
    any_lesion: bool
    lesion_seen: bool = False  # the health worker saw something, with or without an appearance


def site_appearances(sites: Iterable[SiteResult]) -> frozenset[VisualAppearance]:
    """Appearances recorded at any site, examined or reference-only.

    A red patch noted at a site that was meant to be the comparison (reference) site is still a
    red patch: the worker's observation is never discarded because of the site's role.
    """
    return frozenset(a for s in sites for a in s.visual_flag.appearance)


def any_lesion_seen(sites: Iterable[SiteResult]) -> bool:
    """True when the health worker recorded a lesion at any site (ticked or described)."""
    return any(visual_seen(s.visual_flag) for s in sites)


def override_holds(override: Override, ctx: ClinicalContext) -> bool:
    """True when every condition set on the override holds."""
    if override.symptom is not None and not getattr(ctx.risk_factors.symptoms, override.symptom):
        return False
    if override.history is not None and not getattr(ctx.risk_factors, override.history):
        return False
    if override.visual_appearance is not None and not (
        override.visual_appearance & ctx.appearances
    ):
        return False
    if override.visual_lesion_seen is not None and override.visual_lesion_seen != ctx.lesion_seen:
        return False
    return override.any_lesion is None or override.any_lesion == ctx.any_lesion


def matching_overrides(
    overrides: Iterable[Override], ctx: ClinicalContext, *, visual_only: bool = False
) -> list[Override]:
    """Overrides whose conditions hold. ``visual_only`` keeps those conditioned on the health
    worker's observation, which also apply to the single lesion carrying that observation."""
    return [o for o in overrides if (not visual_only or o.lesion_scoped) and override_holds(o, ctx)]


def highest_floor(overrides: Iterable[Override]) -> RiskLevel | None:
    floors = [o.floor for o in overrides]
    return max(floors, key=RISK_ORDER.__getitem__, default=None)


def override_factor(override: Override, scope: str) -> ContributingFactor:
    """Zero-point record of a floor so the report can show why the level is what it is."""
    return make_factor(
        override.code,
        override.label,
        FactorSource.OVERRIDE,
        0.0,
        f"Clinical red flag: {scope} level is at least {override.floor.value}.",
    )
