"""Choosing the reference tissue that relative AF is measured against.

Absolute fluorescence varies between patients (age, keratinisation, pigmentation) and between
captures. The most trustworthy "normal" for a site is the same patient's mirror-image site
(contralateral), then a different site with similar lining mucosa (analogous), then the brighter
part of the same image (in-image). The in-image reference cannot see a loss that covers the whole
field of view, so it is labelled as less reliable and the triage rules down-weight it.

Safety rule: a paired site may only make the analysis MORE sensitive. Dividing by a reference
that is dimmer than this site's own (brighter) mucosa inflates relative AF and can hide a
real loss here (false reassurance). The paired side can be dimmer because of exposure or angle,
or because of a change on that side too (bilateral disease is common in gutka and areca users).
So the reference level is ``max(paired, own)``: a brighter paired site reveals a loss that
covers this whole field; a dimmer one is recorded in the note but never lowers the level.
"""

from __future__ import annotations

import math

from mukha_pipeline.schema.models import AFReference, ReferenceType
from mukha_pipeline.sites import OralSite, reference_site_for, site_label

# Kept only so older callers and configs that pass it keep working; it no longer has any effect
# (see resolve_reference).
DEFAULT_MIN_PAIRED_RATIO = 0.67

# Never "healthy": the device does not check that the comparison tissue is unaffected.
IN_IMAGE_SUFFIX = "compared with the brighter mucosa in the same image — less reliable."


def _positive(value: float | None) -> float | None:
    """The value as a float if it is a usable (finite, positive) reference level, else None."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _label(site: OralSite) -> str:
    text = site_label(site)
    return text[:1].lower() + text[1:]


def _in_image_note(paired: OralSite | None) -> str:
    if paired is None:
        return "No paired site captured; " + IN_IMAGE_SUFFIX
    return f"The patient's {_label(paired)} was not captured or not usable; " + IN_IMAGE_SUFFIX


def _paired_note(paired: OralSite, pairing: str | None, *, used_own: bool) -> str:
    name = _label(paired)
    if used_own:
        return (
            f"The patient's {name} looked dimmer than the brighter mucosa at this site "
            "(exposure or angle, or a change on that side too), so this site's own brighter "
            "level was used; a dimmer reference could hide a loss here."
        )
    if pairing == "contralateral":
        return f"Compared with the patient's own {name}."
    return (
        f"No mirror-image site exists; compared with the patient's own {name}, which has "
        "similar tissue."
    )


def resolve_reference(
    site: OralSite | str,
    own_reference: float | None,
    available: dict[OralSite, float],
    *,
    min_paired_ratio: float = DEFAULT_MIN_PAIRED_RATIO,
) -> AFReference:
    """Pick the reference green for ``site``: contralateral > analogous > in-image > none.

    ``available`` maps OTHER sites that passed QC to their reference green. When the paired
    (contralateral or analogous) site is available, the type is contralateral / analogous and
    ``reference_green = max(paired, own)``: a paired site can only raise the reference, never
    lower it (a lower reference inflates relative AF and could hide a real loss here). When the
    paired site read dimmer, the note says so and this site's own level is used. A brighter
    paired site at worst over-calls loss, which a clinician will check.

    ``min_paired_ratio`` is accepted for compatibility with older callers and configs and is
    ignored: a dimmer paired site is no longer rejected, it simply cannot lower the level.
    """
    del min_paired_ratio  # kept for compatibility; see the docstring
    site = OralSite(site)
    paired, pairing = reference_site_for(site)
    own = _positive(own_reference)
    paired_value = _positive(available.get(paired)) if paired is not None else None
    if paired is not None and paired_value is not None:
        used_own = own is not None and own > paired_value
        level = own if used_own and own is not None else paired_value
        return AFReference(
            type=(
                ReferenceType.CONTRALATERAL
                if pairing == "contralateral"
                else ReferenceType.ANALOGOUS_SITE
            ),
            reference_site=paired,
            reference_green=level,
            note=_paired_note(paired, pairing, used_own=used_own),
        )
    if own is not None:
        return AFReference(
            type=ReferenceType.IN_IMAGE_AUTO,
            reference_site=None,
            reference_green=own,
            note=_in_image_note(paired),
        )
    return AFReference(
        type=ReferenceType.NONE,
        reference_site=None,
        reference_green=None,
        note=(
            "No usable reference tissue, so fluorescence loss could not be measured at this "
            "site. Examine the site clinically."
        ),
    )
