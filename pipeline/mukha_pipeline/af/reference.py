"""Choosing the reference tissue that relative AF is measured against.

Absolute fluorescence varies between patients (age, keratinisation, pigmentation) and between
captures. The most trustworthy "normal" for a site is the same patient's mirror-image site
(contralateral), then a different site with similar lining mucosa (analogous), then the brighter
part of the same image (in-image). The in-image reference cannot see a loss that covers the whole
field of view, so it is labelled as less reliable and the triage rules down-weight it.
"""

from __future__ import annotations

import math

from mukha_pipeline.schema.models import AFReference, ReferenceType
from mukha_pipeline.sites import OralSite, reference_site_for, site_label

DEFAULT_MIN_PAIRED_RATIO = 0.67

IN_IMAGE_SUFFIX = (
    "compared with the brightest healthy-looking tissue in the same image — less reliable."
)


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


def _in_image_note(paired: OralSite | None, reason: str | None) -> str:
    if paired is None:
        return "No paired site captured; " + IN_IMAGE_SUFFIX
    if reason == "dim":
        return (
            f"The patient's {_label(paired)} looked much dimmer than this site (possible "
            "exposure difference or a change there), so it was not used; " + IN_IMAGE_SUFFIX
        )
    return f"The patient's {_label(paired)} was not captured or not usable; " + IN_IMAGE_SUFFIX


def resolve_reference(
    site: OralSite | str,
    own_reference: float | None,
    available: dict[OralSite, float],
    *,
    min_paired_ratio: float = DEFAULT_MIN_PAIRED_RATIO,
) -> AFReference:
    """Pick the reference green for ``site``: contralateral > analogous > in-image > none.

    ``available`` maps OTHER sites that passed QC to their reference green. Safety guard: a paired
    site much dimmer than this site's own tissue (paired < ``min_paired_ratio`` x own) is not
    used, because dividing by a too-low reference inflates relative AF and could hide a real loss
    here (false reassurance). A paired site that is brighter is always accepted: at worst it
    over-calls loss, which a clinician will check.
    """
    site = OralSite(site)
    paired, pairing = reference_site_for(site)
    own = _positive(own_reference)
    reason: str | None = None
    paired_value = _positive(available.get(paired)) if paired is not None else None
    if paired is not None and paired_value is not None:
        if own is not None and paired_value < min_paired_ratio * own:
            reason = "dim"
        elif pairing == "contralateral":
            return AFReference(
                type=ReferenceType.CONTRALATERAL,
                reference_site=paired,
                reference_green=paired_value,
                note=f"Compared with the patient's own {_label(paired)}.",
            )
        else:
            return AFReference(
                type=ReferenceType.ANALOGOUS_SITE,
                reference_site=paired,
                reference_green=paired_value,
                note=(
                    f"No mirror-image site exists; compared with the patient's own "
                    f"{_label(paired)}, which has similar tissue."
                ),
            )
    if own is not None:
        return AFReference(
            type=ReferenceType.IN_IMAGE_AUTO,
            reference_site=None,
            reference_green=own,
            note=_in_image_note(paired, reason),
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
