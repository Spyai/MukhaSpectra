"""Pair each lesion reading with the same patient's reference reading.

Comparing a lesion with the patient's own reference mucosa cancels most between-patient variation
(pigmentation, epithelial thickness, probe pressure) and, because the comparison divides lesion by
reference channel by channel, the probe's LED spectrum and filter responses (see
``features.reference_delta``). The preferred reference is the mirror-image (contralateral) site,
else the analogous site from ``sites.reference_site_for``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mukha_pipeline.sites import OralSite, Pairing, reference_site_for
from mukha_pipeline.types import SpectralInput


@dataclass(frozen=True)
class PairingEntry:
    reading_id: str
    site: OralSite
    role: str  # "lesion" | "reference"
    reference_reading_id: str | None = None


@dataclass(frozen=True)
class AutoPair:
    reference_reading_id: str
    pairing: Pairing


def auto_pair(entries: Iterable[PairingEntry]) -> dict[str, AutoPair]:
    """Lesion reading id -> reference chosen automatically.

    Only lesions without an explicit ``reference_reading_id`` are paired, and only when exactly
    one reference reading exists at the paired site; ambiguity is left unresolved rather than
    guessed, because a wrong reference can hide or invent a difference.
    """
    items = list(entries)
    by_site: dict[OralSite, list[str]] = {}
    for e in items:
        if e.role == "reference":
            by_site.setdefault(e.site, []).append(e.reading_id)
    out: dict[str, AutoPair] = {}
    for e in items:
        if e.role != "lesion" or e.reference_reading_id:
            continue
        paired_site, pairing = reference_site_for(e.site)
        if paired_site is None or pairing is None:
            continue
        candidates = by_site.get(paired_site, [])
        if len(candidates) == 1:
            out[e.reading_id] = AutoPair(candidates[0], pairing)
    return out


def auto_pair_inputs(inputs: Sequence[SpectralInput]) -> dict[str, AutoPair]:
    """:func:`auto_pair` for already-built ``SpectralInput`` objects."""
    return auto_pair(
        PairingEntry(i.reading_id, OralSite(i.site), i.role, i.reference_reading_id) for i in inputs
    )
