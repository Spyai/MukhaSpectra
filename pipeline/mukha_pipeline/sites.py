"""Oral site catalogue: anatomy facts, contralateral pairing and site-map geometry.

The site map is drawn from the examiner's point of view, facing the patient, so the patient's
RIGHT side appears on the viewer's LEFT. Every renderer (PDF report, web UI) uses the geometry
below so the map looks the same everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class OralSite(StrEnum):
    UPPER_LIP_MUCOSA = "upper_lip_mucosa"
    LOWER_LIP_MUCOSA = "lower_lip_mucosa"
    BUCCAL_MUCOSA_LEFT = "buccal_mucosa_left"
    BUCCAL_MUCOSA_RIGHT = "buccal_mucosa_right"
    TONGUE_DORSUM = "tongue_dorsum"
    LATERAL_TONGUE_LEFT = "lateral_tongue_left"
    LATERAL_TONGUE_RIGHT = "lateral_tongue_right"
    VENTRAL_TONGUE_FLOOR_OF_MOUTH = "ventral_tongue_floor_of_mouth"
    HARD_PALATE = "hard_palate"
    SOFT_PALATE = "soft_palate"
    UPPER_GINGIVA = "upper_gingiva"
    LOWER_GINGIVA = "lower_gingiva"
    RETROMOLAR_LEFT = "retromolar_left"
    RETROMOLAR_RIGHT = "retromolar_right"


Pairing = Literal["contralateral", "analogous"]


@dataclass(frozen=True)
class Ellipse:
    """Site shape on the 400 x 320 site-map canvas."""

    cx: float
    cy: float
    rx: float
    ry: float


@dataclass(frozen=True)
class SiteInfo:
    site: OralSite
    label: str
    short_label: str
    # Reference site for AF normalisation. "contralateral" = mirror-image site (same tissue).
    # "analogous" = different site with similar mucosa (e.g. upper vs lower labial mucosa).
    paired_site: OralSite | None
    pairing: Pairing | None
    # Keratinised mucosa fluoresces differently from lining mucosa; FVL there is less specific.
    keratinized: bool
    shape: Ellipse


SITE_MAP_VIEWBOX = (0, 0, 400, 320)

_S = OralSite

SITES: dict[OralSite, SiteInfo] = {
    s.site: s
    for s in [
        SiteInfo(_S.UPPER_LIP_MUCOSA, "Upper lip mucosa", "U lip", _S.LOWER_LIP_MUCOSA,
                 "analogous", False, Ellipse(200, 28, 92, 16)),
        SiteInfo(_S.UPPER_GINGIVA, "Upper gingiva", "U gum", _S.LOWER_GINGIVA,
                 "analogous", True, Ellipse(200, 60, 112, 12)),
        SiteInfo(_S.HARD_PALATE, "Hard palate", "H palate", None, None, True,
                 Ellipse(200, 98, 70, 22)),
        SiteInfo(_S.SOFT_PALATE, "Soft palate", "S palate", None, None, False,
                 Ellipse(200, 136, 54, 13)),
        SiteInfo(_S.RETROMOLAR_RIGHT, "Right retromolar area", "R retro", _S.RETROMOLAR_LEFT,
                 "contralateral", True, Ellipse(112, 132, 17, 14)),
        SiteInfo(_S.RETROMOLAR_LEFT, "Left retromolar area", "L retro", _S.RETROMOLAR_RIGHT,
                 "contralateral", True, Ellipse(288, 132, 17, 14)),
        SiteInfo(_S.BUCCAL_MUCOSA_RIGHT, "Right buccal mucosa", "R cheek", _S.BUCCAL_MUCOSA_LEFT,
                 "contralateral", False, Ellipse(62, 170, 30, 72)),
        SiteInfo(_S.BUCCAL_MUCOSA_LEFT, "Left buccal mucosa", "L cheek", _S.BUCCAL_MUCOSA_RIGHT,
                 "contralateral", False, Ellipse(338, 170, 30, 72)),
        SiteInfo(_S.TONGUE_DORSUM, "Tongue dorsum", "Tongue top", None, None, True,
                 Ellipse(200, 190, 56, 38)),
        SiteInfo(_S.LATERAL_TONGUE_RIGHT, "Right lateral tongue", "R tongue side",
                 _S.LATERAL_TONGUE_LEFT, "contralateral", False, Ellipse(128, 194, 13, 36)),
        SiteInfo(_S.LATERAL_TONGUE_LEFT, "Left lateral tongue", "L tongue side",
                 _S.LATERAL_TONGUE_RIGHT, "contralateral", False, Ellipse(272, 194, 13, 36)),
        SiteInfo(_S.VENTRAL_TONGUE_FLOOR_OF_MOUTH, "Ventral tongue / floor of mouth",
                 "Under tongue", None, None, False, Ellipse(200, 246, 62, 13)),
        SiteInfo(_S.LOWER_GINGIVA, "Lower gingiva", "L gum", _S.UPPER_GINGIVA, "analogous", True,
                 Ellipse(200, 274, 112, 12)),
        SiteInfo(_S.LOWER_LIP_MUCOSA, "Lower lip mucosa", "L lip", _S.UPPER_LIP_MUCOSA,
                 "analogous", False, Ellipse(200, 300, 92, 16)),
    ]
}

assert set(SITES) == set(OralSite), "every OralSite needs a SiteInfo entry"


def site_info(site: OralSite | str) -> SiteInfo:
    return SITES[OralSite(site)]


def site_label(site: OralSite | str) -> str:
    return site_info(site).label


def reference_site_for(site: OralSite | str) -> tuple[OralSite | None, Pairing | None]:
    """Return the preferred AF reference site and the kind of pairing, if any."""
    info = site_info(site)
    return info.paired_site, info.pairing
