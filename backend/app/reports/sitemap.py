"""Oral site map as SVG, drawn from the shared geometry in ``mukha_pipeline.sites``.

Viewed facing the patient: the patient's RIGHT is on the viewer's LEFT. Used by the clinician
PDF and by the API endpoint ``/screenings/{id}/sitemap.svg``.
"""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from mukha_pipeline.schema.models import RiskLevel, ScreeningResult, SiteResult, SiteRole
from mukha_pipeline.sites import SITE_MAP_VIEWBOX, SITES, OralSite

from app.reports.labels import RISK_COLORS, RISK_TITLE
from app.reports.ranking import ordered_lesions

FONT_FAMILY = "Inter, 'DejaVu Sans', Arial, sans-serif"
INK = "#1d2939"
MUTED = "#667085"
HATCH_ID = "ms-hatch-inconclusive"
HATCH_FILL = f"url(#{HATCH_ID})"
REFERENCE_STROKE = "#2563eb"
REFERENCE_FILL = "#eaf1fe"
NOT_EXAMINED_STROKE = "#c4cbd5"
NOT_EXAMINED_FILL = "#ffffff"

_TOP = 30.0  # orientation strip above the map
_MAP_H = 332.0  # mouth outline extends slightly beyond the site viewbox
_LEGEND_H = 46.0


def site_fill(level: RiskLevel | str) -> str:
    """Fill used for an examined site at ``level`` (hatched pattern for INCONCLUSIVE)."""
    level = RiskLevel(level)
    if level == RiskLevel.INCONCLUSIVE:
        return HATCH_FILL
    return RISK_COLORS[level]["fill"]


def _defs() -> str:
    inc = RISK_COLORS[RiskLevel.INCONCLUSIVE]
    return (
        "<defs>"
        f'<pattern id="{HATCH_ID}" patternUnits="userSpaceOnUse" width="5" height="5" '
        'patternTransform="rotate(45)">'
        f'<rect width="5" height="5" fill="{inc["tint"]}"/>'
        f'<line x1="0" y1="0" x2="0" y2="5" stroke="#8d97a8" stroke-width="1.7"/>'
        "</pattern>"
        "</defs>"
    )


def _arrow(x: float, y: float, direction: int) -> str:
    """Small filled triangle pointing left (-1) or right (+1)."""
    tip = x + 5 * direction
    return (
        f'<path d="M{tip:.1f},{y:.1f} L{x:.1f},{y - 3.5:.1f} L{x:.1f},{y + 3.5:.1f} Z" '
        f'fill="{MUTED}"/>'
    )


def _orientation(width: float) -> str:
    y = 15.0
    return (
        '<g class="orientation" font-size="9.5">'
        + _arrow(16, y - 3.2, -1)
        + f'<text x="22" y="{y}" fill="{INK}" font-weight="600">Patient\'s right</text>'
        + f'<text x="{width / 2}" y="{y}" fill="{MUTED}" text-anchor="middle" '
        'font-size="8.5">(viewed facing the patient)</text>'
        + f'<text x="{width - 22}" y="{y}" fill="{INK}" font-weight="600" '
        "text-anchor=\"end\">Patient's left</text>"
        + _arrow(width - 16, y - 3.2, 1)
        + "</g>"
    )


def _mouth_outline() -> str:
    return (
        '<g class="mouth">'
        '<ellipse cx="200" cy="164" rx="194" ry="162" fill="#fbf2f1" stroke="#ebcdc9" '
        'stroke-width="1.6"/>'
        '<ellipse cx="200" cy="168" rx="166" ry="118" fill="#f8e9e7" stroke="none" '
        'opacity="0.7"/>'
        '<line x1="200" y1="8" x2="200" y2="322" stroke="#e3c3bf" stroke-width="0.8" '
        'stroke-dasharray="2 4"/>'
        "</g>"
    )


def _site_state(sr: SiteResult | None) -> tuple[str, str | None]:
    if sr is None:
        return "not_examined", None
    level = RiskLevel(sr.site_risk_level).value
    if sr.role == SiteRole.REFERENCE_ONLY:
        return "reference_only", level
    return "examined", level


def _site_group(site: OralSite, sr: SiteResult | None, n_lesions: int) -> str:
    info = SITES[site]
    e = info.shape
    state, level = _site_state(sr)

    dash = ""
    if state == "examined":
        assert level is not None
        colors = RISK_COLORS[RiskLevel(level)]
        fill = site_fill(level)
        stroke = colors["strong"]
        width = 2.6 if n_lesions else 1.4
        text_fill = colors["text"] if level != RiskLevel.INCONCLUSIVE else INK
        if level == RiskLevel.INCONCLUSIVE:
            dash = ' stroke-dasharray="3 2"'
        status = f"examined, {RISK_TITLE[RiskLevel(level)].lower()}"
    elif state == "reference_only":
        fill, stroke, width, text_fill = REFERENCE_FILL, REFERENCE_STROKE, 1.4, "#1e3a8a"
        dash = ' stroke-dasharray="4 3"'
        status = "reference only (healthy comparison)"
    else:
        fill, stroke, width, text_fill = NOT_EXAMINED_FILL, NOT_EXAMINED_STROKE, 1.0, "#98a2b3"
        status = "not examined"

    title = f"{info.label}: {status}"
    if n_lesions:
        title += f"; {n_lesions} flagged finding{'s' if n_lesions > 1 else ''}"

    narrow = e.rx < 20 and e.ry > 2 * e.rx
    font_size = 7.2 if (narrow or e.rx < 20) else 8.2
    weight = "600" if state != "not_examined" else "500"
    transform = f' transform="rotate(-90 {e.cx} {e.cy})"' if narrow else ""
    label = (
        f'<text x="{e.cx}" y="{e.cy + font_size * 0.36:.1f}" text-anchor="middle" '
        f'font-size="{font_size}" font-weight="{weight}" fill="{text_fill}"{transform}>'
        f"{escape(info.short_label)}</text>"
    )
    risk_attr = f' data-risk="{level}"' if level else ""
    return (
        f'<g id="site-{site.value}" class="site site-{state}" data-site="{site.value}" '
        f'data-state="{state}"{risk_attr} data-lesions="{n_lesions}">'
        f"<title>{escape(title)}</title>"
        f'<ellipse cx="{e.cx}" cy="{e.cy}" rx="{e.rx}" ry="{e.ry}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash}/>'
        f"{label}</g>"
    )


def _lesion_markers(result: ScreeningResult) -> str:
    out: list[str] = []
    per_site: dict[OralSite, int] = {}
    for number, lesion in enumerate(ordered_lesions(result), start=1):
        site = OralSite(lesion.site)
        e = SITES[site].shape
        k = per_site.get(site, 0)
        per_site[site] = k + 1
        x = e.cx + e.rx * 0.72
        y = e.cy - e.ry * 0.62 + k * 17
        color = RISK_COLORS[RiskLevel(lesion.risk_level)]["strong"]
        out.append(
            f'<g class="lesion-marker" data-lesion="{escape(lesion.lesion_id)}" '
            f'data-number="{number}"><title>{escape(f"Finding {number}")}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7.6" fill="#ffffff" stroke="{color}" '
            'stroke-width="2"/>'
            f'<text x="{x:.1f}" y="{y + 3.3:.1f}" text-anchor="middle" font-size="9" '
            f'font-weight="700" fill="{color}">{number}</text></g>'
        )
    return "".join(out)


def _legend(y0: float) -> str:
    items: list[str] = []
    col_w = 98.0
    entries: list[tuple[str, str, str, str]] = [
        (RISK_COLORS[lv]["fill"] if lv != RiskLevel.INCONCLUSIVE else HATCH_FILL,
         RISK_COLORS[lv]["strong"],
         "3 2" if lv == RiskLevel.INCONCLUSIVE else "",
         RISK_TITLE[lv])
        for lv in (RiskLevel.HIGH, RiskLevel.MODERATE, RiskLevel.LOW, RiskLevel.INCONCLUSIVE)
    ]
    entries += [
        (REFERENCE_FILL, REFERENCE_STROKE, "4 3", "Reference only"),
        (NOT_EXAMINED_FILL, NOT_EXAMINED_STROKE, "", "Not examined"),
    ]
    for i, (fill, stroke, dash, text) in enumerate(entries):
        row, col = divmod(i, 4)
        x = 12 + col * col_w
        y = y0 + 12 + row * 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        items.append(
            f'<ellipse cx="{x + 8}" cy="{y}" rx="8" ry="5.2" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1.3"{dash_attr}/>'
            f'<text x="{x + 21}" y="{y + 3}" font-size="8.6" fill="{INK}">{escape(text)}</text>'
        )
    # Lesion marker key in the last slot of the second row.
    x = 12 + 2 * col_w
    y = y0 + 12 + 18
    items.append(
        f'<circle cx="{x + 8}" cy="{y}" r="6.2" fill="#ffffff" stroke="{INK}" stroke-width="1.5"/>'
        f'<text x="{x + 8}" y="{y + 2.9}" text-anchor="middle" font-size="7.6" font-weight="700" '
        f'fill="{INK}">1</text>'
        f'<text x="{x + 21}" y="{y + 3}" font-size="8.6" fill="{INK}">Finding number</text>'
    )
    return (
        f'<g class="legend"><line x1="8" y1="{y0}" x2="392" y2="{y0}" stroke="#e4e7ec" '
        f'stroke-width="0.8"/>{"".join(items)}</g>'
    )


def render_site_map_svg(
    result: ScreeningResult | None = None, *, width: int = 400, show_legend: bool = True
) -> str:
    """Standalone SVG of the oral site map.

    Examined sites are filled by ``site_risk_level`` (LOW teal, MODERATE amber, HIGH red,
    INCONCLUSIVE grey hatched); reference-only sites have a blue dashed outline; sites not in the
    result are light grey outlines. Flagged findings carry numbered markers that match the
    finding numbers in the report. ``result=None`` draws an empty map (all sites not examined).
    """
    vb_w = float(SITE_MAP_VIEWBOX[2])
    height_units = _TOP + _MAP_H + (_LEGEND_H if show_legend else 0.0)
    height = round(width * height_units / vb_w, 1)

    by_site: dict[OralSite, SiteResult] = {}
    lesion_count: dict[OralSite, int] = {}
    if result is not None:
        by_site = {OralSite(s.site): s for s in result.sites}
        for lesion in result.lesions:
            site = OralSite(lesion.site)
            lesion_count[site] = lesion_count.get(site, 0) + 1

    # Draw large shapes first so small ones (retromolar, lateral tongue) stay on top.
    order = sorted(SITES, key=lambda s: -(SITES[s].shape.rx * SITES[s].shape.ry))
    sites_svg = "".join(
        _site_group(site, by_site.get(site), lesion_count.get(site, 0)) for site in order
    )
    markers = _lesion_markers(result) if result is not None else ""
    legend = _legend(_TOP + _MAP_H + 2) if show_legend else ""

    n_examined = sum(1 for s in by_site.values() if s.role == SiteRole.EXAMINED)
    aria = f"Oral site map, viewed facing the patient. {n_examined} site(s) examined."
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {vb_w:g} {height_units:g}" '
        f'role="img" aria-label={quoteattr(aria)} font-family="{FONT_FAMILY}">'
        f"<title>{escape(aria)}</title>"
        + _defs()
        + _orientation(vb_w)
        + f'<g class="map" transform="translate(0 {_TOP + 4:g})">'
        + _mouth_outline()
        + sites_svg
        + markers
        + "</g>"
        + legend
        + "</svg>"
    )
