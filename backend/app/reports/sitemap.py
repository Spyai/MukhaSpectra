"""Oral site map as SVG, drawn from the shared geometry in ``mukha_pipeline.sites``.

Viewed facing the patient: the patient's RIGHT is on the viewer's LEFT. Used by the clinician
PDF and by the API endpoint ``/screenings/{id}/sitemap.svg``.

Print legibility: the report draws the 400-unit-wide map into a 74 mm column, so one unit is
0.185 mm. Every text is at least ``MIN_FONT`` units (>= 2.1 mm, about 6 pt) on paper. Risk is
never shown by colour alone: each examined site also carries the level icon (1/2/3 bars, "?").
"""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from app.reports.charts import risk_icon_shapes
from app.reports.labels import RISK_COLORS, RISK_TITLE
from app.reports.ranking import ordered_lesions
from mukha_pipeline.schema.models import RiskLevel, ScreeningResult, SiteResult, SiteRole
from mukha_pipeline.sites import SITE_MAP_VIEWBOX, SITES, OralSite

FONT_FAMILY = "Inter, 'DejaVu Sans', Arial, sans-serif"
INK = "#1d2939"
MUTED = "#667085"
HATCH_ID = "ms-hatch-inconclusive"
HATCH_FILL = f"url(#{HATCH_ID})"
REFERENCE_STROKE = "#2563eb"
REFERENCE_FILL = "#eaf1fe"
NOT_EXAMINED_STROKE = "#c4cbd5"
NOT_EXAMINED_FILL = "#ffffff"

LABEL_FONT = 12.5
NARROW_LABEL_FONT = 11.5
ORIENT_FONT = 13.0
ORIENT_SUB_FONT = 12.0
MARKER_FONT = 12.0
MARKER_R = 9.6
LEGEND_FONT = 12.0
ICON = 10.0  # level icon size (units)
MIN_FONT = min(LABEL_FONT, NARROW_LABEL_FONT, ORIENT_SUB_FONT, MARKER_FONT, LEGEND_FONT)

# The shared catalogue's short labels use "U"/"L" for upper/lower, which clash with "L" = left
# ("L cheek"): spell those out, so on the map "R"/"L" only ever mean the patient's right/left.
# The catalogue fix is requested from the pipeline owners.
LABEL_OVERRIDES: dict[OralSite, str] = {
    OralSite.UPPER_LIP_MUCOSA: "Upper lip",
    OralSite.LOWER_LIP_MUCOSA: "Lower lip",
    OralSite.UPPER_GINGIVA: "Upper gum",
    OralSite.LOWER_GINGIVA: "Lower gum",
    OralSite.HARD_PALATE: "Hard palate",
    OralSite.SOFT_PALATE: "Soft palate",
    # "R tongue side" is longer than the shape at a legible size.
    OralSite.LATERAL_TONGUE_RIGHT: "R lat. tongue",
    OralSite.LATERAL_TONGUE_LEFT: "L lat. tongue",
}

_TOP = 39.0  # orientation strip (two rows) above the map
_MAP_H = 332.0  # mouth outline extends slightly beyond the site viewbox
_LEGEND_H = 74.0
_CENTRE_X = SITE_MAP_VIEWBOX[2] / 2


def map_label(site: OralSite | str) -> str:
    """Short label printed on the map (upper/lower spelled out, never "L"/"U")."""
    site = OralSite(site)
    return LABEL_OVERRIDES.get(site, SITES[site].short_label)


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
    tip = x + 7 * direction
    return (
        f'<path d="M{tip:.1f},{y:.1f} L{x:.1f},{y - 4.5:.1f} L{x:.1f},{y + 4.5:.1f} Z" '
        f'fill="{INK}"/>'
    )


def _orientation(width: float) -> str:
    y1, y2 = 14.0, 31.0
    return (
        f'<g class="orientation" font-size="{ORIENT_FONT:g}">'
        + _arrow(12, y1 - 4.5, -1)
        + f'<text x="22" y="{y1}" fill="{INK}" font-weight="700">Patient\'s right</text>'
        + f'<text x="{width - 22}" y="{y1}" fill="{INK}" font-weight="700" '
        'text-anchor="end">Patient\'s left</text>'
        + _arrow(width - 12, y1 - 4.5, 1)
        + f'<text x="{width / 2}" y="{y2}" fill="{MUTED}" text-anchor="middle" '
        f'font-size="{ORIENT_SUB_FONT:g}">(viewed facing the patient)</text>' + "</g>"
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


def _text_width(text: str, size: float) -> float:
    """Rough rendered width of Inter text (good enough to centre a label with its icon)."""
    return sum(0.36 if ch in " il.-'" else (0.66 if ch.isupper() else 0.55) for ch in text) * size


def _icon(level: str, x: float, y: float, color: str) -> str:
    """Level icon (1/2/3 bars or "?") with its top-left corner at (x, y), ICON units square."""
    scale = ICON / 16
    return (
        f'<g class="level-icon" data-level="{level}" '
        f'transform="translate({x:.1f} {y:.1f}) scale({scale:g})">'
        f"{risk_icon_shapes(level, color)}</g>"
    )


def _is_narrow(site: OralSite) -> bool:
    e = SITES[site].shape
    return e.rx < 20 and e.ry > 2 * e.rx


def _is_small(site: OralSite) -> bool:
    return SITES[site].shape.rx < 20 and not _is_narrow(site)


def _site_group(site: OralSite, sr: SiteResult | None, n_lesions: int) -> str:
    info = SITES[site]
    e = info.shape
    state, level = _site_state(sr)

    dash = ""
    icon_color = INK
    if state == "examined":
        assert level is not None
        colors = RISK_COLORS[RiskLevel(level)]
        fill = site_fill(level)
        stroke = colors["strong"]
        width = 2.6 if n_lesions else 1.4
        text_fill = colors["text"] if level != RiskLevel.INCONCLUSIVE else INK
        icon_color = text_fill
        if level == RiskLevel.INCONCLUSIVE:
            dash = ' stroke-dasharray="3 2"'
        status = f"examined, {RISK_TITLE[RiskLevel(level)].lower()}"
    elif state == "reference_only":
        fill, stroke, width, text_fill = REFERENCE_FILL, REFERENCE_STROKE, 1.4, "#1e3a8a"
        dash = ' stroke-dasharray="4 3"'
        status = "reference only (healthy comparison)"
    else:
        fill, stroke, width, text_fill = NOT_EXAMINED_FILL, NOT_EXAMINED_STROKE, 1.0, "#7a8494"
        status = "not examined"

    title = f"{info.label}: {status}"
    if n_lesions:
        title += f"; {n_lesions} flagged finding{'s' if n_lesions > 1 else ''}"

    text = map_label(site)
    narrow, small = _is_narrow(site), _is_small(site)
    size = NARROW_LABEL_FONT if (narrow or small) else LABEL_FONT
    weight = "600" if state != "not_examined" else "400"
    with_icon = state == "examined"
    cap = 0.72 * size
    tx, ty = e.cx, e.cy + cap / 2
    icon = ""
    if with_icon:
        assert level is not None
        if narrow:  # rotated label fills the shape: icon just below it
            icon = _icon(
                level, e.cx - ICON / 2, e.cy + e.ry + 6, RISK_COLORS[RiskLevel(level)]["strong"]
            )
        elif small:  # retromolar: icon below the shape
            icon = _icon(
                level, e.cx - ICON / 2, e.cy + e.ry + 3, RISK_COLORS[RiskLevel(level)]["strong"]
            )
        elif e.ry >= 20:  # tall enough for two rows: label, then icon
            block = cap + 3 + ICON
            top = e.cy - block / 2
            ty = top + cap
            icon = _icon(level, e.cx - ICON / 2, ty + 3, icon_color)
        else:  # thin and wide: icon left of the label
            tw = _text_width(text, size)
            start = e.cx - (ICON + 4 + tw) / 2
            icon = _icon(level, start, e.cy - ICON / 2, icon_color)
            tx = start + ICON + 4 + tw / 2
    transform = f' transform="rotate(-90 {e.cx} {e.cy})"' if narrow else ""
    if narrow:
        tx, ty = e.cx, e.cy + cap / 2
    label = (
        f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
        f'font-size="{size:g}" font-weight="{weight}" fill="{text_fill}"{transform}>'
        f"{escape(text)}</text>"
    )
    risk_attr = f' data-risk="{level}"' if level else ""
    return (
        f'<g id="site-{site.value}" class="site site-{state}" data-site="{site.value}" '
        f'data-state="{state}"{risk_attr} data-lesions="{n_lesions}">'
        f"<title>{escape(title)}</title>"
        f'<ellipse cx="{e.cx}" cy="{e.cy}" rx="{e.rx}" ry="{e.ry}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash}/>'
        f"{label}{icon}</g>"
    )


def marker_position(site: OralSite, k: int) -> tuple[float, float]:
    """Centre of the ``k``-th (0-based) finding marker at ``site``, clear of its label."""
    e = SITES[site].shape
    side = 1.0 if e.cx >= _CENTRE_X else -1.0
    if _is_narrow(site):  # beside the shape, on the outer side
        return e.cx + side * (e.rx + MARKER_R + 1.5), e.cy - e.ry * 0.62 + k * 2.2 * MARKER_R
    if _is_small(site):  # above the shape
        return e.cx + k * 2.2 * MARKER_R, e.cy - e.ry - MARKER_R - 1.5
    if e.ry < 20:  # thin and wide: at the outer end
        return e.cx + e.rx - 2 - k * 2.2 * MARKER_R, e.cy - e.ry * 0.25
    return e.cx + e.rx * 0.72, e.cy - e.ry * 0.62 + k * 2.2 * MARKER_R


def _lesion_markers(result: ScreeningResult) -> str:
    out: list[str] = []
    per_site: dict[OralSite, int] = {}
    for number, lesion in enumerate(ordered_lesions(result), start=1):
        site = OralSite(lesion.site)
        k = per_site.get(site, 0)
        per_site[site] = k + 1
        x, y = marker_position(site, k)
        color = RISK_COLORS[RiskLevel(lesion.risk_level)]["strong"]
        out.append(
            f'<g class="lesion-marker" data-lesion="{escape(lesion.lesion_id)}" '
            f'data-number="{number}"><title>{escape(f"Finding {number}")}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{MARKER_R:g}" fill="#ffffff" '
            f'stroke="{color}" stroke-width="2.2"/>'
            f'<text x="{x:.1f}" y="{y + MARKER_FONT * 0.36:.1f}" text-anchor="middle" '
            f'font-size="{MARKER_FONT:g}" font-weight="700" fill="{color}">{number}</text></g>'
        )
    return "".join(out)


def _legend(y0: float) -> str:
    """Three-column legend: swatch + level icon + text (used by the standalone SVG)."""
    items: list[str] = []
    col_w = 128.0
    entries: list[tuple[str, str, str, str, str | None]] = [
        (
            RISK_COLORS[lv]["fill"] if lv != RiskLevel.INCONCLUSIVE else HATCH_FILL,
            RISK_COLORS[lv]["strong"],
            "3 2" if lv == RiskLevel.INCONCLUSIVE else "",
            RISK_TITLE[lv],
            lv.value,
        )
        for lv in (RiskLevel.HIGH, RiskLevel.MODERATE, RiskLevel.LOW, RiskLevel.INCONCLUSIVE)
    ]
    entries += [
        (REFERENCE_FILL, REFERENCE_STROKE, "4 3", "Reference only", None),
        (NOT_EXAMINED_FILL, NOT_EXAMINED_STROKE, "", "Not examined", None),
    ]
    for i, (fill, stroke, dash, text, level) in enumerate(entries):
        row, col = divmod(i, 3)
        x = 8 + col * col_w
        y = y0 + 16 + row * 20
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        items.append(
            f'<ellipse cx="{x + 9}" cy="{y}" rx="9" ry="6" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1.3"{dash_attr}/>'
        )
        text_x = x + 23
        if level is not None:
            items.append(
                _icon(level, x + 22, y - ICON / 2, RISK_COLORS[RiskLevel(level)]["strong"])
            )
            text_x = x + 36
        items.append(
            f'<text x="{text_x}" y="{y + LEGEND_FONT * 0.36:.1f}" font-size="{LEGEND_FONT:g}" '
            f'fill="{INK}">{escape(text)}</text>'
        )
    # Finding-marker key in the third row.
    x = 8
    y = y0 + 16 + 2 * 20
    items.append(
        f'<circle cx="{x + 9}" cy="{y}" r="7.4" fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>'
        f'<text x="{x + 9}" y="{y + 4.1}" text-anchor="middle" font-size="{LEGEND_FONT:g}" '
        f'font-weight="700" fill="{INK}">1</text>'
        f'<text x="{x + 23}" y="{y + LEGEND_FONT * 0.36:.1f}" font-size="{LEGEND_FONT:g}" '
        f'fill="{INK}">Finding number (matches the report)</text>'
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
    INCONCLUSIVE grey hatched) and carry the level icon (1/2/3 bars, "?"), so the map also reads
    in black and white; reference-only sites have a blue dashed outline; sites not in the result
    are light grey outlines. Flagged findings carry numbered markers that match the finding
    numbers in the report. ``result=None`` draws an empty map (all sites not examined).
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
        + f'<g class="map" transform="translate(0 {_TOP + 2:g})">'
        + _mouth_outline()
        + sites_svg
        + markers
        + "</g>"
        + legend
        + "</svg>"
    )
