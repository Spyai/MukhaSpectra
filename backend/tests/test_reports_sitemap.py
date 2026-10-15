"""Oral site map SVG: parses, colours match risk levels, roles are distinguishable."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from report_fixtures import cached_case

from app.reports import render_site_map_svg
from app.reports.labels import RISK_COLORS
from app.reports.sitemap import (
    HATCH_FILL,
    HATCH_ID,
    NOT_EXAMINED_STROKE,
    REFERENCE_STROKE,
    site_fill,
)
from mukha_pipeline.schema.models import RiskLevel, SiteRole
from mukha_pipeline.sites import SITES, OralSite

NS = {"svg": "http://www.w3.org/2000/svg"}


def _parse(svg: str) -> ET.Element:
    root = ET.fromstring(svg)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    return root


def _site_groups(root: ET.Element) -> dict[str, ET.Element]:
    return {g.get("data-site"): g for g in root.iterfind(".//svg:g[@data-site]", NS)}


def _ellipse(group: ET.Element) -> ET.Element:
    el = group.find("svg:ellipse", NS)
    assert el is not None
    return el


def test_empty_map_parses_and_shows_every_site_not_examined():
    root = _parse(render_site_map_svg())
    groups = _site_groups(root)
    assert set(groups) == {s.value for s in OralSite}
    for g in groups.values():
        assert g.get("data-state") == "not_examined"
        assert _ellipse(g).get("stroke") == NOT_EXAMINED_STROKE
    texts = " ".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
    assert "Patient's right" in texts and "Patient's left" in texts
    assert "viewed facing the patient" in texts


def test_geometry_comes_from_shared_site_catalogue():
    groups = _site_groups(_parse(render_site_map_svg()))
    for site, info in SITES.items():
        e = _ellipse(groups[site.value])
        assert float(e.get("cx")) == info.shape.cx
        assert float(e.get("cy")) == info.shape.cy
        assert float(e.get("rx")) == info.shape.rx
        assert float(e.get("ry")) == info.shape.ry


@pytest.mark.parametrize("name", ["high", "low", "inconclusive"])
def test_colours_match_site_levels(name: str):
    result = cached_case(name).result
    groups = _site_groups(_parse(render_site_map_svg(result)))
    examined = {s.site.value: s for s in result.sites}
    for site_value, g in groups.items():
        e = _ellipse(g)
        sr = examined.get(site_value)
        if sr is None:
            assert g.get("data-state") == "not_examined"
            continue
        level = RiskLevel(sr.site_risk_level)
        assert g.get("data-risk") == level.value
        if sr.role == SiteRole.REFERENCE_ONLY:
            assert g.get("data-state") == "reference_only"
            assert e.get("stroke") == REFERENCE_STROKE
            assert e.get("stroke-dasharray")
        else:
            assert g.get("data-state") == "examined"
            assert e.get("fill") == site_fill(level)
            assert e.get("stroke") == RISK_COLORS[level]["strong"]


def test_every_level_has_its_own_colour():
    case = cached_case("high")
    levels = [RiskLevel.HIGH, RiskLevel.MODERATE, RiskLevel.LOW, RiskLevel.INCONCLUSIVE]
    sites = [
        s.model_copy(update={"site_risk_level": lv, "role": SiteRole.EXAMINED})
        for s, lv in zip(case.result.sites[:4], levels, strict=True)
    ]
    result = case.result.model_copy(update={"sites": sites, "lesions": []})
    root = _parse(render_site_map_svg(result))
    groups = _site_groups(root)
    fills = {}
    for s, lv in zip(sites, levels, strict=True):
        fills[lv] = _ellipse(groups[s.site.value]).get("fill")
    assert fills[RiskLevel.HIGH] == RISK_COLORS[RiskLevel.HIGH]["fill"]
    assert fills[RiskLevel.MODERATE] == RISK_COLORS[RiskLevel.MODERATE]["fill"]
    assert fills[RiskLevel.LOW] == RISK_COLORS[RiskLevel.LOW]["fill"]
    assert fills[RiskLevel.INCONCLUSIVE] == HATCH_FILL
    assert len(set(fills.values())) == 4
    # The hatch pattern used for INCONCLUSIVE is defined in the document.
    assert root.find(f".//svg:pattern[@id='{HATCH_ID}']", NS) is not None


def test_lesion_markers_are_numbered_by_severity():
    result = cached_case("high").result
    root = _parse(render_site_map_svg(result))
    markers = root.findall(".//svg:g[@class='lesion-marker']", NS)
    assert [m.get("data-number") for m in markers] == ["1", "2"]
    # Finding 1 is the HIGH lesion even though it is listed second in the result.
    assert markers[0].get("data-lesion") == "les-001"
    groups = _site_groups(root)
    assert groups[OralSite.BUCCAL_MUCOSA_LEFT.value].get("data-lesions") == "1"


def test_width_and_legend_options():
    with_legend = _parse(render_site_map_svg(None, width=600, show_legend=True))
    no_legend = _parse(render_site_map_svg(None, width=300, show_legend=False))
    assert with_legend.get("width") == "600"
    assert no_legend.get("width") == "300"
    assert with_legend.find(".//svg:g[@class='legend']", NS) is not None
    assert no_legend.find(".//svg:g[@class='legend']", NS) is None
    # Aspect ratio is preserved: height follows the viewBox.
    vb = [float(x) for x in no_legend.get("viewBox").split()]
    assert float(no_legend.get("height")) == pytest.approx(300 * vb[3] / vb[2], abs=0.1)


def test_titles_for_accessibility():
    svg = render_site_map_svg(cached_case("high").result)
    root = _parse(svg)
    assert root.get("role") == "img"
    assert "Left buccal mucosa: examined, high risk; 1 flagged finding" in svg
    assert "Right buccal mucosa: reference only" in svg


# --------------------------------------------------------------------------------------------
# Print legibility, unambiguous labels, risk not by colour alone (clinician review P0/P1)
# --------------------------------------------------------------------------------------------

REPORT_MAP_MM = 74.0  # width of the map column in report.css (.cols-main-side)
MIN_TEXT_MM = 2.1  # ~6 pt on paper


def _font_sizes(root: ET.Element) -> list[float]:
    sizes = []
    for el in root.iter():
        if el.get("font-size"):
            sizes.append(float(el.get("font-size")))
    return sizes


@pytest.mark.parametrize("name", ["high", "low", "inconclusive", "stress"])
@pytest.mark.parametrize("legend", [False, True])
def test_every_text_is_legible_in_the_report_column(name: str, legend: bool):
    root = _parse(render_site_map_svg(cached_case(name).result, show_legend=legend))
    vb_width = float(root.get("viewBox").split()[2])
    texts = list(root.iter("{http://www.w3.org/2000/svg}text"))
    assert texts
    # Every text element inherits a size from itself or its group; check all declared sizes.
    smallest = min(_font_sizes(root))
    assert smallest * REPORT_MAP_MM / vb_width >= MIN_TEXT_MM, smallest


def test_report_column_width_matches_the_legibility_assumption():
    from app.reports.render import STATIC_DIR

    css = (STATIC_DIR / "report.css").read_text(encoding="utf-8")
    assert f"grid-template-columns: 1fr {REPORT_MAP_MM:g}mm" in css


def test_upper_and_lower_are_spelled_out():
    """ "L" on the map only ever means the patient's left (never "lower")."""
    root = _parse(render_site_map_svg())
    texts = {site: (g.find("svg:text", NS).text or "") for site, g in _site_groups(root).items()}
    assert texts[OralSite.UPPER_LIP_MUCOSA.value] == "Upper lip"
    assert texts[OralSite.LOWER_LIP_MUCOSA.value] == "Lower lip"
    assert texts[OralSite.UPPER_GINGIVA.value] == "Upper gum"
    assert texts[OralSite.LOWER_GINGIVA.value] == "Lower gum"
    for site, text in texts.items():
        if text.startswith(("L ", "R ")):
            assert ("left" if text[0] == "L" else "right") in site, (site, text)
        assert not text.startswith("U "), (site, text)
    # Titles (hover / screen readers) keep the full catalogue names.
    title = _site_groups(root)[OralSite.LOWER_GINGIVA.value].find("svg:title", NS)
    assert title is not None and title.text.startswith("Lower gingiva")


def test_examined_sites_carry_a_level_icon():
    result = cached_case("high").result
    groups = _site_groups(_parse(render_site_map_svg(result)))
    roles = {s.site.value: s for s in result.sites}
    for site_value, g in groups.items():
        icon = g.find("svg:g[@class='level-icon']", NS)
        sr = roles.get(site_value)
        if sr is not None and sr.role == SiteRole.EXAMINED:
            assert icon is not None, site_value
            assert icon.get("data-level") == sr.site_risk_level.value
            if sr.site_risk_level == RiskLevel.INCONCLUSIVE:
                assert icon.find("svg:circle", NS) is not None  # "?" icon
            else:
                bars = [r for r in icon.findall("svg:rect", NS) if r.get("opacity") == "1"]
                expected = {"LOW": 1, "MODERATE": 2, "HIGH": 3}[sr.site_risk_level.value]
                assert len(bars) == expected
        else:
            assert icon is None, site_value


def test_markers_on_narrow_sites_sit_beside_the_label():
    from app.reports.sitemap import MARKER_R, marker_position

    for site in (OralSite.LATERAL_TONGUE_LEFT, OralSite.LATERAL_TONGUE_RIGHT):
        e = SITES[site].shape
        x, _ = marker_position(site, 0)
        # The whole circle is outside the shape, on the outer side (away from the midline).
        assert abs(x - e.cx) >= e.rx + MARKER_R
        assert (x - e.cx) * (e.cx - 200) > 0
    e = SITES[OralSite.RETROMOLAR_LEFT].shape
    _, y = marker_position(OralSite.RETROMOLAR_LEFT, 0)
    assert y + MARKER_R <= e.cy - e.ry  # above the small retromolar shape


def test_svg_legend_has_three_columns_with_icons():
    root = _parse(render_site_map_svg(None, show_legend=True))
    legend = root.find(".//svg:g[@class='legend']", NS)
    assert legend is not None
    xs = {float(e.get("cx")) for e in legend.findall("svg:ellipse", NS)}
    assert len(xs) == 3
    assert len(legend.findall("svg:g[@class='level-icon']", NS)) == 4
    words = " ".join(t.text or "" for t in legend.iter("{http://www.w3.org/2000/svg}text"))
    for word in ("High risk", "Moderate risk", "Low risk", "Inconclusive", "Reference only",
                 "Not examined", "Finding number"):  # fmt: skip
        assert word in words
