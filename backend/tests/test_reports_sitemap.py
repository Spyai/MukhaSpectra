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
