"""Spectral plot, score gauge and risk icons."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest
from report_fixtures import cached_case

from app.reports import spectral_plot_svg
from app.reports.charts import risk_icon_svg, score_gauge_svg, spectral_y_label
from mukha_pipeline.schema.models import RiskLevel


def _readings():
    readings = {r.reading_id: r for r in cached_case("high").result.spectral_readings}
    return readings["s1"], readings["s2"]


def _svg_text(svg: str) -> str:
    root = ET.fromstring(svg.encode("utf-8"))
    assert root.tag.endswith("svg")
    return " ".join(t for t in root.itertext())


def test_spectral_plot_with_reference():
    lesion, reference = _readings()
    svg = spectral_plot_svg(lesion, reference)
    text = _svg_text(svg)
    assert "Reflectance" in text
    assert "Wavelength (nm)" in text
    for nm in (415, 445, 480, 515, 555, 590, 630, 680):
        assert str(nm) in text
    assert "Lesion — Left buccal mucosa (n=5)" in text
    assert "Reference — Right buccal mucosa (n=5)" in text
    assert "Hb band" in text


def test_spectral_plot_without_reference():
    lesion, _ = _readings()
    text = _svg_text(spectral_plot_svg(lesion, None))
    assert "Lesion — Left buccal mucosa" in text
    assert "Reference" not in text


def test_relative_signal_label_and_missing_channels():
    lesion, _ = _readings()
    values = dict(lesion.values)
    values["F3_480"] = math.nan
    del values["F8_680"]
    relative = lesion.model_copy(update={"signal_type": "relative_signal", "values": values})
    text = _svg_text(spectral_plot_svg(relative, None))
    assert "Relative signal" in text
    assert spectral_y_label("relative_signal") == "Relative signal"
    assert spectral_y_label("reflectance") == "Reflectance"


def test_spectral_plot_with_no_usable_values():
    lesion, _ = _readings()
    empty = lesion.model_copy(update={"values": {}, "values_std": {}})
    assert "No usable channel values" in _svg_text(spectral_plot_svg(empty, None))


def test_spectral_plot_is_deterministic():
    lesion, reference = _readings()
    assert spectral_plot_svg(lesion, reference) == spectral_plot_svg(lesion, reference)


@pytest.mark.parametrize("level", list(RiskLevel))
def test_risk_icons_parse_and_differ_by_shape(level: RiskLevel):
    svg = risk_icon_svg(level, "#000000")
    root = ET.fromstring(svg)
    shapes = {child.tag.split("}")[-1] for child in root}
    if level == RiskLevel.INCONCLUSIVE:
        assert "circle" in shapes and "path" in shapes
    else:
        opaque = [c for c in root if c.get("opacity") == "1"]
        assert len(opaque) == {"LOW": 1, "MODERATE": 2, "HIGH": 3}[level.value]


def test_score_gauge_marks_thresholds_and_score():
    svg = score_gauge_svg(12.8, 3.0, 6.0, RiskLevel.HIGH)
    text = _svg_text(svg)
    assert "Score 12.8" in text
    assert "Moderate ≥ 3.0" in text and "High ≥ 6.0" in text
    no_score = _svg_text(score_gauge_svg(None, 3.0, 6.0, RiskLevel.INCONCLUSIVE))
    assert "Score" not in no_score


def _font_sizes(svg: str) -> list[float]:
    import re

    sizes = [float(x) for x in re.findall(r'font-size[:=]\s*"?(\d+(?:\.\d+)?)', svg)]
    assert sizes
    return sizes


def test_chart_text_is_legible_when_printed():
    # Spectral plot: 4.3 in (109 mm) drawn into a 90 mm column; text must stay >= ~6 pt.
    lesion, reference = _readings()
    spectral = _font_sizes(spectral_plot_svg(lesion, reference))
    assert min(spectral) * 90 / 109 >= 5.9
    # Gauge: 290 px drawn at 290 CSS px (0.75 pt per px).
    gauge = _font_sizes(score_gauge_svg(12.8, 3.0, 6.0, RiskLevel.HIGH, width=290))
    assert min(gauge) >= 9


def test_gauge_threshold_labels_do_not_overlap():
    import re

    svg = score_gauge_svg(12.8, 3.0, 6.0, RiskLevel.HIGH, width=290)
    labels = re.findall(r'<text x="([\d.]+)"[^>]*>((?:Moderate|High) ≥ [\d.]+)</text>', svg)
    assert len(labels) == 2
    (x1, t1), (x2, t2) = [(float(x), t) for x, t in labels]
    half = (len(t1) + len(t2)) * 9 * 0.28
    assert x2 - x1 >= half
