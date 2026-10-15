"""The console's English strings stay in step with the pipeline and respect positioning."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.qc import ISSUE_CATALOGUE
from mukha_pipeline.schema.models import ConfounderType, RiskLevel

EN = Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n" / "en.json"


def _strings(node: object, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(value, f"{path}.{key}" if path else key)
    elif isinstance(node, str):
        yield path, node


def _en() -> dict:
    return json.loads(EN.read_text(encoding="utf-8"))


def test_every_qc_issue_has_a_console_prompt():
    missing = sorted(set(ISSUE_CATALOGUE) - set(_en()["qc"]))
    assert missing == [], f"add these QC codes to frontend/src/i18n/en.json: {missing}"


def test_console_strings_respect_positioning():
    bad = [
        (path, v)
        for path, text in _strings(_en())
        if (v := find_positioning_violations(text))
    ]
    assert bad == []


def test_console_never_claims_performance():
    words = ("accuracy", "sensitivity", "specificity", "validated model")
    hits = [p for p, t in _strings(_en()) if any(w in t.lower() for w in words)]
    assert hits == []


def test_reference_tissue_is_never_called_healthy_in_results():
    results = _en().get("results", {})
    hits = [p for p, t in _strings(results) if "healthy" in t.lower()]
    assert hits == [], hits


def test_enums_the_console_renders_are_covered():
    text = EN.read_text(encoding="utf-8")
    for level in RiskLevel:
        assert level.value.lower() in text.lower()
    for c in ConfounderType:
        assert c.value in text
