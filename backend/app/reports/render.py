"""Render a ScreeningResult to the clinician report: Jinja2 HTML, then WeasyPrint PDF.

The HTML is fully self-contained (inline CSS, images as data URIs), so the API can serve it as
``report.html`` and the PDF needs no network or file access beyond installed fonts.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup

from app.reports.context import build_context
from app.reports.images import svg_data_uri
from app.reports.types import ArtifactReader, PatientIdentity, ReportOptions
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import ScreeningResult

log = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = REPORTS_DIR / "templates"
STATIC_DIR = REPORTS_DIR / "static"
CLINICIAN_TEMPLATE = "clinician_report.html.j2"


def css_string(value: object) -> Markup:
    """Quote ``value`` as a CSS string literal, safe inside a <style> element."""
    text = str(value)
    out = []
    for ch in text:
        if ch in ('"', "\\", "<", ">", "&") or ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\{ord(ch):x} ")
        else:
            out.append(ch)
    return Markup('"' + "".join(out) + '"')


@lru_cache(maxsize=1)
def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    env.filters["css_str"] = css_string
    return env


@lru_cache(maxsize=1)
def _static_css() -> Markup:
    css = (STATIC_DIR / "report.css").read_text(encoding="utf-8")
    # The stylesheet is our own file; guard against an accidental closing tag anyway.
    return Markup(css.replace("</", "<\\/"))


@lru_cache(maxsize=1)
def _logo_uri() -> str:
    return svg_data_uri((STATIC_DIR / "logo.svg").read_text(encoding="utf-8"))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("style", "script", "title"):
            self._skip += 1
        for name, value in attrs:
            if name == "alt" and value:
                self.parts.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script", "title") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def visible_text(html: str) -> str:
    """User-visible text of a report (body text + image alt text), whitespace-collapsed."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def render_clinician_report_html(
    result: ScreeningResult,
    artifacts: ArtifactReader,
    *,
    identity: PatientIdentity | None = None,
    options: ReportOptions,
) -> str:
    """Self-contained HTML of the clinician / referral report (2–4 A4 pages when printed)."""
    context = build_context(result, artifacts, identity=identity, options=options)
    context["css"] = _static_css()
    context["logo_uri"] = _logo_uri()
    html = _environment().get_template(CLINICIAN_TEMPLATE).render(**context)

    violations = find_positioning_violations(visible_text(html))
    if violations:
        # Wording comes partly from pipeline outputs (labels, explanations, free text). Never
        # block a referral over wording, but make the problem loud in the logs.
        log.warning("positioning violations in report %s: %s", options.report_id, violations)
    return html


def render_clinician_report_pdf(
    result: ScreeningResult,
    artifacts: ArtifactReader,
    *,
    identity: PatientIdentity | None = None,
    options: ReportOptions,
) -> bytes:
    """Clinician / referral report as PDF bytes (A4, self-contained)."""
    from weasyprint import HTML

    html = render_clinician_report_html(result, artifacts, identity=identity, options=options)
    document = HTML(string=html, base_url=str(STATIC_DIR))
    pdf = document.write_pdf()
    assert pdf is not None
    return pdf
