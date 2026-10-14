"""Preview the clinician report from a fixture or a saved result.

Examples::

    python -m app.reports.preview --fixture high --out /tmp/report.pdf
    python -m app.reports.preview --fixture inconclusive --out /tmp/report.html
    python -m app.reports.preview --result result.json --artifacts results/v1/ --out /tmp/r.pdf
    python -m app.reports.preview --fixture high --sitemap /tmp/sitemap.svg
    python -m app.reports.preview --fixture stress --out /tmp/stress.pdf   # 5 findings

Fixtures are SYNTHETIC and live in ``backend/tests/report_fixtures`` (development checkouts
only). ``--out`` ending in ``.html`` writes HTML; anything else writes PDF. ``--html`` writes the
HTML alongside the PDF.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from app.reports.facilities import load_facilities
from app.reports.render import render_clinician_report_html, render_clinician_report_pdf
from app.reports.sitemap import render_site_map_svg
from app.reports.types import ArtifactReader, PatientIdentity, ReportOptions
from mukha_pipeline.schema.models import ScreeningResult

TESTS_DIR = Path(__file__).resolve().parents[2] / "tests"


def _fixture(name: str) -> tuple[ScreeningResult, ArtifactReader, PatientIdentity | None]:
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    try:
        from report_fixtures import build_case
    except ImportError as exc:  # pragma: no cover - only in installs without tests
        raise SystemExit(f"report fixtures not found under {TESTS_DIR}: {exc}") from exc
    case = build_case(name)
    identity = PatientIdentity(name="Demo Patient (synthetic)", external_id="DEMO-0001")
    return case.result, case.reader, identity


def _from_files(
    result_path: Path, artifacts_dir: Path | None
) -> tuple[ScreeningResult, ArtifactReader]:
    result = ScreeningResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    root = (artifacts_dir or result_path.parent).resolve()

    def reader(key: str) -> bytes | None:
        path = (root / key).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return None
        return path.read_bytes()

    return result, reader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixture", choices=["high", "low", "inconclusive", "stress"])
    src.add_argument("--result", type=Path, help="ScreeningResult JSON file")
    parser.add_argument("--artifacts", type=Path, help="artifact root (default: result's dir)")
    parser.add_argument("--out", type=Path, required=True, help=".pdf or .html output path")
    parser.add_argument("--html", action="store_true", help="also write the HTML next to a PDF")
    parser.add_argument("--sitemap", type=Path, help="also write the site map SVG here")
    parser.add_argument("--record-url", default=None, help="URL for the QR code")
    parser.add_argument("--report-id", default=None)
    args = parser.parse_args(argv)

    identity: PatientIdentity | None = None
    if args.fixture:
        result, reader, identity = _fixture(args.fixture)
    else:
        result, reader = _from_files(args.result, args.artifacts)

    record_url = args.record_url
    if record_url is None and args.fixture and args.fixture != "inconclusive":
        # The inconclusive fixture demonstrates a report without a QR code (see README).
        record_url = f"https://mukhaspectra.example.org/r/{result.screening_id}"
    options = ReportOptions(
        report_id=args.report_id or f"PREVIEW-{result.screening_id}",
        generated_at=datetime.now(UTC),
        record_url=record_url,
        facilities=load_facilities(),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.out.suffix.lower() in (".html", ".htm"):
        html = render_clinician_report_html(result, reader, identity=identity, options=options)
        args.out.write_text(html, encoding="utf-8")
    else:
        pdf = render_clinician_report_pdf(result, reader, identity=identity, options=options)
        args.out.write_bytes(pdf)
        if args.html:
            html = render_clinician_report_html(result, reader, identity=identity, options=options)
            args.out.with_suffix(".html").write_text(html, encoding="utf-8")
    if args.sitemap:
        args.sitemap.write_text(render_site_map_svg(result), encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"wrote {args.out} in {elapsed:.2f} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
