"""SYNTHETIC ScreeningResult fixtures + artifact images for report-engine tests and previews."""

from __future__ import annotations

from datetime import datetime

from report_fixtures.results import (
    BUILDERS,
    GENERATED_AT,
    ReportCase,
    build_case,
    build_high,
    build_inconclusive,
    build_low,
)

__all__ = [
    "BUILDERS",
    "GENERATED_AT",
    "ReportCase",
    "build_case",
    "build_high",
    "build_inconclusive",
    "build_low",
    "dict_reader",
    "default_options",
]


def dict_reader(artifacts: dict[str, bytes]):
    """ArtifactReader backed by a dict (missing key -> None)."""
    return artifacts.get


def default_options(case: ReportCase, *, record_url: str | None = "default",
                    generated_at: datetime | None = None):
    """ReportOptions for a fixture case with the example facilities from facilities.yaml."""
    from app.reports import ReportOptions, load_facilities

    if record_url == "default":
        record_url = f"https://mukhaspectra.example.org/r/{case.result.screening_id}"
    return ReportOptions(
        report_id=f"MS-{case.result.screening_id.removeprefix('scr-').upper()}-R1",
        generated_at=generated_at or GENERATED_AT,
        record_url=record_url,
        facilities=load_facilities(),
        report_version=1,
    )
