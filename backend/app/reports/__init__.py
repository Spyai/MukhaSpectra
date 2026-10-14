"""MukhaSpectra report engine: ScreeningResult -> clinician / referral report (HTML, PDF).

Research prototype — not for clinical use. Reports are a screening aid, not a diagnosis.
See README.md in this package for the report specification and a preview CLI.
"""

from app.reports.charts import spectral_plot_svg
from app.reports.facilities import DEFAULT_FACILITIES_PATH, load_facilities
from app.reports.render import render_clinician_report_html, render_clinician_report_pdf
from app.reports.sitemap import render_site_map_svg
from app.reports.types import ArtifactReader, Facility, PatientIdentity, ReportOptions

__all__ = [
    "DEFAULT_FACILITIES_PATH",
    "ArtifactReader",
    "Facility",
    "PatientIdentity",
    "ReportOptions",
    "load_facilities",
    "render_clinician_report_html",
    "render_clinician_report_pdf",
    "render_site_map_svg",
    "spectral_plot_svg",
]
