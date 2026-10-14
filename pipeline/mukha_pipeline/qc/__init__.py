"""Capture quality control: white light, autofluorescence, dark frame, registration,
segmentation and spectral probe readings.

QC exists to stop false reassurance: a capture that cannot show a lesion must never be analysed
as if it were good. Every function returns a ``QCResult`` whose issues carry a stable code, a
plain-language retake prompt for the health worker and an i18n key (``qc.<code>``).
"""

from mukha_pipeline.qc.image_qc import (
    qc_autofluorescence,
    qc_dark_frame,
    qc_registration,
    qc_segmentation,
    qc_white_light,
)
from mukha_pipeline.qc.messages import (
    ISSUE_CATALOGUE,
    IssueSpec,
    issue_spec,
    make_issue,
    missing_dark_frame_issue,
)
from mukha_pipeline.qc.result import build_result, combine_qc, score_issues
from mukha_pipeline.qc.spectral_qc import qc_spectral

__all__ = [
    "ISSUE_CATALOGUE",
    "IssueSpec",
    "build_result",
    "combine_qc",
    "issue_spec",
    "make_issue",
    "missing_dark_frame_issue",
    "qc_autofluorescence",
    "qc_dark_frame",
    "qc_registration",
    "qc_segmentation",
    "qc_spectral",
    "qc_white_light",
    "score_issues",
]
