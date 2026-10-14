"""QC issue catalogue: required codes, prompt style, i18n keys and positioning."""

from pathlib import Path

import pytest

from mukha_pipeline.config import CONFIG_DIR, QC_CONFIG
from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.qc import ISSUE_CATALOGUE, issue_spec, make_issue, missing_dark_frame_issue
from mukha_pipeline.qc.messages import MAX_PROMPT_WORDS
from mukha_pipeline.schema.models import QCSeverity

REQUIRED_CODES = {
    "blur",
    "too_dark",
    "too_bright",
    "clipping_dark",
    "clipping_bright",
    "glare",
    "glare_minor",
    "af_signal_low",
    "af_filter_leak",
    "af_overexposed",
    "af_blur",
    "ambient_light",
    "ambient_light_minor",
    "no_dark_frame",
    "misaligned",
    "alignment_marginal",
    "insufficient_mucosa",
    "spectral_no_readings",
    "spectral_few_readings",
    "spectral_saturated",
    "spectral_ambient_leak",
    "probe_no_contact",
    "spectral_poor_repeatability",
    "spectral_signal_low",
}


def test_catalogue_has_every_required_code():
    assert set(ISSUE_CATALOGUE) >= REQUIRED_CODES


@pytest.mark.parametrize("code", sorted(ISSUE_CATALOGUE))
def test_prompt_is_short_plain_and_keyed(code):
    spec = ISSUE_CATALOGUE[code]
    assert spec.code == code
    assert spec.message_key == f"qc.{code}"
    assert 3 <= len(spec.message.split()) <= MAX_PROMPT_WORDS, spec.message
    assert spec.message[0].isupper() and spec.message.endswith(".")
    assert "—" in spec.message  # "what went wrong — what to do"
    assert isinstance(spec.severity, QCSeverity)


def test_prompts_are_unique():
    messages = [s.message for s in ISSUE_CATALOGUE.values()]
    assert len(messages) == len(set(messages))


def test_no_prompt_violates_positioning():
    text = "\n".join(s.message for s in ISSUE_CATALOGUE.values())
    assert find_positioning_violations(text) == []


def test_qc_config_text_respects_positioning():
    text = (Path(CONFIG_DIR) / QC_CONFIG).read_text(encoding="utf-8")
    assert find_positioning_violations(text) == []


def test_severity_defaults_match_clinical_intent():
    errors = {c for c, s in ISSUE_CATALOGUE.items() if s.severity == QCSeverity.ERROR}
    warnings = {c for c, s in ISSUE_CATALOGUE.items() if s.severity == QCSeverity.WARNING}
    assert {"blur", "too_dark", "af_signal_low", "af_filter_leak", "misaligned"} <= errors
    assert {"probe_no_contact", "spectral_saturated", "ambient_light"} <= errors
    assert {"glare_minor", "no_dark_frame", "alignment_marginal", "clipping_dark"} <= warnings
    assert {"spectral_few_readings", "spectral_poor_repeatability"} <= warnings


def test_make_issue_fills_from_catalogue():
    issue = make_issue("blur", metric="wl.blur_laplacian_var", value=1.5, threshold=5.0)
    assert issue.code == "blur"
    assert issue.severity == QCSeverity.ERROR
    assert issue.message == ISSUE_CATALOGUE["blur"].message
    assert issue.message_key == "qc.blur"
    assert (issue.metric, issue.value, issue.threshold) == ("wl.blur_laplacian_var", 1.5, 5.0)


def test_make_issue_severity_override_and_non_finite_values():
    issue = make_issue(
        "spectral_signal_low",
        metric="spectral.min_net_signal",
        value=float("nan"),
        threshold=float("inf"),
        severity=QCSeverity.ERROR,
    )
    assert issue.severity == QCSeverity.ERROR
    assert issue.value is None and issue.threshold is None
    assert make_issue("glare").metric is None


def test_unknown_code_fails_loudly():
    with pytest.raises(KeyError, match="unknown QC issue code"):
        make_issue("definitely_not_a_code")
    with pytest.raises(KeyError):
        issue_spec("nope")


def test_missing_dark_frame_is_a_warning_not_an_error():
    issue = missing_dark_frame_issue()
    assert issue.code == "no_dark_frame"
    assert issue.severity == QCSeverity.WARNING
    assert issue.message_key == "qc.no_dark_frame"
