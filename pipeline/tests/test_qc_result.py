"""QCResult assembly: scoring, pass/fail, combining per-capture results."""

import math

import pytest

from mukha_pipeline.qc import build_result, combine_qc, make_issue, score_issues
from mukha_pipeline.qc.result import finite_metrics
from mukha_pipeline.schema.models import QCResult, QCSeverity


def test_no_issues_scores_one_and_passes(qc_cfg):
    r = build_result([], {"wl.blur_laplacian_var": 12.0}, qc_cfg)
    assert r.passed and r.score == 1.0
    assert r.metrics == {"wl.blur_laplacian_var": 12.0}


def test_score_uses_config_weights(qc_cfg):
    w = qc_cfg["score_weights"]
    warn = make_issue("glare_minor")
    err = make_issue("blur")
    assert score_issues([warn], qc_cfg) == pytest.approx(1 - w["warning"])
    assert score_issues([warn, err], qc_cfg) == pytest.approx(1 - w["warning"] - w["error"])
    info = make_issue("glare_minor", severity=QCSeverity.INFO)
    assert score_issues([info], qc_cfg) == 1.0


def test_score_is_clipped_to_unit_interval(qc_cfg):
    many = [make_issue("blur"), make_issue("too_dark"), make_issue("glare")]
    r = build_result(many, {}, qc_cfg)
    assert r.score == 0.0 and not r.passed
    assert score_issues([], {}) == 1.0  # no weights configured -> no penalty


def test_warnings_alone_do_not_fail(qc_cfg):
    r = build_result([make_issue("glare_minor"), make_issue("no_dark_frame")], {}, qc_cfg)
    assert r.passed and 0.0 < r.score < 1.0


def test_metrics_drop_non_finite_and_non_numeric():
    m = finite_metrics({"a": 1, "b": math.nan, "c": math.inf, "d": "x", "e": None, "f": 2.5})
    assert m == {"a": 1.0, "f": 2.5}


def _res(passed, score, issues=(), metrics=None):
    return QCResult(passed=passed, score=score, issues=list(issues), metrics=metrics or {})


def test_combine_all_pass_takes_min_score_and_merges_metrics():
    a = _res(True, 0.85, [make_issue("glare_minor")], {"wl.glare_fraction": 0.02, "k": 1.0})
    b = _res(True, 1.0, [], {"af.green_p90": 0.4, "k": 2.0})
    c = combine_qc([a, b])
    assert c.passed and c.score == 0.85
    assert c.metrics == {"wl.glare_fraction": 0.02, "af.green_p90": 0.4, "k": 2.0}
    assert [i.code for i in c.issues] == ["glare_minor"]


def test_combine_fails_if_any_fails_and_dedupes_keeping_most_severe():
    warn = make_issue("spectral_signal_low", value=10.0)
    err = make_issue("spectral_signal_low", value=5.0, severity=QCSeverity.ERROR)
    a = _res(True, 0.85, [warn, make_issue("glare_minor")])
    b = _res(False, 0.5, [err])
    c = combine_qc([a, b, a])
    assert not c.passed and c.score == 0.5
    assert [i.code for i in c.issues] == ["spectral_signal_low", "glare_minor"]
    assert c.issues[0].severity == QCSeverity.ERROR and c.issues[0].value == 5.0


def test_combine_never_passes_with_an_error_issue():
    # A malformed input result claiming passed=True with an error issue cannot hide the error.
    c = combine_qc([_res(True, 1.0, [make_issue("blur")])])
    assert not c.passed


def test_combine_accepts_generators_and_empty_input():
    c = combine_qc(r for r in [_res(True, 0.7)])
    assert c.passed and c.score == 0.7
    empty = combine_qc([])
    assert empty.passed and empty.score == 1.0 and empty.issues == []
