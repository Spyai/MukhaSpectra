"""Turning issues + metrics into a QCResult, and combining per-capture results.

Score = 1 - sum of per-severity weights (``score_weights`` in qc.v1.yaml), clipped to [0, 1].
``passed`` is False as soon as one issue has severity ERROR: a capture that could hide a lesion
must never be analysed as if it were good (false reassurance is the worst failure mode).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from mukha_pipeline.schema.models import QCIssue, QCResult, QCSeverity

SEVERITY_RANK: Mapping[QCSeverity, int] = {
    QCSeverity.INFO: 0,
    QCSeverity.WARNING: 1,
    QCSeverity.ERROR: 2,
}


def finite_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Keep only finite numbers so the result stays JSON-safe (no NaN / inf)."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out[key] = v
    return out


def score_issues(issues: Iterable[QCIssue], cfg: Mapping) -> float:
    """1 - sum(score_weights[severity]); unknown severities (e.g. info) weigh 0."""
    weights = cfg.get("score_weights", {}) or {}
    penalty = sum(float(weights.get(str(i.severity), 0.0)) for i in issues)
    return float(min(1.0, max(0.0, 1.0 - penalty)))


def build_result(issues: list[QCIssue], metrics: Mapping[str, float], cfg: Mapping) -> QCResult:
    """Assemble a QCResult; ``cfg`` is the full qc config (only ``score_weights`` is read)."""
    return QCResult(
        passed=not any(i.severity == QCSeverity.ERROR for i in issues),
        score=score_issues(issues, cfg),
        issues=list(issues),
        metrics=finite_metrics(metrics),
    )


def combine_qc(results: Iterable[QCResult]) -> QCResult:
    """Merge several QC results (e.g. WL + AF + dark + registration for one site).

    passed = all passed; score = the worst score; issues are concatenated and de-duplicated by
    (code, message), keeping the most severe copy at the position of the first one; metrics are
    merged with later results overwriting earlier keys. An empty input is the neutral element
    (passed, score 1): callers must not treat "nothing captured" as a good capture.
    """
    results = list(results)
    issues: list[QCIssue] = []
    index: dict[tuple[str, str], int] = {}
    metrics: dict[str, float] = {}
    for r in results:
        for issue in r.issues:
            key = (issue.code, issue.message)
            if key not in index:
                index[key] = len(issues)
                issues.append(issue)
            elif SEVERITY_RANK[issue.severity] > SEVERITY_RANK[issues[index[key]].severity]:
                issues[index[key]] = issue
        metrics.update(r.metrics)
    passed = all(r.passed for r in results) and not any(
        i.severity == QCSeverity.ERROR for i in issues
    )
    score = min((r.score for r in results), default=1.0)
    return QCResult(passed=passed, score=score, issues=issues, metrics=finite_metrics(metrics))
