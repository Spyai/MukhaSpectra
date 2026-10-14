"""Rule-based fusion and triage (v1, baseline heuristic, not clinically validated).

Public API:
* ``run_triage(sites, spectral, patient, risk_factors, rules) -> TriageOutput``
* ``parse_triage_rules(data) -> TriageRules`` (raises ``TriageRulesError`` on a malformed file)
"""

from mukha_pipeline.fusion.rules import TriageRules, TriageRulesError, parse_triage_rules
from mukha_pipeline.fusion.triage import SEVERITY, TriageOutput, run_triage

__all__ = [
    "SEVERITY",
    "TriageOutput",
    "TriageRules",
    "TriageRulesError",
    "parse_triage_rules",
    "run_triage",
]
