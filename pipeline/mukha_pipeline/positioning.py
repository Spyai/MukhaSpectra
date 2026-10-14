"""Positioning guard: MukhaSpectra is a screening and triage aid, never a diagnostic device.

`find_positioning_violations` is used by tests to scan UI strings, report HTML, API output and
config text. The only allowed uses of "diagnosis" are the mandated disclaimers.
"""

from __future__ import annotations

import re

# Phrases that must appear verbatim where they are used; they are removed before scanning.
ALLOWED_PHRASES: tuple[str, ...] = (
    "not a diagnosis",
    "not a diagnostic device",
    "never a diagnostic device",
    "final diagnosis requires",
    "diagnosis requires a clinician",
)

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("diagnosis", re.compile(r"\bdiagnos[a-z]*", re.IGNORECASE)),
    ("malignant", re.compile(r"\bmalignan[a-z]*", re.IGNORECASE)),
    (
        "cancer detected",
        re.compile(
            r"\bcancer\s+(?:was\s+|is\s+)?(?:detected|found|present|confirmed)", re.IGNORECASE
        ),
    ),
    ("no cancer", re.compile(r"\bno\s+(?:sign\s+of\s+)?cancer\b", re.IGNORECASE)),
    ("cancer-free", re.compile(r"\bcancer[\s-]free\b", re.IGNORECASE)),
)


def find_positioning_violations(text: str) -> list[str]:
    """Return a list of human-readable violations found in ``text`` (empty if clean)."""
    scrubbed = text
    for phrase in ALLOWED_PHRASES:
        scrubbed = re.sub(re.escape(phrase), " ", scrubbed, flags=re.IGNORECASE)
    violations: list[str] = []
    for name, pattern in FORBIDDEN_PATTERNS:
        for m in pattern.finditer(scrubbed):
            start = max(0, m.start() - 40)
            context = scrubbed[start : m.end() + 40].replace("\n", " ")
            violations.append(f"{name!r}: …{context}…")
    return violations
