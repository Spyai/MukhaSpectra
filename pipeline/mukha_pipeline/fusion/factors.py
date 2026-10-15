"""Contributing-factor ledger.

Every point added or removed is a ``ContributingFactor``. Multipliers from the rules are turned
into explicit point deltas against the running total, so a score is always exactly the sum of its
factors and a clinician can see what each adjustment did.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from mukha_pipeline.schema.models import ContributingFactor, FactorSource

POINT_DECIMALS = 4


def make_factor(
    code: str,
    label: str,
    source: FactorSource,
    points: float = 0.0,
    detail: str | None = None,
) -> ContributingFactor:
    """Build a factor with points rounded for display; scores are summed from these values."""
    rounded = round(points, POINT_DECIMALS) + 0.0  # + 0.0 turns -0.0 into 0.0
    return ContributingFactor(code=code, label=label, source=source, points=rounded, detail=detail)


def total_points(factors: Iterable[ContributingFactor]) -> float:
    return math.fsum(f.points for f in factors)


@dataclass
class Ledger:
    """Ordered list of factors with a running total."""

    factors: list[ContributingFactor] = field(default_factory=list)

    @property
    def total(self) -> float:
        return total_points(self.factors)

    def add(
        self,
        code: str,
        label: str,
        source: FactorSource,
        points: float = 0.0,
        detail: str | None = None,
    ) -> ContributingFactor:
        f = make_factor(code, label, source, points, detail)
        self.factors.append(f)
        return f

    def note(self, code: str, label: str, source: FactorSource, detail: str | None = None) -> None:
        """Zero-point informational factor (shown in reports, never changes the score)."""
        self.add(code, label, source, 0.0, detail)

    def scale(
        self,
        multiplier: float,
        code: str,
        label: str,
        source: FactorSource,
        *,
        detail: str | None = None,
    ) -> ContributingFactor | None:
        """Express ``running total * multiplier`` as a delta factor.

        No factor is recorded when the multiplier is 1 or the delta rounds to zero, so reports
        only list adjustments that changed something.
        """
        delta = self.total * (multiplier - 1.0)
        if multiplier == 1.0 or round(delta, POINT_DECIMALS) == 0.0:
            return None
        extra = f"x{multiplier:.2f}"
        return self.add(code, label, source, delta, f"{extra}; {detail}" if detail else extra)

    def extend(self, factors: Iterable[ContributingFactor]) -> None:
        self.factors.extend(factors)
