"""Whether a tobacco or areca-nut habit is current, former or absent.

A questionnaire can record the details of a habit (types, years, amount per day) and still leave
the "uses" box empty. Details only make sense for someone who uses the product, so recorded
details count as current use unless the habit is marked as former. Without this, a gutka user of
20 years whose box was missed would lose the habit's points, its mention in the explanation and
the cessation advice.
"""

from __future__ import annotations

from mukha_pipeline.schema.models import ArecaNutUse, SmokelessTobaccoUse, SmokingUse

Habit = SmokelessTobaccoUse | SmokingUse | ArecaNutUse


def has_details(habit: Habit) -> bool:
    """Types, years > 0 or amount per day > 0 were recorded."""
    types = getattr(habit, "types", [])
    return bool(types) or (habit.years or 0) > 0 or (habit.per_day or 0) > 0


def current_use(habit: Habit) -> bool:
    """Ticked as current use, or details recorded without being marked as former use."""
    return habit.uses or (not habit.former and has_details(habit))


def former_use(habit: Habit) -> bool:
    """Former use only (current use takes precedence)."""
    return habit.former and not current_use(habit)
