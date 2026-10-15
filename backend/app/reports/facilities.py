"""Referral facility list (configurable per deployment)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.reports.types import Facility

DEFAULT_FACILITIES_PATH = Path(__file__).resolve().parents[1] / "config" / "facilities.yaml"

_FIELDS = ("name", "address", "phone", "hours", "notes")
_LIST_FIELDS = ("levels", "capabilities")
_LEVELS = ("LOW", "MODERATE", "HIGH", "INCONCLUSIVE")


def _optional_str(entry: dict[str, Any], key: str, index: int) -> str | None:
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str | int | float):
        raise ValueError(f"facilities[{index}].{key} must be text")
    text = str(value).strip()
    return text or None


def _str_list(entry: dict[str, Any], key: str, index: int) -> list[str]:
    value = entry.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"facilities[{index}].{key} must be a list of text")
    return [v.strip() for v in value]


def load_facilities(path: Path | None = None) -> list[Facility]:
    """Load referral facilities from YAML.

    The file holds ``facilities: [{name, address, phone?, hours?, notes?, example?, levels?,
    capabilities?}]``, first-choice facility first. ``levels`` (risk levels the facility is
    listed for; empty = all) lets a deployment send HIGH results to a biopsy centre and LOW
    results to a local clinic. The default file ships with clearly marked example entries
    (``example: true``) that the report labels "Example facility — configure facilities.yaml"
    and drops from reports of non-synthetic results. A missing default file yields an empty
    list; a missing explicit ``path`` raises ``FileNotFoundError``.
    """
    target = Path(path) if path is not None else DEFAULT_FACILITIES_PATH
    if path is None and not target.exists():
        return []
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("facilities") or []
    else:
        raise ValueError("facilities file must be a mapping with a 'facilities' list")
    if not isinstance(entries, list):
        raise ValueError("'facilities' must be a list")

    facilities: list[Facility] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"facilities[{index}] must be a mapping")
        unknown = set(entry) - {*_FIELDS, *_LIST_FIELDS, "example"}
        if unknown:
            raise ValueError(f"facilities[{index}] has unknown keys: {sorted(unknown)}")
        name = _optional_str(entry, "name", index)
        address = _optional_str(entry, "address", index)
        if not name or not address:
            raise ValueError(f"facilities[{index}] needs a name and an address")
        levels = [lv.upper() for lv in _str_list(entry, "levels", index)]
        if any(lv not in _LEVELS for lv in levels):
            raise ValueError(f"facilities[{index}].levels must be among {list(_LEVELS)}")
        facilities.append(
            Facility(
                name=name,
                address=address,
                phone=_optional_str(entry, "phone", index),
                hours=_optional_str(entry, "hours", index),
                notes=_optional_str(entry, "notes", index),
                example=bool(entry.get("example", False)),
                levels=levels,
                capabilities=_str_list(entry, "capabilities", index),
            )
        )
    return facilities
