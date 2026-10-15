"""Generate the ScreeningResult JSON Schema from the pydantic models.

python -m mukha_pipeline.schema.export            # rewrite the checked-in schema
python -m mukha_pipeline.schema.export --check    # exit 1 if it is out of date
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mukha_pipeline import SCHEMA_VERSION
from mukha_pipeline.schema.models import ScreeningResult

SCHEMA_PATH = Path(__file__).with_name("screening_result.v1.schema.json")
SCHEMA_ID = "https://schemas.mukhaspectra.in/screening_result/v1.json"


def build_schema() -> dict:
    schema = ScreeningResult.model_json_schema(mode="serialization")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "MukhaSpectra ScreeningResult",
        "x-schema-version": SCHEMA_VERSION,
        **{k: v for k, v in schema.items() if k != "title"},
    }


def render() -> str:
    return json.dumps(build_schema(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        current = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else ""
        if current != text:
            print(f"{SCHEMA_PATH} is out of date; run python -m mukha_pipeline.schema.export")
            return 1
        return 0
    SCHEMA_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
