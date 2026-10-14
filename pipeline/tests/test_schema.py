"""The ScreeningResult contract: JSON Schema drift and positioning/honesty invariants."""

import json

from mukha_pipeline.schema import export
from mukha_pipeline.schema.models import (
    DISCLAIMER,
    PROTOTYPE_BANNER,
    ModelComponent,
    ModelInfo,
    ModelStatus,
    Notices,
    ScreeningResult,
)


def test_checked_in_schema_matches_models():
    assert (
        export.SCHEMA_PATH.read_text(encoding="utf-8") == export.render()
    ), "Run: python -m mukha_pipeline.schema.export"


def test_schema_is_versioned_and_valid_json_schema():
    import jsonschema

    schema = json.loads(export.SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["x-schema-version"] == "1.0.0"
    assert schema["properties"]["schema_version"]["const"] == "1.0.0"


def test_notices_carry_mandated_text():
    n = Notices()
    assert n.prototype_banner == PROTOTYPE_BANNER == "Research prototype — not for clinical use"
    assert DISCLAIMER.startswith("Screening aid — not a diagnosis")
    assert "histopathology" in DISCLAIMER


def test_models_default_to_not_validated():
    info = ModelInfo(
        component=ModelComponent.WL_LESION_CLASSIFIER,
        name="x",
        version="0",
        status=ModelStatus.MODEL_UNAVAILABLE,
    )
    assert info.validated is False


def test_result_requires_clinician_review_by_default():
    field = ScreeningResult.model_fields["triage"].annotation
    assert field.model_fields["requires_clinician_review"].default is True
