from mukha_pipeline.positioning import find_positioning_violations
from mukha_pipeline.schema.models import DISCLAIMER, PROTOTYPE_BANNER


def test_mandated_disclaimers_are_allowed():
    assert find_positioning_violations(DISCLAIMER) == []
    assert find_positioning_violations(PROTOTYPE_BANNER) == []


def test_forbidden_phrases_are_caught():
    for bad in [
        "Diagnosis: oral cancer",
        "Cancer detected at left buccal mucosa",
        "Lesion appears malignant",
        "No cancer found",
        "Patient is cancer-free",
        "AI diagnostic result",
    ]:
        assert find_positioning_violations(bad), bad


def test_screening_language_is_clean():
    ok = "HIGH risk: suspicious finding. Refer for clinical examination and biopsy."
    assert find_positioning_violations(ok) == []
