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
        "No disease seen",
        "All clear — see you next year",
        "Lesion looks benign",
        "Precancerous change",
        "Possible tumour on the tongue",
        "Consistent with carcinoma",
        "The patient is healthy",
        "Your mouth is normal",
    ]:
        assert find_positioning_violations(bad), bad


def test_screening_language_is_clean():
    ok = "HIGH risk: suspicious finding. Refer for clinical examination and biopsy."
    assert find_positioning_violations(ok) == []


def test_low_result_wording_is_clean():
    text = (
        "No high-risk features were found by this screen. This does not rule out disease; "
        "re-screening in 12 months matters, sooner if any symptom appears."
    )
    assert find_positioning_violations(text) == []
