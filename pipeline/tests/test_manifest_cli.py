"""Upload manifest → ScreeningInput, and the ``python -m mukha_pipeline`` CLI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mukha_pipeline.__main__ import main
from mukha_pipeline.manifest import (
    CONSENT_REQUIRED,
    ManifestError,
    build_screening_input,
    parse_manifest,
)
from mukha_pipeline.orchestrator import analyze_screening
from mukha_pipeline.schema.models import DataProvenance, RiskLevel

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "sample_data" / "synthetic"


def _reader(folder: Path):
    def read(field: str):
        for p in folder.iterdir():
            if p.stem == field:
                return p.read_bytes(), p.name
        return None

    return read


def _manifest(kind: str) -> dict:
    return json.loads((SAMPLES / kind / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("kind", "level"),
    [("low", RiskLevel.LOW), ("moderate", RiskLevel.MODERATE), ("high", RiskLevel.HIGH)],
)
def test_sample_files_reach_the_designed_levels(kind, level):
    """The JPEG/PNG sample set (sRGB-encoded files, as a phone writes them) must reach the
    same designed levels as the in-memory demo patients."""
    built = build_screening_input(_manifest(kind), _reader(SAMPLES / kind))
    result = analyze_screening(built.screening_input, strict=True).result
    assert result.triage.overall_risk_level == level
    assert result.synthetic is True


def test_sample_manifest_builds_and_analyses_to_high():
    built = build_screening_input(_manifest("high"), _reader(SAMPLES / "high"))
    inp = built.screening_input
    assert inp.data_provenance == DataProvenance.SYNTHETIC
    assert all(s.wl.meta.synthetic for s in inp.sites)
    assert inp.spectral and inp.spectral_calibrations
    assert built.manifest.patient.name == "SYNTHETIC PATIENT HIGH"
    # PII stays out of the pipeline input: only a pseudonymous id travels on.
    assert "SYNTHETIC PATIENT" not in inp.patient.model_dump_json()
    result = analyze_screening(inp, strict=True).result
    assert result.triage.overall_risk_level == RiskLevel.HIGH


def test_consent_is_checked_first():
    data = _manifest("low")
    data["consent"]["consent_recorded"] = False
    with pytest.raises(ManifestError, match=CONSENT_REQUIRED):
        parse_manifest(data)
    del data["consent"]
    with pytest.raises(ManifestError, match=CONSENT_REQUIRED):
        parse_manifest(data)


def test_invalid_manifest_reports_location():
    data = _manifest("low")
    data["sites"][0]["site"] = "left_ear"
    with pytest.raises(ManifestError, match="sites.0.site"):
        parse_manifest(data)


def test_duplicate_sites_rejected():
    data = _manifest("low")
    data["sites"].append(dict(data["sites"][0]))
    with pytest.raises(ManifestError, match="only once"):
        parse_manifest(data)


def test_missing_file_rejected():
    data = _manifest("low")
    with pytest.raises(ManifestError, match="missing"):
        build_screening_input(data, lambda field: None)


def test_cli_demo(tmp_path: Path, capsys):
    assert main(["analyze", "--demo", "low", "--out", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["triage"]["overall_risk_level"] == "LOW"
    assert any(tmp_path.joinpath("sites").rglob("heatmap_overlay.jpg"))
    assert "Research prototype" in capsys.readouterr().out


def test_cli_manifest(tmp_path: Path):
    folder = tmp_path / "moderate"
    shutil.copytree(SAMPLES / "moderate", folder)
    out = tmp_path / "out"
    assert main(["analyze", "--manifest", str(folder / "manifest.json"), "--out", str(out)]) == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["triage"]["overall_risk_level"] == "MODERATE"


def test_cli_manifest_error(tmp_path: Path, capsys):
    folder = tmp_path / "bad"
    shutil.copytree(SAMPLES / "low", folder)
    data = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    data["consent"]["consent_recorded"] = False
    (folder / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    code = main(["analyze", "--manifest", str(folder / "manifest.json"), "--out", str(tmp_path)])
    assert code == 2
    assert CONSENT_REQUIRED in capsys.readouterr().err


def test_cli_schema(capsys):
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["x-schema-version"] == "1.0.0"


def _run_cli(folder: Path, out: Path) -> int:
    return main(["analyze", "--manifest", str(folder / "manifest.json"), "--out", str(out)])


@pytest.mark.parametrize("target", ["image", "spectrum", "manifest_json"])
def test_cli_unreadable_input_is_a_plain_error(tmp_path: Path, capsys, target):
    """correctness.7: a damaged image, spectrum or manifest gives exit code 2 and one plain
    'error: ...' line, never a Python traceback."""
    folder = tmp_path / "bad"
    shutil.copytree(SAMPLES / "low", folder)
    if target == "image":
        (folder / "SYNTHETIC_buccal_mucosa_left_wl.jpg").write_bytes(b"garbage")
    elif target == "spectrum":
        (folder / "spectrum.csv").write_text("not,a,spectral,file\n1,2,3,4\n", encoding="utf-8")
    else:
        (folder / "manifest.json").write_text("{not json", encoding="utf-8")
    code = _run_cli(folder, tmp_path / "out")
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("error: ") and "Traceback" not in err
    assert not (tmp_path / "out" / "result.json").exists()


def test_cli_missing_manifest_is_a_plain_error(tmp_path: Path, capsys):
    code = main(["analyze", "--manifest", str(tmp_path / "nope.json"), "--out", str(tmp_path)])
    assert code == 2
    assert capsys.readouterr().err.startswith("error: ")
