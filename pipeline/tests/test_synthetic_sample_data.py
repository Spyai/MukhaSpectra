"""Writing the SYNTHETIC sample set and its documentation."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from synthetic_helpers import SAMPLE_DATA, SMALL, assert_positioning_clean

from mukha_pipeline.schema.models import RiskFactors, VisualFlag
from mukha_pipeline.sites import OralSite
from mukha_pipeline.synthetic import DEMO_KINDS, write_sample_data
from mukha_pipeline.synthetic.sample_data import (
    EXIF_DESCRIPTION,
    default_out_dir,
    main,
)
from mukha_pipeline.synthetic.spectra import CSV_COLUMNS

MANIFEST_KEYS = {"patient", "consent", "risk_factors", "context", "data_provenance", "sites"}


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> tuple[Path, list[Path]]:
    out = tmp_path_factory.mktemp("sample")
    return out, write_sample_data(out, size=SMALL)


def test_writes_every_expected_file(written) -> None:
    out, paths = written
    assert all(p.exists() for p in paths)
    for kind in DEMO_KINDS:
        folder = out / kind
        manifest = json.loads((folder / "manifest.json").read_text())
        assert (folder / "spectrum.csv").exists()
        for site in manifest["sites"]:
            for key, ext in (("wl", "jpg"), ("af", "jpg"), ("dark", "png")):
                name = f"SYNTHETIC_{site['site']}_{key}.{ext}"
                assert site[key] == Path(name).stem
                assert (folder / name).exists()
        images = sorted(p.name for p in folder.glob("*.jpg")) + sorted(
            p.name for p in folder.glob("*.png")
        )
        assert len(images) == 12 and all(n.startswith("SYNTHETIC_") for n in images)


def test_manifest_follows_the_upload_format(written) -> None:
    out, _ = written
    for kind in DEMO_KINDS:
        folder = out / kind
        m = json.loads((folder / "manifest.json").read_text())
        assert set(m) >= MANIFEST_KEYS
        assert m["data_provenance"] == "synthetic"
        assert m["patient"]["name"] == f"SYNTHETIC PATIENT {kind.upper()}"
        assert m["patient"]["external_id"].startswith("SYN-")
        assert m["consent"]["consent_recorded"] is True
        assert m["context"] == {
            "screening_site_name": "SYNTHETIC DEMO CAMP",
            "health_worker_id": "SYN-HW-01",
        }
        RiskFactors.model_validate(m["risk_factors"])
        stems = {p.stem for p in folder.iterdir()}
        assert m["spectral_file"] in stems
        for site in m["sites"]:
            OralSite(site["site"])
            VisualFlag.model_validate(site["visual_flag"])
            assert site["role"] in {"examined", "reference_only"}
            assert site["diascopy"] == "not_done" and site["device_id"] == "SYN-SCOUT-01"
            for key in ("wl", "af", "dark"):
                assert site[key] in stems


def test_manifests_match_the_cases(written) -> None:
    out, _ = written
    moderate = json.loads((out / "moderate" / "manifest.json").read_text())
    assert moderate["risk_factors"]["smokeless_tobacco"]["types"] == ["khaini"]
    flagged = [s for s in moderate["sites"] if s["visual_flag"]["lesion_seen"]]
    assert [s["site"] for s in flagged] == ["buccal_mucosa_left"]
    assert flagged[0]["visual_flag"]["appearance"] == ["white_patch"]
    high = json.loads((out / "high" / "manifest.json").read_text())
    assert high["risk_factors"]["symptoms"]["non_healing_ulcer_over_2_weeks"] is True
    assert "red_patch" in high["sites"][0]["visual_flag"]["appearance"]
    low = json.loads((out / "low" / "manifest.json").read_text())
    assert not any(s["visual_flag"]["lesion_seen"] for s in low["sites"])


def test_spectrum_csv_has_white_rows_only_when_calibrated(written) -> None:
    out, _ = written
    for kind, calibrated in (("low", True), ("moderate", False), ("high", True)):
        lines = (out / kind / "spectrum.csv").read_text().splitlines()
        assert lines[0] == ",".join(CSV_COLUMNS)
        has_white = any(line.startswith("calibration,,,white,") for line in lines)
        assert has_white == calibrated, kind


def test_images_decode_and_carry_synthetic_exif(written) -> None:
    out, _ = written
    wl = Image.open(out / "high" / "SYNTHETIC_lateral_tongue_right_wl.jpg")
    assert wl.size == (SMALL[1], SMALL[0]) and wl.mode == "RGB"
    exif = wl.getexif()
    assert exif[0x010E] == EXIF_DESCRIPTION and "SYNTHETIC" in exif[0x010F]
    sub = exif.get_ifd(0x8769)
    assert float(sub[0x829A]) == pytest.approx(0.008) and sub[0x8827] == 100
    af = Image.open(out / "high" / "SYNTHETIC_lateral_tongue_right_af.jpg")
    assert float(af.getexif().get_ifd(0x8769)[0x829A]) == pytest.approx(0.033)
    dark = Image.open(out / "high" / "SYNTHETIC_lateral_tongue_right_dark.png")
    assert dark.mode == "RGB" and "SYNTHETIC" in dark.info["Description"]
    assert np.asarray(dark).mean() < 10


def test_writing_is_deterministic(written, tmp_path) -> None:
    out, _ = written
    again = write_sample_data(tmp_path, size=SMALL, kinds=("moderate",))
    for p in again:
        assert p.read_bytes() == (out / "moderate" / p.name).read_bytes(), p.name


def test_text_outputs_are_positioning_clean(written) -> None:
    out, _ = written
    for kind in DEMO_KINDS:
        assert_positioning_clean((out / kind / "manifest.json").read_text(), kind)
    assert_positioning_clean(EXIF_DESCRIPTION, "exif")


@pytest.mark.parametrize("readme", ["README.md", "bench/README.md"])
def test_readmes_exist_and_are_positioning_clean(readme: str) -> None:
    text = (SAMPLE_DATA / readme).read_text()
    assert "SYNTHETIC" in text or "synthetic" in text
    assert_positioning_clean(text, readme)


def test_bench_sidecar_template_is_valid_json() -> None:
    template = json.loads((SAMPLE_DATA / "bench" / "sidecar.template.json").read_text())
    assert template["data_provenance"] in {"phantom", "ex_vivo", "volunteer"}
    assert_positioning_clean(json.dumps(template), "sidecar template")


def test_cli_main_writes_to_given_folder(tmp_path, capsys) -> None:
    assert main([str(tmp_path), "--size", "96x128", "--seed", "1"]) == 0
    assert (tmp_path / "high" / "manifest.json").exists()
    assert "SYNTHETIC" in capsys.readouterr().out


def test_module_entry_point(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["mukha_pipeline.synthetic", str(tmp_path), "--size", "96x128"]
    )
    monkeypatch.delitem(sys.modules, "mukha_pipeline.synthetic.__main__", raising=False)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("mukha_pipeline.synthetic", run_name="__main__")
    assert exc.value.code == 0
    assert (tmp_path / "low" / "spectrum.csv").exists()


def test_default_out_dir_points_at_repo_sample_data() -> None:
    assert default_out_dir() == SAMPLE_DATA / "synthetic"


def test_committed_sample_set_is_complete_and_small() -> None:
    root = SAMPLE_DATA / "synthetic"
    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        pytest.skip("sample_data/synthetic not generated in this checkout")
    assert sum(p.stat().st_size for p in files) < 6_500_000
    for kind in DEMO_KINDS:
        m = json.loads((root / kind / "manifest.json").read_text())
        for site in m["sites"]:
            for key in ("wl", "af", "dark"):
                assert any((root / kind).glob(site[key] + ".*"))
