"""QC against the project's own SYNTHETIC generator (mukha_pipeline.synthetic).

The demo patients are built from these images, so a plausible synthetic capture must pass QC and
a clearly degraded one must fail; otherwise every demo would come out INCONCLUSIVE (or worse,
degraded captures would be analysed). Parts of the generator that do not exist yet are skipped.
"""

import cv2
import numpy as np
import pytest

from mukha_pipeline.qc import qc_autofluorescence, qc_dark_frame, qc_white_light
from mukha_pipeline.sites import OralSite

synthetic = pytest.importorskip("mukha_pipeline.synthetic.wl")

SITES = [OralSite.BUCCAL_MUCOSA_LEFT, OralSite.TONGUE_DORSUM, OralSite.LOWER_GINGIVA]


def _wl(site: OralSite, seed: int = 0) -> np.ndarray:
    wl, _ = synthetic.generate_wl_image(np.random.default_rng(seed), site=site)
    return wl


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.value)
def test_synthetic_wl_passes_and_blurred_fails(qc_cfg, site):
    wl = _wl(site)
    r = qc_white_light(wl, qc_cfg)
    assert r.passed, [i.code for i in r.issues]
    blurred = qc_white_light(cv2.GaussianBlur(wl, (0, 0), 4.0), qc_cfg)
    assert [i.code for i in blurred.issues] == ["blur"]
    assert not qc_white_light(np.clip(wl * 0.2, 0, 1), qc_cfg).passed
    assert not qc_white_light(np.clip(wl * 2.5, 0, 1), qc_cfg).passed


def _af_generator():
    try:
        from mukha_pipeline.synthetic import generate_af_from_wl
    except ImportError:
        pytest.skip("synthetic AF generator not available yet")
    return generate_af_from_wl


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.value)
def test_synthetic_af_passes_and_degraded_fails(qc_cfg, site):
    generate_af_from_wl = _af_generator()
    af = generate_af_from_wl(_wl(site), np.random.default_rng(1), misalign=False)
    r = qc_autofluorescence(af, qc_cfg)
    assert r.passed, ([i.code for i in r.issues], r.metrics)
    assert not qc_autofluorescence(cv2.GaussianBlur(af, (0, 0), 8.0), qc_cfg).passed
    leak = af.copy()
    leak[..., 2] = np.clip(leak[..., 2] + 0.6, 0, 1)  # no emission filter: violet floods blue
    assert "af_filter_leak" in [i.code for i in qc_autofluorescence(leak, qc_cfg).issues]


def test_synthetic_dark_frame_passes(qc_cfg):
    try:
        from mukha_pipeline.synthetic import generate_dark_frame
    except ImportError:
        pytest.skip("synthetic dark-frame generator not available yet")
    dark = generate_dark_frame((768, 1024), np.random.default_rng(2))
    assert qc_dark_frame(dark, qc_cfg).passed
