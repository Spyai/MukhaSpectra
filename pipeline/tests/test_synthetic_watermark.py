"""The visible SYNTHETIC stamp."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic_helpers import SMALL, assert_positioning_clean

from mukha_pipeline.synthetic import stamp_mask, watermark_synthetic
from mukha_pipeline.synthetic._fields import erode
from mukha_pipeline.synthetic.watermark import STAMP_SUBTEXT, STAMP_TEXT, stamp_box


def _pink(shape=SMALL) -> np.ndarray:
    img = np.empty((*shape, 3), np.float32)
    img[:] = (0.85, 0.5, 0.5)
    return img


def test_stamp_changes_only_the_corner() -> None:
    img = _pink()
    out = watermark_synthetic(img)
    stamp = stamp_mask(SMALL)
    assert out.dtype == np.float32 and out.shape == img.shape
    assert np.array_equal(out[~stamp], img[~stamp])
    assert (np.abs(out[stamp] - img[stamp]).max(axis=-1) > 0.1).mean() > 0.9
    x0, y0, x1, y1 = stamp_box(SMALL)
    assert (x0, y1) == (0, SMALL[0]), "flush with the bottom-left corner"
    assert (x1 - x0) * (y1 - y0) < 0.08 * SMALL[0] * SMALL[1]
    assert np.array_equal(img, _pink()), "input untouched"


def test_stamp_is_readable_dark_band_and_not_glare() -> None:
    out = watermark_synthetic(np.ones((*SMALL, 3), np.float32))  # worst case: white below
    stamp = erode(stamp_mask(SMALL), 2)  # skip the band's anti-aliased rim
    value = out.max(axis=-1)[stamp]
    assert np.median(value) <= 0.12, "band reads as background, never as mucosa"
    assert value.max() < 0.96, "text below the glare threshold"
    assert value.max() > 0.8, "text clearly visible"


def test_stamp_handles_uint8_and_grey() -> None:
    img8 = np.full((*SMALL, 3), 200, np.uint8)
    out8 = watermark_synthetic(img8)
    assert out8.dtype == np.uint8 and not np.array_equal(out8, img8)
    grey = np.full(SMALL, 0.5, np.float32)
    out_grey = watermark_synthetic(grey)
    assert out_grey.shape == SMALL and not np.array_equal(out_grey, grey)


def test_dim_and_bandless_variants() -> None:
    dark = np.full((*SMALL, 3), 0.01, np.float32)
    dim = watermark_synthetic(dark, dim=True)
    assert 0.01 < dim.max() <= 0.01 + 0.07, "faint text just above the dark level"
    text_only = watermark_synthetic(_pink(), band=False)
    stamp = stamp_mask(SMALL)
    changed = np.abs(text_only - _pink()).max(axis=-1) > 1e-6
    assert changed.any() and not (changed & ~stamp).any()
    assert changed.sum() < 0.5 * stamp.sum(), "no band, only the text"


def test_stamp_works_on_large_and_tiny_frames() -> None:
    for shape in ((64, 64), (1536, 2048)):
        out = watermark_synthetic(_pink(shape))
        assert not np.array_equal(out, _pink(shape))


def test_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        watermark_synthetic(np.zeros((2, 2, 2, 2), np.float32))


def test_stamp_wording_is_positioning_clean() -> None:
    assert "SYNTHETIC" in STAMP_TEXT
    assert_positioning_clean(STAMP_TEXT + " " + STAMP_SUBTEXT, "stamp")
