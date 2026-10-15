"""Image capture QC: white light (WL), 405 nm autofluorescence (AF), dark frame, registration and
segmentation.

Each function takes the full qc.v1.yaml dict and reads its own section. All metrics are recorded
under namespaced keys ("wl.*", "af.*", "dark.*", "registration.*", "segmentation.*") even when
they pass, so thresholds can later be re-tuned on real Scout captures.

Exposure problems are reported before sharpness: the Laplacian variance scales with image
contrast, so a very dark or blown-out photo also "looks" blurred. Telling the health worker to
hold the phone steady when the real fault is the light would waste the retake, so the blur verdict
is suppressed (metric still recorded) whenever an exposure error is already raised.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from mukha_pipeline.qc.imageops import (
    clip_fractions,
    glare_fraction,
    gray255,
    laplacian_variance,
    prepare_image,
)
from mukha_pipeline.qc.messages import make_issue
from mukha_pipeline.qc.result import build_result
from mukha_pipeline.schema.models import QCIssue, QCResult, QCSeverity
from mukha_pipeline.types import Masks, RegistrationResult

# Defaults for keys added to qc.v1.yaml during Phase 1, so older config files keep working.
AF_DEFAULTS: Mapping[str, float] = {
    "signal_sum_min": 0.06,
    "green_fraction_min": 0.36,
    "green_dominance_min": 0.03,
    "white_light_blue_fraction_min": 0.18,
    "white_light_red_excess_min": 0.05,
}
MIN_SIGNAL_PIXEL_FRACTION = 0.01
# Overall exposure faults. They also depress the Laplacian variance (it scales with contrast), so
# a blur prompt on top would send the health worker after the wrong fault.
EXPOSURE_CODES = frozenset({"too_dark", "too_bright", "af_signal_low", "af_overexposed"})


def _below(code: str, metric: str, value: float, threshold: float) -> list[QCIssue]:
    if value < threshold:
        return [make_issue(code, metric=metric, value=value, threshold=threshold)]
    return []


def _above(code: str, metric: str, value: float, threshold: float) -> list[QCIssue]:
    if value > threshold:
        return [make_issue(code, metric=metric, value=value, threshold=threshold)]
    return []


def _exposure_error(found: list[QCIssue]) -> bool:
    return any(i.code in EXPOSURE_CODES for i in found)


# --------------------------------------------------------------------------------------------
# White light
# --------------------------------------------------------------------------------------------


def qc_white_light(img: np.ndarray, cfg: Mapping) -> QCResult:
    """Sharpness, exposure, clipping and specular glare of the white-light photo.

    Consequences are not reported on top of their cause: a too-bright photo is not also told
    about clipping and glare, a too-dark one not about black areas or blur.
    """
    c = cfg["white_light"]
    prep = prepare_image(img, cfg, "wl")
    issues, metrics = prep.issues, prep.metrics
    if prep.rgb is None:
        return build_result(issues, metrics, cfg)
    rgb = prep.rgb

    mean = float(gray255(rgb).mean())
    low, high = clip_fractions(rgb)
    glare = glare_fraction(rgb, c["glare_value_min"], c["glare_saturation_max"])
    sharpness = laplacian_variance(rgb)
    metrics |= {
        "wl.mean_brightness": mean,
        "wl.clip_low_fraction": low,
        "wl.clip_high_fraction": high,
        "wl.glare_fraction": glare,
        "wl.blur_laplacian_var": sharpness,
    }

    found = _below("too_dark", "wl.mean_brightness", mean, c["mean_brightness_min"])
    found += _above("too_bright", "wl.mean_brightness", mean, c["mean_brightness_max"])
    codes = {i.code for i in found}
    if "too_dark" not in codes:
        # Warning only: the oropharynx and shadows beside the clip are legitimately black, and
        # segmentation keeps them out of the analysed mucosa.
        found += _above("clipping_dark", "wl.clip_low_fraction", low, c["clip_low_fraction_max"])
    if "too_bright" not in codes:
        found += _above(
            "clipping_bright", "wl.clip_high_fraction", high, c["clip_high_fraction_max"]
        )
        found += _above("glare", "wl.glare_fraction", glare, c["glare_fraction_max"]) or _above(
            "glare_minor", "wl.glare_fraction", glare, c["glare_warning_fraction"]
        )
    if not _exposure_error(found):
        found += _below("blur", "wl.blur_laplacian_var", sharpness, c["blur_laplacian_var_min"])
    return build_result(issues + found, metrics, cfg)


# --------------------------------------------------------------------------------------------
# Autofluorescence
# --------------------------------------------------------------------------------------------


def _af_param(c: Mapping, key: str) -> float:
    return float(c.get(key, AF_DEFAULTS[key]))


def colour_shares(rgb: np.ndarray, signal_sum_min: float) -> tuple[float, np.ndarray]:
    """(fraction of pixels with signal, median R, G, B share of R+G+B over those pixels).

    Shares are only meaningful where there is light: in near-black pixels read noise makes
    every share wander around 1/3, which could hide a missing filter or fake one.
    """
    total = rgb.sum(axis=2)
    signal = total >= signal_sum_min
    frac = float(signal.mean())
    if frac < MIN_SIGNAL_PIXEL_FRACTION:
        return frac, np.full(3, np.nan)
    shares = rgb[signal] / total[signal][:, None]
    return frac, np.median(shares, axis=0)


def _af_colour_issues(shares: np.ndarray, c: Mapping) -> list[QCIssue]:
    """Is this really a filtered 405 nm fluorescence photo?

    * blue share high: reflected violet excitation reaches the sensor (filter missing) - error;
    * red-dominant with ordinary blue: a white-light photo (pink mucosa reflects blue, whereas
      genuine AF behind the long-pass filter has almost none). Analysing it as AF is dangerous:
      a white patch is bright green in white light and would hide fluorescence loss - error;
    * green not dominant at all (green share at most ``green_dominance_min`` above the larger of
      red and blue): auto white balance or a greyscale file neutralised the green cast, or the
      photo is red-dominant (white light through the filter, or red porphyrin over the whole
      field). Tissue green cannot be judged: every pixel would look like red porphyrin (plaque)
      to the analysis, which then discounts a real fluorescence loss - error, retake;
    * green dominant but weakly (share below ``green_fraction_min``): a partial colour shift;
      the analysis still runs, with a warning.
    """
    red, green, blue = (float(x) for x in shares)
    excess = _af_param(c, "white_light_red_excess_min")
    found = _above("af_filter_leak", "af.blue_fraction", blue, c["blue_fraction_max"])
    if not found and blue >= _af_param(c, "white_light_blue_fraction_min"):
        found = _above("af_white_light", "af.red_minus_green_fraction", red - green, excess)
    if found:
        return found
    dominance = green - max(red, blue)
    dominance_min = _af_param(c, "green_dominance_min")
    if dominance <= dominance_min:
        return [
            make_issue(
                "af_not_green",
                metric="af.green_dominance",
                value=dominance,
                threshold=dominance_min,
                severity=QCSeverity.ERROR,
            )
        ]
    return _below("af_not_green", "af.green_fraction", green, _af_param(c, "green_fraction_min"))


def _auto_white_balance(white_balance: str | None) -> bool:
    return isinstance(white_balance, str) and white_balance.strip().lower().startswith("auto")


def qc_autofluorescence(
    img: np.ndarray, cfg: Mapping, *, white_balance: str | None = None
) -> QCResult:
    """Excitation on, emission filter fitted, really fluorescence, not over-exposed, sharp.

    Physics: under 405 nm excitation healthy mucosa emits broadband green (collagen, NADH, FAD,
    peak ~500-520 nm). The long-pass emission filter blocks the reflected violet light, so a good
    AF photo is green-dominant with little blue. A weak green 90th percentile means the violet
    LED was off or covered.

    ``white_balance`` is the capture's recorded white-balance mode (EXIF / capture metadata).
    ``"auto"`` adds the warning ``af_auto_white_balance``: auto white balance pulls the green
    glow towards grey and shifts the green/red ratio the analysis relies on, even when the photo
    still looks green enough to pass.
    """
    c = cfg["autofluorescence"]
    prep = prepare_image(img, cfg, "af")
    issues, metrics = prep.issues, prep.metrics
    if _auto_white_balance(white_balance):
        issues.append(make_issue("af_auto_white_balance"))
    if prep.rgb is None:
        return build_result(issues, metrics, cfg)
    rgb = prep.rgb
    greyscale = any(i.code == "image_grayscale" for i in issues)

    green_p90 = float(np.percentile(rgb[..., 1], 90))
    _, high = clip_fractions(rgb)
    mean = float(gray255(rgb).mean())
    signal_frac, shares = colour_shares(rgb, _af_param(c, "signal_sum_min"))
    sharpness = laplacian_variance(rgb)
    metrics |= {
        "af.green_p90": green_p90,
        "af.clip_high_fraction": high,
        "af.mean_brightness": mean,
        "af.signal_pixel_fraction": signal_frac,
        "af.red_fraction": float(shares[0]),
        "af.green_fraction": float(shares[1]),
        "af.blue_fraction": float(shares[2]),
        "af.green_dominance": float(shares[1] - max(shares[0], shares[2])),
        "af.blur_laplacian_var": sharpness,
    }

    found = _below("af_signal_low", "af.green_p90", green_p90, c["green_p90_min"])
    found += _above(
        "af_overexposed", "af.clip_high_fraction", high, c["clip_high_fraction_max"]
    ) or _above("af_overexposed", "af.mean_brightness", mean, c["mean_brightness_max"])
    if np.isfinite(shares).all() and not greyscale:  # greyscale already has its retake prompt
        found += _af_colour_issues(shares, c)
    if not _exposure_error(found):
        found += _below("af_blur", "af.blur_laplacian_var", sharpness, c["blur_laplacian_var_min"])
    return build_result(issues + found, metrics, cfg)


# --------------------------------------------------------------------------------------------
# Dark frame
# --------------------------------------------------------------------------------------------


def qc_dark_frame(img: np.ndarray, cfg: Mapping) -> QCResult:
    """Room light check: the frame taken with every LED off should be close to black.

    Ambient light adds to the AF green signal and is not uniform, so it can fill in (mask) a
    real area of fluorescence loss. A grayscale dark frame is acceptable: only its level matters.
    """
    c = cfg["dark_frame"]
    prep = prepare_image(img, cfg, "dark", require_colour=False)
    issues, metrics = prep.issues, prep.metrics
    if prep.rgb is None:
        return build_result(issues, metrics, cfg)
    mean = float(prep.rgb.mean())
    metrics["dark.mean"] = mean
    found = _above("ambient_light", "dark.mean", mean, c["mean_max"]) or _above(
        "ambient_light_minor", "dark.mean", mean, c["mean_warning"]
    )
    return build_result(issues + found, metrics, cfg)


# --------------------------------------------------------------------------------------------
# Registration and segmentation
# --------------------------------------------------------------------------------------------


def _finite(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def qc_registration(reg: RegistrationResult, cfg: Mapping) -> QCResult:
    """WL<->AF alignment. A misaligned overlay puts the AF-loss heatmap on the wrong tissue,
    which could send a biopsy to the wrong spot, so low confidence blocks the site.

    A registration the aligner itself did not accept (``passed=False``, identity fallback) is an
    error whatever its confidence.
    """
    c = cfg["registration"]
    conf = _finite(getattr(reg, "confidence", None))
    passed = bool(getattr(reg, "passed", False))
    metrics = {"registration.confidence": conf, "registration.passed": float(passed)}
    found = _below("misaligned", "registration.confidence", conf, c["confidence_min"])
    if not found and not passed:
        found = [make_issue("misaligned", metric="registration.passed", value=0, threshold=1)]
    if not found:
        found = _below(
            "alignment_marginal", "registration.confidence", conf, c["confidence_warning"]
        )
    return build_result(found, metrics, cfg)


def _fraction(mask: object) -> float:
    try:
        arr = np.asarray(mask, dtype=bool)
    except (TypeError, ValueError):
        return 0.0
    return float(arr.mean()) if arr.size else 0.0


def qc_segmentation(masks: Masks, cfg: Mapping, *, status: str | None = None) -> QCResult:
    """Enough analysable mucosa in view (teeth, lips, background and glare excluded), and a
    tissue outline the segmenter itself trusts.

    ``status`` is ``SegmentationSummary.status`` (``ok`` | ``low_confidence`` | ``failed``).
    Tissue the segmenter could not classify is left out of the mucosa mask and never checked
    for fluorescence loss, so:

    * ``failed``: the tissue could not be outlined at all - error ``segmentation_failed``
      (unless ``insufficient_mucosa`` already blocks the site for the same reason);
    * ``low_confidence``: warning ``segmentation_low_confidence``; the site is analysed, and the
      triage does not let such a site support a LOW result.
    """
    c = cfg["segmentation"]
    fraction = _fraction(getattr(masks, "mucosa", None))
    metrics = {
        "segmentation.mucosa_fraction": fraction,
        "segmentation.confidence": _finite(getattr(masks, "confidence", None), float("nan")),
        "segmentation.specular_fraction": _fraction(getattr(masks, "specular", None)),
    }
    minimum = c["mucosa_fraction_min"]
    found = _below("insufficient_mucosa", "segmentation.mucosa_fraction", fraction, minimum)
    if status == "failed" and not found:
        found = [make_issue("segmentation_failed", metric="segmentation.confidence")]
    elif status == "low_confidence":
        found.append(make_issue("segmentation_low_confidence", metric="segmentation.confidence"))
    return build_result(found, metrics, cfg)
