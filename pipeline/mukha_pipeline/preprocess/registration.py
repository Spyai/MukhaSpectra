"""Register the autofluorescence (AF) image onto the white-light (WL) frame.

Why this is hard: WL is a colour reflectance photo, AF is green emission under 405 nm light, so
pixel intensities are not comparable (teeth: bright in both; mucosa: pink vs green; blood and
vessels: dark in both because haemoglobin absorbs the violet excitation and the green emission).
What the two share is *structure*: tissue folds, the teeth line, lesion borders and the mouth
outline. Both images are therefore turned into a structural representation (CLAHE-equalised
gray -> Gaussian -> Sobel gradient magnitude -> robust normalisation) before alignment.

Why it matters clinically: FVL is a per-pixel comparison of AF against the WL-defined mucosa. A
misaligned AF image can paint a dark tooth gap or the mouth border onto mucosa (false FVL) or
shift a true dark region off the lesion (false reassurance). Every warp is therefore sanity
checked, and an untrusted alignment returns identity with ``passed=False`` so QC can stop the
site rather than analyse it.

Warp convention (``RegistrationResult.warp``): a 3x3 float64 matrix ``H`` in homogeneous pixel
coordinates (x = column, y = row, pixel centres at integers) mapping an **AF pixel coordinate to
the WL pixel coordinate** showing the same tissue: ``[x_wl, y_wl, 1]^T ~ H @ [x_af, y_af, 1]^T``.
It includes any AF -> WL resolution change. ``registered`` is the AF image resampled on the WL
grid (``registered(p_wl) = AF(H^-1 p_wl)``), so ``cv2.warpPerspective(af, H, (W_wl, H_wl))``
reproduces it.

ECC note: ``cv2.findTransformECC(template, input, W)`` maximises the correlation of
``template(x)`` with ``input(W x)``; with template = WL and input = AF, ``W`` maps WL -> AF, so the
AF -> WL warp is ``W^-1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from mukha_pipeline.types import RegistrationResult

DEFAULTS: dict[str, Any] = {
    "method_order": ["ecc_affine", "orb_homography"],
    "estimation_max_side_px": 512,
    "ecc_max_iterations": 200,
    "ecc_epsilon": 1.0e-5,
    "ecc_pyramid_levels": 3,
    "ecc_gaussian_sigma": 2.0,
    "ecc_gauss_filt_size": 5,
    "orb_features": 3000,
    "orb_ratio_test": 0.75,
    "orb_min_matches": 12,
    "orb_full_confidence_inliers": 50,
    "ransac_reproj_threshold": 4.0,
    "max_translation_fraction": 0.15,
    "max_scale_change": 0.15,
    "max_rotation_deg": 10.0,
    "accept_confidence_min": 0.30,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": 8,
    "valid_erode_px": 2,
}

METHODS = ("ecc_affine", "ecc_homography", "orb_homography")
MIN_ECC_LEVEL_SIDE_PX = 24  # coarser pyramid levels hold too few pixels for a stable ECC fit
MIN_ESTIMATION_SIDE_PX = 16
# An identity warp was never verified by any method, so its confidence is capped below every
# acceptance threshold; QC can never pass it on confidence alone.
IDENTITY_CONFIDENCE_CAP = 0.2


@dataclass
class _Attempt:
    """Outcome of one estimation method at estimation resolution."""

    method: str
    warp_est: np.ndarray | None  # 3x3, AF-estimation coords -> WL-estimation coords
    confidence: float
    reason: str | None  # None when the method produced a usable estimate


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def register_af_to_wl(wl: np.ndarray, af: np.ndarray, cfg: dict | None) -> RegistrationResult:
    """Align ``af`` to ``wl``; ``cfg`` is the ``registration`` section of the analysis config.

    Methods are tried in ``method_order`` (default ECC affine, then ORB + RANSAC homography) on
    copies whose longest side is ``estimation_max_side_px``; the accepted warp is then scaled to
    the full WL resolution. A method is accepted when its warp passes the sanity limits
    (translation, scale, rotation, no fold) and its confidence exceeds
    ``accept_confidence_min``. Otherwise identity is returned with ``passed=False``.
    """
    p = _params(cfg)
    wl_f = _as_rgb(wl)
    af_f = _as_rgb(af)
    h, w = wl_f.shape[:2]
    if min(h, w) == 0 or min(af_f.shape[:2]) == 0:
        return _empty_result(h, w, "Empty white-light or fluorescence image; nothing to align.")

    # 1. Put the AF image on the WL pixel grid; this resize is part of the returned warp.
    to_wl_grid = _resize_matrix(af_f.shape[1], af_f.shape[0], w, h)
    af_on_grid = _resize(af_f, w, h)

    # 2. Estimation copies (speed, and gradients at this scale reflect anatomy, not noise).
    scale = min(1.0, float(p["estimation_max_side_px"]) / max(h, w))
    ew, eh = max(1, round(w * scale)), max(1, round(h * scale))
    to_est = _resize_matrix(w, h, ew, eh)
    wl_gray = _to_u8(_resize(_wl_gray(wl_f), ew, eh))
    af_gray = _to_u8(_resize(_af_gray(af_on_grid), ew, eh))
    clahe = cv2.createCLAHE(
        clipLimit=float(p["clahe_clip_limit"]),
        tileGridSize=(int(p["clahe_tile_grid"]), int(p["clahe_tile_grid"])),
    )
    wl_eq, af_eq = clahe.apply(wl_gray), clahe.apply(af_gray)
    sigma = float(p["ecc_gaussian_sigma"])
    wl_struct, af_struct = _structure(wl_eq, sigma), _structure(af_eq, sigma)

    attempts: list[_Attempt] = []
    for method in _method_order(p):
        if min(ew, eh) < MIN_ESTIMATION_SIDE_PX:
            attempts.append(_Attempt(method, None, 0.0, "image too small to align"))
            continue
        if method.startswith("ecc_"):
            attempt = _ecc_attempt(wl_struct, af_struct, method, p)
        else:
            attempt = _orb_attempt(wl_eq, af_eq, p)
        if attempt.warp_est is not None and attempt.reason is None:
            residual = np.linalg.inv(to_est) @ attempt.warp_est @ to_est  # AF grid -> WL
            attempt.reason = _sanity_problem(residual, w, h, p)
            if attempt.reason is None and attempt.confidence <= p["accept_confidence_min"]:
                attempt.reason = (
                    f"confidence {attempt.confidence:.2f} at or below "
                    f"{float(p['accept_confidence_min']):.2f}"
                )
            if attempt.reason is None:
                note = _success_note(method, attempts)
                return _build_result(
                    af_on_grid, residual, to_wl_grid, method, attempt.confidence, True, note, p
                )
        attempts.append(attempt)

    identity_conf = min(_correlation(wl_struct, af_struct), IDENTITY_CONFIDENCE_CAP)
    return _build_result(
        af_on_grid,
        np.eye(3),
        to_wl_grid,
        "identity",
        identity_conf,
        False,
        _failure_note(attempts, wl_f.shape, af_f.shape),
        p,
    )


# --------------------------------------------------------------------------------------------
# Configuration and image preparation
# --------------------------------------------------------------------------------------------


def _params(cfg: dict | None) -> dict[str, Any]:
    """Config with defaults filled in, so partial or missing sections still work."""
    merged = dict(DEFAULTS)
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _method_order(p: dict[str, Any]) -> list[str]:
    """Known methods from ``method_order`` in order, without duplicates."""
    order = p.get("method_order") or []
    if isinstance(order, str):
        order = [order]
    seen: list[str] = []
    for m in order:
        if m in METHODS and m not in seen:
            seen.append(m)
    return seen


def _as_rgb(img: np.ndarray) -> np.ndarray:
    """Float32 (H, W, 3) in [0, 1] from gray / RGB / RGBA / integer input; non-finite -> 0."""
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., :3]
    else:
        arr = np.zeros((0, 0, 3), dtype=np.float32)
    return np.ascontiguousarray(np.clip(arr, 0.0, 1.0))


def _wl_gray(img: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of the WL photo."""
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def _af_gray(img: np.ndarray) -> np.ndarray:
    """Green channel of the AF photo: the tissue autofluorescence band (red is porphyrin,
    blue is leaked excitation; both would add structure that WL does not share)."""
    return np.ascontiguousarray(img[..., 1])


def _to_u8(gray: np.ndarray) -> np.ndarray:
    return np.clip(gray * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize with area averaging when shrinking (no aliasing), linear when enlarging."""
    if img.shape[1] == width and img.shape[0] == height:
        return img
    shrinking = img.shape[1] > width and img.shape[0] > height
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(img, (width, height), interpolation=interp)


def _resize_matrix(src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    """3x3 map from source to destination pixel coords under ``cv2.resize``.

    OpenCV resizes with the pixel-centre convention ``x_dst = (x_src + 0.5) * s - 0.5``; using
    the exact map (not a bare scale) keeps sub-pixel accuracy when warps change resolution.
    """
    sx = dst_w / max(src_w, 1)
    sy = dst_h / max(src_h, 1)
    return np.array(
        [[sx, 0.0, 0.5 * (sx - 1.0)], [0.0, sy, 0.5 * (sy - 1.0)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _structure(gray_eq: np.ndarray, sigma: float) -> np.ndarray:
    """Modality-independent structure map: Gaussian -> Sobel gradient magnitude -> [0, 1].

    Gradient magnitude ignores the sign and level of contrast, which differ between WL and AF,
    and keeps where edges are. Normalising by the 99th percentile keeps a few specular edges from
    dominating.
    """
    f = gray_eq.astype(np.float32) / 255.0
    if sigma > 0:
        f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    mag = cv2.magnitude(gx, gy)
    top = float(np.percentile(mag, 99)) if mag.size else 0.0
    if not math.isfinite(top) or top <= 1e-8:
        return np.zeros_like(mag)
    return np.clip(mag / top, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------------
# ECC (primary)
# --------------------------------------------------------------------------------------------


def _ecc_attempt(template: np.ndarray, moving: np.ndarray, method: str, p: dict) -> _Attempt:
    """Coarse-to-fine ECC. Returns the AF -> WL warp at estimation resolution."""
    motion = cv2.MOTION_HOMOGRAPHY if method == "ecc_homography" else cv2.MOTION_AFFINE
    try:
        wl_to_af, cc = _ecc_pyramid(template, moving, motion, p)
    except (cv2.error, np.linalg.LinAlgError) as e:
        return _Attempt(method, None, 0.0, f"did not converge ({_short(e)})")
    if not np.all(np.isfinite(wl_to_af)) or abs(np.linalg.det(wl_to_af)) < 1e-9:
        return _Attempt(method, None, 0.0, "degenerate warp")
    conf = float(np.clip(cc, 0.0, 1.0)) if math.isfinite(cc) else 0.0
    return _Attempt(method, np.linalg.inv(wl_to_af), conf, None)


def _ecc_pyramid(
    template: np.ndarray, moving: np.ndarray, motion: int, p: dict
) -> tuple[np.ndarray, float]:
    """Run ECC from the coarsest to the finest level; returns (W: WL -> AF, final ECC cc)."""
    levels = max(1, int(p["ecc_pyramid_levels"]))
    pyr_t, pyr_m, down = [template], [moving], []  # down[k]: level k -> level k+1 coords
    while len(pyr_t) < levels:
        h, w = pyr_t[-1].shape
        nw, nh = w // 2, h // 2
        if min(nw, nh) < MIN_ECC_LEVEL_SIDE_PX:
            break
        down.append(_resize_matrix(w, h, nw, nh))
        pyr_t.append(cv2.resize(pyr_t[-1], (nw, nh), interpolation=cv2.INTER_AREA))
        pyr_m.append(cv2.resize(pyr_m[-1], (nw, nh), interpolation=cv2.INTER_AREA))

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max(1, int(p["ecc_max_iterations"])),
        float(p["ecc_epsilon"]),
    )
    gauss = int(p["ecc_gauss_filt_size"])
    gauss = gauss if gauss % 2 == 1 else gauss + 1
    warp = _initial_translation(pyr_t[-1], pyr_m[-1])
    cc = 0.0
    for level in range(len(pyr_t) - 1, -1, -1):
        if level < len(pyr_t) - 1:
            d = down[level]  # finer (level) -> coarser (level + 1)
            warp = np.linalg.inv(d) @ warp @ d
        init = warp.astype(np.float32) if motion == cv2.MOTION_HOMOGRAPHY else (
            warp[:2].astype(np.float32)
        )
        cc, out = cv2.findTransformECC(
            pyr_t[level], pyr_m[level], init, motion, criteria, None, gauss
        )
        warp = np.vstack([out, [0.0, 0.0, 1.0]]) if out.shape[0] == 2 else out
        warp = warp.astype(np.float64)
    return warp, float(cc)


def _initial_translation(template: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Start ECC from the better of identity and a phase-correlation translation.

    ECC is a local optimiser; a hand-held shift of tens of pixels can exceed its capture range
    at full resolution. Phase correlation finds a global shift cheaply, and is kept only if it
    improves the structural correlation (so a spurious peak cannot make things worse).
    """
    identity = np.eye(3)
    if min(template.shape) < 8:
        return identity
    window = cv2.createHanningWindow((template.shape[1], template.shape[0]), cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(template, moving, window)
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return identity
    shifted = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    if _correlation(template, moving, shifted) > _correlation(template, moving, identity):
        return shifted
    return identity


def _correlation(template: np.ndarray, moving: np.ndarray, wl_to_af: np.ndarray | None = None
                 ) -> float:
    """Pearson correlation of ``template(x)`` with ``moving(W x)`` over the overlap, in [0, 1]."""
    if wl_to_af is None:
        warped, mask = moving, np.ones(moving.shape, dtype=bool)
    else:
        h, w = template.shape
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        warped = cv2.warpAffine(moving, wl_to_af[:2], (w, h), flags=flags)
        ones = np.ones(moving.shape, dtype=np.uint8)
        mask = cv2.warpAffine(ones, wl_to_af[:2], (w, h), flags=cv2.INTER_NEAREST
                              | cv2.WARP_INVERSE_MAP) > 0
    if int(mask.sum()) < 16:
        return 0.0
    a = template[mask].astype(np.float64)
    b = warped[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


# --------------------------------------------------------------------------------------------
# ORB + RANSAC (fallback)
# --------------------------------------------------------------------------------------------


def _orb_attempt(wl_eq: np.ndarray, af_eq: np.ndarray, p: dict) -> _Attempt:
    """Feature matching on CLAHE grayscale; confidence = inlier ratio x min(1, inliers / N)."""
    method = "orb_homography"
    try:
        orb = cv2.ORB_create(nfeatures=max(10, int(p["orb_features"])))
        kp_af, des_af = orb.detectAndCompute(af_eq, None)
        kp_wl, des_wl = orb.detectAndCompute(wl_eq, None)
    except cv2.error as e:
        return _Attempt(method, None, 0.0, f"feature detection failed ({_short(e)})")
    if des_af is None or des_wl is None or len(kp_af) < 4 or len(kp_wl) < 4:
        return _Attempt(method, None, 0.0, "too few distinctive features")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_af, des_wl, k=2)
    ratio = float(p["orb_ratio_test"])
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    min_matches = max(4, int(p["orb_min_matches"]))
    if len(good) < min_matches:
        return _Attempt(method, None, 0.0, f"only {len(good)} reliable feature matches")
    src = np.float32([kp_af[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_wl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        src, dst, cv2.RANSAC, float(p["ransac_reproj_threshold"])
    )
    if homography is None or inlier_mask is None or not np.all(np.isfinite(homography)):
        return _Attempt(method, None, 0.0, "no consistent homography")
    inliers = int(inlier_mask.sum())
    full = max(1, int(p["orb_full_confidence_inliers"]))
    conf = (inliers / len(good)) * min(1.0, inliers / full)
    homography = homography.astype(np.float64)
    if abs(homography[2, 2]) > 1e-12:
        homography /= homography[2, 2]
    return _Attempt(method, homography, float(np.clip(conf, 0.0, 1.0)), None)


# --------------------------------------------------------------------------------------------
# Sanity checks and result assembly
# --------------------------------------------------------------------------------------------


def _apply(h: np.ndarray, x: float, y: float) -> tuple[float, float, float]:
    """Map (x, y) with homography ``h``; returns (u, v, w) with w the projective denominator."""
    vec = h @ np.array([x, y, 1.0])
    return float(vec[0] / vec[2]), float(vec[1] / vec[2]), float(vec[2])


def _jacobian(h: np.ndarray, x: float, y: float) -> np.ndarray:
    """2x2 local linear part of ``h`` at (x, y); equals the affine part for affine warps."""
    u, v, den = _apply(h, x, y)
    g, k = h[2, 0], h[2, 1]
    return np.array(
        [
            [(h[0, 0] - g * u) / den, (h[0, 1] - k * u) / den],
            [(h[1, 0] - g * v) / den, (h[1, 1] - k * v) / den],
        ]
    )


def _sanity_problem(warp: np.ndarray, w: int, h: int, p: dict) -> str | None:
    """Return why ``warp`` (AF grid -> WL, full res) is implausible, or None if it is fine.

    WL and AF are taken a fraction of a second apart through the same clip, so the true motion
    is a small shift, rotation and zoom. Checked at the centre and the four corners: centre shift
    per axis <= ``max_translation_fraction`` of that dimension, local scale (singular values)
    within 1 +/- ``max_scale_change``, rotation <= ``max_rotation_deg``, no mirror or fold.
    """
    if warp.shape != (3, 3) or not np.all(np.isfinite(warp)):
        return "non-finite warp"
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    points = [(cx, cy), (0.0, 0.0), (w - 1.0, 0.0), (0.0, h - 1.0), (w - 1.0, h - 1.0)]
    for x, y in points:
        if (warp[2] @ np.array([x, y, 1.0])) <= 1e-6:
            return "perspective fold"
    u, v, _ = _apply(warp, cx, cy)
    max_t = float(p["max_translation_fraction"])
    if abs(u - cx) > max_t * w or abs(v - cy) > max_t * h:
        return f"shift ({u - cx:.0f}, {v - cy:.0f}) px exceeds {max_t:.0%} of the image"
    max_s = float(p["max_scale_change"])
    max_rot = float(p["max_rotation_deg"])
    for x, y in points:
        jac = _jacobian(warp, x, y)
        if np.linalg.det(jac) <= 0:
            return "mirrored or folded warp"
        sv = np.linalg.svd(jac, compute_uv=False)
        if sv.min() < 1.0 - max_s or sv.max() > 1.0 + max_s:
            return f"scale change {sv.min():.2f}-{sv.max():.2f} outside 1 +/- {max_s:.2f}"
        rot = math.degrees(math.atan2(jac[1, 0] - jac[0, 1], jac[0, 0] + jac[1, 1]))
        if abs(rot) > max_rot:
            return f"rotation {rot:.1f} deg exceeds {max_rot:.0f} deg"
    return None


def _warp_image(img: np.ndarray, warp: np.ndarray, w: int, h: int, interp: int) -> np.ndarray:
    """Resample ``img`` onto the WL grid with a forward (source -> destination) warp."""
    if np.allclose(warp[2], [0.0, 0.0, 1.0]):
        return cv2.warpAffine(img, warp[:2], (w, h), flags=interp,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.warpPerspective(img, warp, (w, h), flags=interp,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _build_result(
    af_on_grid: np.ndarray,
    residual: np.ndarray,
    to_wl_grid: np.ndarray,
    method: str,
    confidence: float,
    passed: bool,
    note: str | None,
    p: dict,
) -> RegistrationResult:
    """Warp the grid-resized AF into the WL frame and package the result."""
    h, w = af_on_grid.shape[:2]
    if np.allclose(residual, np.eye(3)):
        registered = af_on_grid.copy()
        covered = np.ones((h, w), dtype=np.uint8)
    else:
        registered = _warp_image(af_on_grid, residual, w, h, cv2.INTER_LINEAR)
        ones = np.ones((h, w), dtype=np.uint8)
        covered = _warp_image(ones, residual, w, h, cv2.INTER_NEAREST)
        # Pixels within a couple of px of the warped border blend AF with the zero fill.
        erode = max(0, int(p["valid_erode_px"]))
        if erode:
            kernel = np.ones((2 * erode + 1, 2 * erode + 1), dtype=np.uint8)
            covered = cv2.erode(covered, kernel)
    registered = np.clip(registered, 0.0, 1.0).astype(np.float32)
    return RegistrationResult(
        warp=(residual @ to_wl_grid).astype(np.float64),
        method=method,  # type: ignore[arg-type]
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        passed=passed,
        registered=registered,
        valid=covered.astype(bool),
        note=note,
    )


def _empty_result(h: int, w: int, note: str) -> RegistrationResult:
    """Result for an empty input: identity, nothing valid."""
    return RegistrationResult(
        warp=np.eye(3),
        method="identity",
        confidence=0.0,
        passed=False,
        registered=np.zeros((h, w, 3), dtype=np.float32),
        valid=np.zeros((h, w), dtype=bool),
        note=note,
    )


def _short(err: Exception) -> str:
    """First line of an OpenCV error, trimmed; OpenCV messages include source paths."""
    text = str(err).strip().splitlines()
    msg = text[-1] if text else type(err).__name__
    return msg.strip()[:120]


def _success_note(method: str, earlier: list[_Attempt]) -> str | None:
    """Audit note when a fallback was needed."""
    if not earlier:
        return None
    tried = "; ".join(f"{a.method}: {a.reason}" for a in earlier)
    return f"Aligned with {method} after the preferred method was rejected ({tried})."


def _failure_note(attempts: list[_Attempt], wl_shape: tuple, af_shape: tuple) -> str:
    """Plain explanation of why the AF image could not be aligned."""
    parts = "; ".join(f"{a.method}: {a.reason}" for a in attempts) or "no method configured"
    note = (
        "The fluorescence image could not be aligned to the white-light photo "
        f"({parts}); it is left unaligned. Retake both photos holding the device steady."
    )
    wl_aspect = wl_shape[1] / max(wl_shape[0], 1)
    af_aspect = af_shape[1] / max(af_shape[0], 1)
    if abs(math.log(max(wl_aspect, 1e-6) / max(af_aspect, 1e-6))) > 0.1:
        note += " The two photos have different aspect ratios (orientation or crop differs)."
    return note
