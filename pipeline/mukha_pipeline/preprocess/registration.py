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
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from mukha_pipeline.types import RegistrationResult

DEFAULTS: dict[str, Any] = {
    "method_order": ["ecc_affine", "orb_homography"],
    "estimation_max_side_px": 512,
    "orb_max_side_px": 1024,
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
    "cross_check": True,
    "cross_check_max_disagreement_fraction": 0.02,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": 8,
    "valid_erode_px": 2,
    "auto_exclude_identical": True,
    "identical_tolerance": 0.05,
    "identical_min_luminance": 0.10,
    "identical_min_area_fraction": 0.0001,
    "identical_margin_fraction": 0.04,
}

METHODS = ("ecc_affine", "ecc_homography", "orb_homography")
MIN_ECC_LEVEL_SIDE_PX = 24  # coarser pyramid levels hold too few pixels for a stable ECC fit
MIN_ESTIMATION_SIDE_PX = 16
# An identity warp was never verified by any method, so its confidence is capped below every
# acceptance threshold; QC can never pass it on confidence alone.
IDENTITY_CONFIDENCE_CAP = 0.2
ORB_EXCLUDE_MARGIN_PX = 16  # ORB describes a 31 px patch; keep keypoints that far from overlays
MIN_KEEP_FRACTION = 0.2  # with less usable area left, an estimate is not worth trusting
DUPLICATE_UPLOAD_FRACTION = 0.5  # identical over this share of the frame = same photo twice
_CV_ERROR = re.compile(r"error: \(-?\d+:([^)]*)\)\s*(.*?)(?:\s*in function.*)?$")


@dataclass
class _Attempt:
    """Outcome of one estimation method."""

    method: str
    warp_est: np.ndarray | None  # 3x3, AF-estimation coords -> WL-estimation coords
    confidence: float
    reason: str | None  # why the estimate was rejected; None = accepted
    residual: np.ndarray | None = None  # 3x3, AF on WL grid -> WL, full resolution


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def register_af_to_wl(
    wl: np.ndarray,
    af: np.ndarray,
    cfg: dict | None,
    *,
    exclude: np.ndarray | None = None,
) -> RegistrationResult:
    """Align ``af`` to ``wl``; ``cfg`` is the ``registration`` section of the analysis config.

    ``exclude`` (optional, bool (H, W) in the WL frame, resized if needed) marks camera-fixed
    pixels to ignore during estimation: anything at the same image position in both captures
    regardless of tissue motion, such as the SYNTHETIC watermark, a burned-in timestamp or the
    clip housing. Such structure would otherwise pull every estimate towards identity. Labels
    burned identically into both photos are also found automatically
    (``auto_exclude_identical``); a duplicate upload therefore fails instead of "aligning".

    Methods run in ``method_order`` (default: ECC affine, then ORB + RANSAC homography) on
    reduced-size copies; the warp is then lifted to the full WL resolution. A method is accepted
    when its warp passes the sanity limits (translation, scale, rotation, no fold) and its
    confidence exceeds ``accept_confidence_min``; the first accepted method wins.

    With ``cross_check`` on, the remaining methods also run, and any confident estimate that
    disagrees with the winner by more than ``cross_check_max_disagreement_fraction`` of the
    image diagonal vetoes it: two independent methods pointing to different alignments mean one
    of them locked onto a repeating structure (e.g. the teeth row), and we cannot tell which.
    When nothing is accepted, identity is returned with ``passed=False`` and a note.
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

    # 2. Pixels that cannot inform alignment: the caller's camera-fixed mask plus labels burned
    #    identically into both photos.
    excluded = _exclusion_mask(exclude, h, w)
    identical = _identical_overlay_mask(wl_f, af_on_grid, p)
    identical_fraction = float(identical.mean()) if identical is not None else 0.0
    if identical is not None:
        excluded = identical if excluded is None else (excluded | identical)
    keep = None if excluded is None else ~excluded

    # 3. Estimate: ECC on gradient structure at a reduced size (speed; gradients at this scale
    #    reflect anatomy rather than sensor noise), ORB at a larger size (keypoints are
    #    pixel-quantised, so corner accuracy scales with resolution). First accepted method
    #    wins, subject to the cross-check.
    est = _Estimator(wl_f, af_on_grid, p, keep)
    attempts: list[_Attempt] = []
    winner: _Attempt | None = None
    for method in _method_order(p):
        attempt = est.run(method)
        attempts.append(attempt)
        if winner is None and attempt.reason is None:
            winner = attempt
            if not p["cross_check"]:
                break

    if winner is not None:
        conflict = _cross_check_conflict(winner, attempts, w, h, p)
        if conflict is None:
            earlier = attempts[: attempts.index(winner)]
            return _build_result(
                af_on_grid,
                winner.residual,  # type: ignore[arg-type]
                to_wl_grid,
                winner.method,
                winner.confidence,
                True,
                _success_note(winner.method, earlier, identical_fraction),
                p,
            )
        winner.reason = conflict

    identity_conf = min(est.identity_confidence(), IDENTITY_CONFIDENCE_CAP)
    return _build_result(
        af_on_grid,
        np.eye(3),
        to_wl_grid,
        "identity",
        identity_conf,
        False,
        _failure_note(attempts, wl_f.shape, af_f.shape, identical_fraction),
        p,
    )


class _Estimator:
    """Lazily built estimation images shared by all methods of one registration call."""

    def __init__(
        self,
        wl_f: np.ndarray,
        af_on_grid: np.ndarray,
        p: dict[str, Any],
        keep: np.ndarray | None,
    ) -> None:
        self.p = p
        self.keep = keep  # bool (H, W) or None: pixels usable for estimation
        self.h, self.w = wl_f.shape[:2]
        self.wl_gray, self.af_gray = _wl_gray(wl_f), _af_gray(af_on_grid)
        tiles = max(1, int(p["clahe_tile_grid"]))
        self.clahe = cv2.createCLAHE(
            clipLimit=float(p["clahe_clip_limit"]), tileGridSize=(tiles, tiles)
        )
        self._frames: dict[tuple[int, int], _EstimationFrame] = {}
        self._structs: tuple[np.ndarray, np.ndarray] | None = None

    def frame(self, max_side: int, margin_px: int) -> _EstimationFrame:
        key = (max_side, margin_px)
        if key not in self._frames:
            self._frames[key] = _EstimationFrame.build(
                self.wl_gray, self.af_gray, max_side, self.clahe, self.keep, margin_px
            )
        return self._frames[key]

    def ecc_frame(self) -> _EstimationFrame:
        # Gaussian + Sobel spread an excluded edge by about 3 sigma + 2 px.
        margin = int(math.ceil(3.0 * float(self.p["ecc_gaussian_sigma"]))) + 3
        return self.frame(int(self.p["estimation_max_side_px"]), margin)

    def orb_frame(self) -> _EstimationFrame:
        return self.frame(int(self.p["orb_max_side_px"]), ORB_EXCLUDE_MARGIN_PX)

    def structures(self) -> tuple[np.ndarray, np.ndarray]:
        """Gradient-structure maps at the ECC size; excluded pixels are zeroed in both."""
        if self._structs is None:
            fr = self.ecc_frame()
            sigma = float(self.p["ecc_gaussian_sigma"])
            t, m = _structure(fr.wl_eq, sigma), _structure(fr.af_eq, sigma)
            if fr.keep is not None:
                t, m = t * (fr.keep > 0), m * (fr.keep > 0)
            self._structs = (t, m)
        return self._structs

    def identity_confidence(self) -> float:
        keep = self.ecc_frame().keep
        return _correlation(*self.structures(), valid=None if keep is None else keep > 0)

    def run(self, method: str) -> _Attempt:
        """Estimate with ``method``, lift to full resolution, and apply the acceptance rules."""
        is_ecc = method.startswith("ecc_")
        frame = self.ecc_frame() if is_ecc else self.orb_frame()
        if min(frame.wl_eq.shape) < MIN_ESTIMATION_SIDE_PX:
            return _Attempt(method, None, 0.0, "image too small to align")
        if frame.keep is not None and float(np.mean(frame.keep > 0)) < MIN_KEEP_FRACTION:
            return _Attempt(method, None, 0.0, "too little of the image left after exclusions")
        if is_ecc:
            attempt = _ecc_attempt(*self.structures(), method, self.p, frame.keep)
        else:
            attempt = _orb_attempt(frame.wl_eq, frame.af_eq, self.p, frame.keep)
        if attempt.warp_est is None:
            return attempt
        to_est = frame.to_est
        attempt.residual = np.linalg.inv(to_est) @ attempt.warp_est @ to_est
        attempt.reason = _sanity_problem(attempt.residual, self.w, self.h, self.p)
        accept_min = float(self.p["accept_confidence_min"])
        if attempt.reason is None and attempt.confidence <= accept_min:
            attempt.reason = f"confidence {attempt.confidence:.2f} at or below {accept_min:.2f}"
        return attempt


def _cross_check_conflict(
    winner: _Attempt, attempts: list[_Attempt], w: int, h: int, p: dict[str, Any]
) -> str | None:
    """Reason to veto ``winner`` when another confident estimate points elsewhere, else None.

    Estimates rejected only by the sanity limits still count: a confident ORB match showing a
    shift beyond the limit is evidence that an in-range ECC answer is a false lock.
    """
    if not p["cross_check"]:
        return None
    limit = float(p["cross_check_max_disagreement_fraction"]) * math.hypot(w, h)
    accept_min = float(p["accept_confidence_min"])
    for other in attempts:
        if other is winner or other.residual is None or other.confidence <= accept_min:
            continue
        gap = _max_displacement_gap(winner.residual, other.residual, w, h)  # type: ignore[arg-type]
        if gap > limit:
            return (
                f"disagrees with {other.method} by {gap:.0f} px (limit {limit:.0f} px); "
                "possible false lock on a repeating structure"
            )
    return None


def _max_displacement_gap(a: np.ndarray, b: np.ndarray, w: int, h: int) -> float:
    """Largest distance between where two warps send a 3x3 grid at 10/50/90 % of the image.

    Inner points rather than corners: they are where the tissue is, and they are less sensitive
    to the extrapolation differences between an affine and a homography fit.
    """
    xs = [0.1 * (w - 1), 0.5 * (w - 1), 0.9 * (w - 1)]
    ys = [0.1 * (h - 1), 0.5 * (h - 1), 0.9 * (h - 1)]
    gap = 0.0
    for x in xs:
        for y in ys:
            ua, va, _ = _apply(a, x, y)
            ub, vb, _ = _apply(b, x, y)
            gap = max(gap, math.hypot(ua - ub, va - vb))
    return gap


# --------------------------------------------------------------------------------------------
# Configuration and image preparation
# --------------------------------------------------------------------------------------------


@dataclass
class _EstimationFrame:
    """CLAHE-equalised WL / AF grays at an estimation size, and the WL -> estimation map."""

    wl_eq: np.ndarray
    af_eq: np.ndarray
    to_est: np.ndarray
    keep: np.ndarray | None = None  # uint8 (255 = usable) at this size, shrunk by a margin

    @classmethod
    def build(
        cls,
        wl_gray: np.ndarray,
        af_gray: np.ndarray,
        max_side: int,
        clahe: cv2.CLAHE,
        keep_full: np.ndarray | None = None,
        margin_px: int = 0,
    ) -> _EstimationFrame:
        h, w = wl_gray.shape
        scale = min(1.0, max(1, max_side) / max(h, w))
        ew, eh = max(1, round(w * scale)), max(1, round(h * scale))
        wl_eq = clahe.apply(_to_u8(_resize(wl_gray, ew, eh)))
        af_eq = clahe.apply(_to_u8(_resize(af_gray, ew, eh)))
        keep = None
        if keep_full is not None:
            small = cv2.resize(
                keep_full.astype(np.uint8) * 255, (ew, eh), interpolation=cv2.INTER_NEAREST
            )
            if margin_px > 0:
                kernel = np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8)
                small = cv2.erode(small, kernel)  # image borders are not eroded
            keep = small
        return cls(wl_eq, af_eq, _resize_matrix(w, h, ew, eh), keep)


def _exclusion_mask(exclude: np.ndarray | None, h: int, w: int) -> np.ndarray | None:
    """Caller's exclusion mask as bool (H, W) at WL size; None when absent, empty or unusable."""
    if exclude is None:
        return None
    ex = np.asarray(exclude)
    if ex.ndim == 3:
        ex = ex.any(axis=2)
    if ex.ndim != 2 or ex.size == 0:
        return None
    ex = np.nan_to_num(ex.astype(np.float32)) > 0
    if ex.shape != (h, w):
        ex = cv2.resize(ex.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return ex if ex.any() else None


def _identical_overlay_mask(
    wl_f: np.ndarray, af_on_grid: np.ndarray, p: dict[str, Any]
) -> np.ndarray | None:
    """Pixels burned identically into both photos (labels, watermarks), dilated; else None.

    Tissue never looks the same under white light and 405 nm excitation (WL mucosa is pink with
    a strong red channel, AF is green with almost no red), so non-dark pixels whose RGB agrees
    within ``identical_tolerance`` in both photos are an overlay that does not move with the
    tissue. Left in, such a label pulls every estimate towards identity: a silent false lock.
    The dilation swallows the label's dark background band. A duplicate upload (the WL photo
    sent as AF) makes almost everything identical, so registration then fails, as it should.
    """
    if not p["auto_exclude_identical"]:
        return None
    h, w = wl_f.shape[:2]
    tol = float(p["identical_tolerance"])
    same = np.all(np.abs(wl_f - af_on_grid) <= tol, axis=2)
    same &= _wl_gray(wl_f) >= float(p["identical_min_luminance"])
    if not same.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(same.astype(np.uint8), 8)
    min_area = max(4, int(float(p["identical_min_area_fraction"]) * h * w))
    big = 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area)
    if big.size == 0:
        return None
    seeds = np.isin(labels, big).astype(np.uint8)
    radius = max(1, int(round(float(p["identical_margin_fraction"]) * min(h, w))))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(seeds, disk) > 0


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


def _ecc_attempt(
    template: np.ndarray,
    moving: np.ndarray,
    method: str,
    p: dict,
    keep: np.ndarray | None = None,
) -> _Attempt:
    """Coarse-to-fine ECC. Returns the AF -> WL warp at estimation resolution."""
    motion = cv2.MOTION_HOMOGRAPHY if method == "ecc_homography" else cv2.MOTION_AFFINE
    try:
        wl_to_af, cc = _ecc_pyramid(template, moving, motion, p, keep)
    except (cv2.error, np.linalg.LinAlgError) as e:
        return _Attempt(method, None, 0.0, f"did not converge ({_short(e)})")
    if not np.all(np.isfinite(wl_to_af)) or abs(np.linalg.det(wl_to_af)) < 1e-9:
        return _Attempt(method, None, 0.0, "degenerate warp")
    conf = float(np.clip(cc, 0.0, 1.0)) if math.isfinite(cc) else 0.0
    return _Attempt(method, np.linalg.inv(wl_to_af), conf, None)


def _ecc_pyramid(
    template: np.ndarray,
    moving: np.ndarray,
    motion: int,
    p: dict,
    keep: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Run ECC from the coarsest to the finest level; returns (W: WL -> AF, final ECC cc).

    ``keep`` (uint8, 255 = usable) is passed as ECC's input mask: excluded AF pixels are dropped
    from the objective after warping, so camera-fixed overlays cannot vote for identity.
    """
    levels = max(1, int(p["ecc_pyramid_levels"]))
    pyr_t, pyr_m, down = [template], [moving], []  # down[k]: level k -> level k+1 coords
    pyr_k = [keep]
    while len(pyr_t) < levels:
        h, w = pyr_t[-1].shape
        nw, nh = w // 2, h // 2
        if min(nw, nh) < MIN_ECC_LEVEL_SIDE_PX:
            break
        down.append(_resize_matrix(w, h, nw, nh))
        pyr_t.append(cv2.resize(pyr_t[-1], (nw, nh), interpolation=cv2.INTER_AREA))
        pyr_m.append(cv2.resize(pyr_m[-1], (nw, nh), interpolation=cv2.INTER_AREA))
        prev = pyr_k[-1]
        pyr_k.append(
            None if prev is None else cv2.resize(prev, (nw, nh), interpolation=cv2.INTER_NEAREST)
        )

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max(1, int(p["ecc_max_iterations"])),
        float(p["ecc_epsilon"]),
    )
    gauss = max(1, int(p["ecc_gauss_filt_size"]))
    gauss = gauss if gauss % 2 == 1 else gauss + 1  # OpenCV requires an odd kernel size
    coarse_keep = None if pyr_k[-1] is None else pyr_k[-1] > 0
    warp = _initial_translation(pyr_t[-1], pyr_m[-1], coarse_keep)
    cc = 0.0
    for level in range(len(pyr_t) - 1, -1, -1):
        if level < len(pyr_t) - 1:
            d = down[level]  # finer (level) -> coarser (level + 1)
            warp = np.linalg.inv(d) @ warp @ d
        init = (warp if motion == cv2.MOTION_HOMOGRAPHY else warp[:2]).astype(np.float32)
        cc, out = cv2.findTransformECC(
            pyr_t[level], pyr_m[level], init, motion, criteria, pyr_k[level], gauss
        )
        warp = np.vstack([out, [0.0, 0.0, 1.0]]) if out.shape[0] == 2 else out
        warp = warp.astype(np.float64)
    return warp, float(cc)


def _initial_translation(
    template: np.ndarray, moving: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
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
    shifted_corr = _correlation(template, moving, shifted, valid=valid)
    if shifted_corr > _correlation(template, moving, identity, valid=valid):
        return shifted
    return identity


def _correlation(
    template: np.ndarray,
    moving: np.ndarray,
    wl_to_af: np.ndarray | None = None,
    *,
    valid: np.ndarray | None = None,
) -> float:
    """Pearson correlation of ``template(x)`` with ``moving(W x)`` over the overlap, in [0, 1].

    ``valid`` (bool, template frame) restricts the overlap further (excluded overlays).
    """
    if wl_to_af is None:
        warped, mask = moving, np.ones(moving.shape, dtype=bool)
    else:
        h, w = template.shape
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        warped = cv2.warpAffine(moving, wl_to_af[:2], (w, h), flags=flags)
        ones = np.ones(moving.shape, dtype=np.uint8)
        mask = (
            cv2.warpAffine(
                ones, wl_to_af[:2], (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
            )
            > 0
        )
    if valid is not None:
        mask = mask & valid
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


def _orb_attempt(
    wl_eq: np.ndarray, af_eq: np.ndarray, p: dict, keep: np.ndarray | None = None
) -> _Attempt:
    """Feature matching on CLAHE grayscale; confidence = inlier ratio x min(1, inliers / N).

    ``keep`` (uint8, 255 = usable) restricts keypoint detection in both images.
    """
    method = "orb_homography"
    try:
        orb = cv2.ORB_create(nfeatures=max(10, int(p["orb_features"])))
        kp_af, des_af = orb.detectAndCompute(af_eq, keep)
        kp_wl, des_wl = orb.detectAndCompute(wl_eq, keep)
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
    src = np.float64([kp_af[m.queryIdx].pt for m in good])
    dst = np.float64([kp_wl[m.trainIdx].pt for m in good])
    homography, inlier_mask = cv2.findHomography(
        src.reshape(-1, 1, 2),
        dst.reshape(-1, 1, 2),
        cv2.RANSAC,
        float(p["ransac_reproj_threshold"]),
    )
    if homography is None or inlier_mask is None or not np.all(np.isfinite(homography)):
        return _Attempt(method, None, 0.0, "no consistent homography")
    inl = inlier_mask.ravel().astype(bool)
    inliers = int(inl.sum())
    full = max(1, int(p["orb_full_confidence_inliers"]))
    conf = (inliers / len(good)) * min(1.0, inliers / full)
    warp = _refine_on_inliers(src[inl], dst[inl], homography.astype(np.float64))
    return _Attempt(method, warp, float(np.clip(conf, 0.0, 1.0)), None)


def _reprojection_errors(warp: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    pts = np.hstack([src, np.ones((len(src), 1))]) @ warp.T
    with np.errstate(divide="ignore", invalid="ignore"):
        proj = pts[:, :2] / pts[:, 2:3]
    err = np.linalg.norm(proj - dst, axis=1)
    return np.where(np.isfinite(err), err, np.inf)


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares affine (as a 3x3) from point pairs; None if under-determined."""
    if len(src) < 3:
        return None
    design = np.hstack([src, np.ones((len(src), 1))])
    if np.linalg.matrix_rank(design) < 3:
        return None
    sol, *_ = np.linalg.lstsq(design, dst, rcond=None)
    out = np.eye(3)
    out[:2] = sol.T
    return out


def _refine_on_inliers(src: np.ndarray, dst: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Refit on the RANSAC inliers and pick the simpler model unless perspective is earned.

    ORB keypoints are quantised to pixel positions at their pyramid octave, and RANSAC keeps
    matches up to the reprojection threshold. A free homography fits that noise with its two
    perspective terms, which swings the image corners by pixels. The tissue is not a plane
    anyway, so after trimming loose inliers we fit both an affine and a homography and keep the
    homography only if it wins on BIC (residual vs parameter count).
    """
    if abs(homography[2, 2]) > 1e-12:
        homography = homography / homography[2, 2]
    err = _reprojection_errors(homography, src, dst)
    keep = err <= max(2.5 * float(np.median(err)), 0.5)
    s, d = src[keep], dst[keep]
    n = len(s)
    affine = _fit_affine(s, d)
    if affine is None or n < 8:
        return homography
    refit, _ = cv2.findHomography(s.reshape(-1, 1, 2), d.reshape(-1, 1, 2), 0)
    if refit is None or not np.all(np.isfinite(refit)) or abs(refit[2, 2]) < 1e-12:
        return affine
    refit = refit / refit[2, 2]
    rss_a = float(np.sum(_reprojection_errors(affine, s, d) ** 2)) + 1e-12
    rss_h = float(np.sum(_reprojection_errors(refit, s, d) ** 2)) + 1e-12
    bic_a = n * math.log(rss_a / n) + 6 * math.log(n)
    bic_h = n * math.log(rss_h / n) + 8 * math.log(n)
    return affine if bic_a <= bic_h else refit


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
        return cv2.warpAffine(
            img, warp[:2], (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    return cv2.warpPerspective(
        img, warp, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


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
    """OpenCV error reduced to its cause (drops version, source path and function name)."""
    text = " ".join(str(err).split())
    m = _CV_ERROR.search(text)
    msg = f"{m.group(1).strip()}: {m.group(2).strip()}".rstrip(": ") if m else text
    return (msg or type(err).__name__)[:120]


def _identical_note(fraction: float) -> str:
    if fraction >= DUPLICATE_UPLOAD_FRACTION:
        return (
            " The two photos are identical over most of the frame; check that the fluorescence "
            "photo, not a second white-light photo, was uploaded."
        )
    return (
        f" Ignored {fraction:.0%} of the frame where both photos are identical "
        "(a label or overlay burned in at the same position)."
    )


def _success_note(
    method: str, earlier: list[_Attempt], identical_fraction: float = 0.0
) -> str | None:
    """Audit note when a fallback was needed or an overlay was ignored; None otherwise."""
    note = ""
    if earlier:
        tried = "; ".join(f"{a.method}: {a.reason}" for a in earlier)
        note = f"Aligned with {method} after the preferred method was rejected ({tried})."
    if identical_fraction > 0:
        note = (note + _identical_note(identical_fraction)).strip()
    return note or None


def _failure_note(
    attempts: list[_Attempt], wl_shape: tuple, af_shape: tuple, identical_fraction: float = 0.0
) -> str:
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
    if identical_fraction > 0:
        note += _identical_note(identical_fraction)
    return note
