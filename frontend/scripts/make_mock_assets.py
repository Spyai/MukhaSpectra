"""Generate SYNTHETIC artifact images for the frontend mock mode (src/mocks/assets/).

    ../.venv/bin/python scripts/make_mock_assets.py        # from frontend/

Everything written here is synthetic: every image carries a visible "SYNTHETIC" stamp and the
file names start with "synthetic_". The pictures only need to look plausible enough to exercise
the UI (WL/AF wipe, heatmap overlay, region markers, 3D terrain). They are not from a patient
and must never be used to support any performance claim.

Outputs (deterministic, seeded):
  synthetic_<case>_wl.jpg            white-light photo of pink, textured mucosa
  synthetic_<case>_af.jpg            405 nm autofluorescence photo (raw, slightly misaligned)
  synthetic_<case>_af_registered.jpg AF warped into the WL frame
  synthetic_<case>_heatmap.jpg       AF-loss heatmap composited on the WL photo
  synthetic_<case>_grid.json         96 x 72 relative-AF grid {width,height,values,min,max,threshold}
  synthetic_mock_report.pdf          one-page placeholder PDF for mock mode
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "src" / "mocks" / "assets"
W, H = 640, 480
GRID_W, GRID_H = 96, 72
FVL_THRESHOLD = 0.75


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def smooth_noise(rng: np.random.Generator, scale: float, shape=(H, W)) -> np.ndarray:
    """Band-limited noise in [-1, 1] with a feature size of roughly `scale` pixels."""
    h, w = shape
    sh, sw = max(2, int(h / scale)), max(2, int(w / scale))
    base = rng.standard_normal((sh, sw)).astype(np.float32)
    up = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), scale / 3)
    up /= np.abs(up).max() + 1e-6
    return up


def blob_mask(rng: np.random.Generator, cx: float, cy: float, r: float, wobble: float,
              shape=(H, W)) -> np.ndarray:
    """Irregular soft blob (float 0..1) centred at (cx, cy) in pixels."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.hypot(xx - cx, yy - cy)
    radius = np.full_like(ang, r)
    for k, amp in ((3, wobble), (5, wobble * 0.6), (8, wobble * 0.35)):
        phase = rng.uniform(0, 2 * math.pi)
        radius += r * amp * np.sin(k * ang + phase)
    m = np.clip((radius - rad) / (r * 0.18), 0, 1)
    return cv2.GaussianBlur(m, (0, 0), 3)


def tissue_mask(kind: str) -> np.ndarray:
    """Soft mask of where mucosa is in the frame (irregular edge, fills most of the view)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rng = np.random.default_rng({"tongue": 101, "buccal": 102}.get(kind, 103))
    if kind == "tongue":
        # Lateral tongue: a long rounded shape filling most of the frame.
        e = ((xx - W * 0.52) / (W * 0.58)) ** 2 + ((yy - H * 0.56) / (H * 0.47)) ** 2
    elif kind == "buccal":
        e = ((xx - W * 0.5) / (W * 0.56)) ** 2 + ((yy - H * 0.5) / (H * 0.56)) ** 2
    else:  # gingiva: a band
        e = ((xx - W * 0.5) / (W * 0.62)) ** 2 + ((yy - H * 0.5) / (H * 0.30)) ** 2
    e = e + 0.18 * smooth_noise(rng, 120)
    return np.clip((1.0 - e) * 7, 0, 1)


def shading(rng: np.random.Generator) -> np.ndarray:
    """Lambert-ish shading of a gently bumpy, moist surface lit from the upper left."""
    height = smooth_noise(rng, 70) * 1.0 + smooth_noise(rng, 18) * 0.35
    gy, gx = np.gradient(height * 40)
    nz = 1 / np.sqrt(gx ** 2 + gy ** 2 + 1)
    light = np.array([-0.45, -0.55, 0.70])
    light /= np.linalg.norm(light)
    lam = (-gx * light[0] - gy * light[1] + light[2]) * nz
    return np.clip(lam, 0, 1).astype(np.float32)


def stamp(img: np.ndarray) -> np.ndarray:
    """Visible SYNTHETIC stamp: bottom-left label plus a faint diagonal watermark."""
    out = img.copy()
    overlay = out.copy()
    # Bottom-left, like the pipeline's own watermark, so UI labels at the top stay clear.
    cv2.rectangle(overlay, (10, H - 44), (196, H - 10), (20, 16, 24), -1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    cv2.putText(out, "SYNTHETIC", (20, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                cv2.LINE_AA)
    wm = np.zeros_like(out)
    cv2.putText(wm, "SYNTHETIC", (int(W * 0.12), int(H * 0.62)), cv2.FONT_HERSHEY_DUPLEX, 2.6,
                (255, 255, 255), 4, cv2.LINE_AA)
    rot = cv2.getRotationMatrix2D((W / 2, H / 2), 24, 1.0)
    wm = cv2.warpAffine(wm, rot, (W, H))
    out = cv2.addWeighted(out, 1.0, wm, 0.10, 0)
    return out


def to_bgr8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def save_jpg(name: str, rgb: np.ndarray, quality: int = 84) -> None:
    img = stamp(to_bgr8(rgb))
    cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, quality,
                                        cv2.IMWRITE_JPEG_OPTIMIZE, 1])


# ---------------------------------------------------------------------------------------------
# image synthesis
# ---------------------------------------------------------------------------------------------


def white_light(rng: np.random.Generator, kind: str, lesion: np.ndarray | None,
                lesion_color=(0.86, 0.84, 0.80), blur: float = 0.0, glare: float = 0.0
                ) -> np.ndarray:
    tissue = tissue_mask(kind)
    low = smooth_noise(rng, 90)
    mid = smooth_noise(rng, 22)
    fine = smooth_noise(rng, 4)
    base = np.array([0.80, 0.47, 0.50], np.float32)
    if kind == "gingiva":
        base = np.array([0.84, 0.55, 0.56], np.float32)
    img = np.empty((H, W, 3), np.float32)
    for c in range(3):
        img[..., c] = base[c] + 0.06 * low + 0.035 * mid + 0.02 * fine
    # Vessels: thin dark-red meanders.
    veins = np.zeros((H, W), np.float32)
    for _ in range(9):
        pts = []
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        ang = rng.uniform(0, 2 * math.pi)
        for _ in range(40):
            ang += rng.normal(0, 0.35)
            x += 9 * math.cos(ang)
            y += 9 * math.sin(ang)
            pts.append((int(x), int(y)))
        cv2.polylines(veins, [np.array(pts, np.int32)], False, 1.0, 1, cv2.LINE_AA)
    veins = cv2.GaussianBlur(veins, (0, 0), 1.2)
    img[..., 0] -= 0.10 * veins
    img[..., 1] -= 0.16 * veins
    img[..., 2] -= 0.12 * veins
    # Papillae speckle on tongue.
    if kind == "tongue":
        spk = (rng.random((H, W)) > 0.985).astype(np.float32)
        spk = cv2.GaussianBlur(spk, (0, 0), 1.4) * 2.0
        img += 0.05 * spk[..., None]
    if lesion is not None:
        lc = np.array(lesion_color, np.float32)
        tex = 0.04 * smooth_noise(rng, 6)
        for c in range(3):
            img[..., c] = img[..., c] * (1 - lesion * 0.85) + (lc[c] + tex) * lesion * 0.85
    shade = shading(rng)
    img *= (0.78 + 0.32 * shade)[..., None]
    # Background: dark oral cavity / shadow.
    bg = np.array([0.10, 0.04, 0.05], np.float32)
    img = img * tissue[..., None] + bg * (1 - tissue[..., None])
    # Light falloff.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - 0.35 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2
    img *= vig[..., None]
    # Specular glints.
    n_glints = 7 if glare == 0 else 7 + int(glare * 10)
    g = np.zeros((H, W), np.float32)
    for _ in range(n_glints):
        cx, cy = rng.uniform(W * 0.15, W * 0.85), rng.uniform(H * 0.2, H * 0.8)
        size = rng.uniform(2, 6) * (1 + glare * 2.2)
        cv2.ellipse(g, (int(cx), int(cy)), (int(size * 2), int(size)), rng.uniform(0, 180), 0,
                    360, 1.0, -1)
    g = cv2.GaussianBlur(g, (0, 0), 2) * tissue
    img = img * (1 - g[..., None]) + g[..., None] * 0.98
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return np.clip(img, 0, 1)


def autofluorescence(rng: np.random.Generator, kind: str, fvl: np.ndarray | None,
                     depth: float = 0.55, porphyrin: int = 0, dim: float = 1.0) -> np.ndarray:
    tissue = tissue_mask(kind)
    low = smooth_noise(rng, 70)
    mid = smooth_noise(rng, 14)
    green = (0.47 + 0.07 * low + 0.05 * mid + 0.02 * smooth_noise(rng, 4)) * dim
    green *= 0.82 + 0.25 * shading(rng)
    if kind in ("gingiva",):
        green *= 0.8
    if fvl is not None:
        green = green * (1 - depth * fvl)
    img = np.zeros((H, W, 3), np.float32)
    img[..., 1] = green
    img[..., 0] = 0.16 * green + 0.02
    img[..., 2] = 0.22 * green + 0.05
    # Red porphyrin fluorescence from plaque/bacteria (~635 nm).
    for _ in range(porphyrin):
        cx, cy = rng.uniform(W * 0.25, W * 0.75), rng.uniform(H * 0.3, H * 0.7)
        spot = blob_mask(rng, cx, cy, rng.uniform(10, 22), 0.25)
        img[..., 0] += 0.62 * spot * dim
        img[..., 1] -= 0.18 * spot * dim
        img[..., 2] += 0.04 * spot * dim
    bg = np.array([0.03, 0.03, 0.09], np.float32)
    img = img * tissue[..., None] + bg * (1 - tissue[..., None])
    # Glow bloom.
    bloom = cv2.GaussianBlur(img, (0, 0), 14)
    img = img + 0.18 * bloom
    img += rng.normal(0, 0.012, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def misalign(img: np.ndarray, dx: float, dy: float, rot: float, scale: float) -> np.ndarray:
    m = cv2.getRotationMatrix2D((W / 2, H / 2), rot, scale)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REFLECT)


def relative_af(kind: str, fvl: np.ndarray | None, depth: float, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray]:
    tissue = tissue_mask(kind)
    rel = 1.0 + 0.05 * smooth_noise(rng, 60) + 0.02 * smooth_noise(rng, 12)
    if fvl is not None:
        rel = rel * (1 - depth * fvl)
    rel = cv2.GaussianBlur(rel.astype(np.float32), (0, 0), 3)
    return rel, tissue


def heatmap(wl: np.ndarray, rel: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    loss = np.clip((FVL_THRESHOLD - rel) / FVL_THRESHOLD, 0, 1) / 0.6
    loss = np.clip(loss, 0, 1) * (tissue > 0.5)
    gray = cv2.cvtColor((wl * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    base = np.stack([gray] * 3, -1) * 0.85
    cmap = cv2.applyColorMap((loss * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    alpha = np.clip(loss * 1.4, 0, 0.75)[..., None]
    out = base * (1 - alpha) + cmap * alpha
    # Region outline at the FVL threshold.
    region = ((rel < FVL_THRESHOLD) & (tissue > 0.5)).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out8 = (np.clip(out, 0, 1) * 255).astype(np.uint8).copy()
    cv2.drawContours(out8, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    return out8.astype(np.float32) / 255


def grid_json(rel: np.ndarray, tissue: np.ndarray) -> dict:
    small = cv2.resize(rel, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
    tmask = cv2.resize(tissue, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
    values: list[list[float | None]] = []
    vmin, vmax = math.inf, -math.inf
    for y in range(GRID_H):
        row: list[float | None] = []
        for x in range(GRID_W):
            if tmask[y, x] < 0.5:
                row.append(None)
            else:
                v = round(float(small[y, x]), 4)
                vmin, vmax = min(vmin, v), max(vmax, v)
                row.append(v)
        values.append(row)
    return {
        "width": GRID_W,
        "height": GRID_H,
        "values": values,
        "min": round(vmin, 4),
        "max": round(vmax, 4),
        "threshold": FVL_THRESHOLD,
        "synthetic": True,
    }


def region_stats(rel: np.ndarray, tissue: np.ndarray) -> list[dict]:
    """Print helper: centroid/bbox of FVL regions in normalised coordinates (for fixtures)."""
    region = ((rel < FVL_THRESHOLD) & (tissue > 0.5)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(region)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 150:
            continue
        vals = rel[labels == i]
        out.append({
            "area_px": int(area),
            "centroid": {"x": round(cents[i][0] / W, 3), "y": round(cents[i][1] / H, 3)},
            "bbox": {"x": round(x / W, 3), "y": round(y / H, 3), "w": round(w / W, 3),
                     "h": round(h / H, 3)},
            "mean_rel": round(float(vals.mean()), 3),
            "p5_rel": round(float(np.percentile(vals, 5)), 3),
        })
    return out


# ---------------------------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------------------------


def make_case(name: str, kind: str, seed: int, *, lesion: tuple | None = None, depth=0.55,
              porphyrin=0, lesion_wl_color=(0.86, 0.84, 0.80), wl_lesion_strength=1.0,
              blur=0.0, glare=0.0, af_dim=1.0, derived=True) -> None:
    rng = np.random.default_rng(seed)
    fvl = None
    if lesion is not None:
        cx, cy, r, wob = lesion
        fvl = blob_mask(rng, cx * W, cy * H, r * W, wob)
    wl_lesion = None if fvl is None or wl_lesion_strength == 0 else fvl * wl_lesion_strength
    wl = white_light(rng, kind, wl_lesion, lesion_wl_color, blur=blur, glare=glare)
    af_reg = autofluorescence(rng, kind, fvl, depth=depth, porphyrin=porphyrin, dim=af_dim)
    af_raw = misalign(af_reg, dx=9, dy=-6, rot=2.0, scale=1.03)
    save_jpg(f"synthetic_{name}_wl.jpg", wl)
    save_jpg(f"synthetic_{name}_af.jpg", af_raw)
    if not derived:
        return
    save_jpg(f"synthetic_{name}_af_registered.jpg", af_reg)
    rel, tissue = relative_af(kind, fvl, depth, rng)
    save_jpg(f"synthetic_{name}_heatmap.jpg", heatmap(wl, rel, tissue))
    (OUT / f"synthetic_{name}_grid.json").write_text(json.dumps(grid_json(rel, tissue),
                                                                separators=(",", ":")))
    print(name, json.dumps(region_stats(rel, tissue)))


def minimal_pdf(path: Path, lines: list[str]) -> None:
    """Write a one-page PDF by hand (no extra dependencies)."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_lines = ["BT", "/F1 22 Tf", "56 760 Td", f"({esc(lines[0])}) Tj", "/F1 11 Tf"]
    for line in lines[1:]:
        content_lines += ["0 -20 Td", f"({esc(line)}) Tj"]
    content_lines += ["ET",
                      # diagonal SYNTHETIC watermark
                      "q 0.85 g BT /F1 64 Tf 0.819 0.574 -0.574 0.819 120 260 Tm (SYNTHETIC DATA) Tj"
                      " ET Q"]
    stream = "\n".join(content_lines).encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # HIGH case: right lateral tongue with a white patch and a deep FVL basin.
    make_case("tongue_r", "tongue", 11, lesion=(0.43, 0.56, 0.11, 0.22), depth=0.6,
              lesion_wl_color=(0.88, 0.85, 0.82), wl_lesion_strength=0.9)
    # Contralateral reference: healthy left lateral tongue.
    make_case("tongue_l", "tongue", 12, derived=False)
    # Left buccal mucosa: shallower FVL, overlapped by red porphyrin spots (plaque).
    make_case("buccal_l", "buccal", 13, lesion=(0.60, 0.46, 0.075, 0.3), depth=0.38,
              porphyrin=3, wl_lesion_strength=0.0)
    # Lower gingiva: blurred, glare -> fails QC.
    make_case("gingiva", "gingiva", 14, blur=6.0, glare=1.0, af_dim=0.25, derived=False)
    # LOW case: healthy right buccal mucosa (+ healthy left as reference).
    make_case("buccal_r_healthy", "buccal", 21)
    make_case("buccal_l_healthy", "buccal", 22, derived=False)
    minimal_pdf(OUT / "synthetic_mock_report.pdf", [
        "MukhaSpectra - mock report (SYNTHETIC)",
        "Research prototype - not for clinical use.",
        "This placeholder PDF is served by the frontend mock mode only.",
        "The real clinician report is rendered by the backend report engine",
        "(GET /api/v1/screenings/{id}/report.pdf).",
        "Screening aid - not a diagnosis.",
    ])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
