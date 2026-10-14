"""Test helpers for io/, capture/, wl/ and spectral/ tests.

SYNTHETIC test data only: a physically motivated mucosa reflectance model, AS7341-like band
integration, a tiny ONNX classifier built in memory, and image-file writers.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image, TiffImagePlugin

from mukha_pipeline.schema.models import (
    AS7341_CHANNELS,
    AS7341_SPECTRAL_CHANNELS,
    CaptureKind,
    CaptureMetadata,
    LedMode,
    QCIssue,
    QCResult,
    QCSeverity,
    SourceAdapter,
)
from mukha_pipeline.sites import OralSite
from mukha_pipeline.types import SpectralCalibration, SpectralFrame, SpectralInput

# --------------------------------------------------------------------------------------------
# Spectral physics (SYNTHETIC)
# --------------------------------------------------------------------------------------------

LAMBDA = np.arange(380.0, 1000.0, 1.0)
CENTRE_NM = {
    "F1_415": 415,
    "F2_445": 445,
    "F3_480": 480,
    "F4_515": 515,
    "F5_555": 555,
    "F6_590": 590,
    "F7_630": 630,
    "F8_680": 680,
    "CLEAR": 600,
    "NIR": 910,
}
FWHM_NM = {
    "F1_415": 26,
    "F2_445": 30,
    "F3_480": 36,
    "F4_515": 39,
    "F5_555": 39,
    "F6_590": 40,
    "F7_630": 50,
    "F8_680": 52,
    "CLEAR": 400,
    "NIR": 60,
}


def _gauss(x: np.ndarray, centre: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def hb_absorption(lam: np.ndarray) -> np.ndarray:
    """Oxy-haemoglobin-like absorption: Soret 415 nm + Q bands 542 / 577 nm, ~0 beyond 620 nm."""
    return 3.0 * _gauss(lam, 415, 12) + 0.5 * _gauss(lam, 542, 14) + 0.55 * _gauss(lam, 577, 10)


def tissue_reflectance(lam: np.ndarray, blood: float) -> np.ndarray:
    """Scattering background rising towards the red, attenuated by blood (Beer-Lambert-like)."""
    scatter = 0.30 + 0.25 * (lam - 400.0) / 300.0
    return np.clip(scatter, 0.05, 0.9) * np.exp(-blood * hb_absorption(lam))


def white_led(lam: np.ndarray) -> np.ndarray:
    """Phosphor white LED: narrow blue pump + broad yellow phosphor."""
    return 1.0 * _gauss(lam, 450, 10) + 0.7 * _gauss(lam, 570, 60) + 0.02


def band_counts(reflectance: np.ndarray, gain: float, integration_ms: float) -> dict[str, float]:
    """Noise-free net counts per channel; counts scale with gain x integration time."""
    led = white_led(LAMBDA)
    out: dict[str, float] = {}
    for c in AS7341_CHANNELS:
        resp = _gauss(LAMBDA, CENTRE_NM[c], FWHM_NM[c] / 2.355)
        out[c] = float(np.sum(reflectance * led * resp)) * gain * integration_ms * 0.05
    return out


def true_band_reflectance(blood: float) -> dict[str, float]:
    """What a perfect white-referenced measurement should return for this tissue."""
    tissue = band_counts(tissue_reflectance(LAMBDA, blood), 1.0, 1.0)
    white = band_counts(np.ones_like(LAMBDA), 1.0, 1.0)
    return {c: tissue[c] / white[c] for c in AS7341_CHANNELS}


def make_frames(
    rng: np.random.Generator,
    reflectance: np.ndarray,
    *,
    n: int,
    gain: float,
    integration_ms: float,
    dark_level: float = 200.0,
    noise: float = 0.003,
) -> tuple[list[SpectralFrame], list[SpectralFrame]]:
    net = band_counts(reflectance, gain, integration_ms)
    lit = []
    for _ in range(n):
        ch = {c: net[c] * (1 + noise * rng.standard_normal()) + dark_level for c in net}
        lit.append(SpectralFrame(ch, gain, integration_ms))
    dark = [
        SpectralFrame({c: dark_level for c in AS7341_CHANNELS}, gain, integration_ms)
        for _ in range(2)
    ]
    return lit, dark


def make_meta(
    site: OralSite, *, device_id: str | None = "confirm-01", calibration_id: str | None = None
) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=str(uuid.uuid4()),
        kind=CaptureKind.SPECTRAL,
        oral_site=site,
        source_adapter=SourceAdapter.SYNTHETIC,
        device_id=device_id,
        led_mode=LedMode.WHITE,
        calibration_id=calibration_id,
        synthetic=True,
    )


def make_reading(
    rng: np.random.Generator,
    reading_id: str,
    site: OralSite,
    role: str,
    blood: float,
    *,
    n: int = 5,
    gain: float = 64.0,
    integration_ms: float = 50.0,
    noise: float = 0.003,
    reference_reading_id: str | None = None,
    device_id: str | None = "confirm-01",
    calibration_id: str | None = None,
) -> SpectralInput:
    lit, dark = make_frames(
        rng,
        tissue_reflectance(LAMBDA, blood),
        n=n,
        gain=gain,
        integration_ms=integration_ms,
        noise=noise,
    )
    return SpectralInput(
        reading_id=reading_id,
        site=site,
        role=role,  # type: ignore[arg-type]
        lit=lit,
        dark=dark,
        meta=make_meta(site, device_id=device_id, calibration_id=calibration_id),
        reference_reading_id=reference_reading_id,
    )


def make_calibration(
    now: datetime,
    *,
    gain: float = 64.0,
    integration_ms: float = 50.0,
    valid_hours: float = 24,
    device_id: str = "confirm-01",
    dark_level: float = 200.0,
) -> SpectralCalibration:
    white = band_counts(np.ones_like(LAMBDA), gain, integration_ms)
    return SpectralCalibration(
        calibration_id="spcal-test",
        device_id=device_id,
        created_at=now,
        expires_at=now + timedelta(hours=valid_hours),
        dark={c: dark_level for c in AS7341_CHANNELS},
        white={c: white[c] + dark_level for c in AS7341_CHANNELS},
        gain=gain,
        integration_ms=integration_ms,
        synthetic=True,
    )


def qc_pass(lit: list[SpectralFrame], dark: list[SpectralFrame], cfg: dict) -> QCResult:
    return QCResult(passed=True, score=1.0)


def qc_fail(lit: list[SpectralFrame], dark: list[SpectralFrame], cfg: dict) -> QCResult:
    return QCResult(
        passed=False,
        score=0.2,
        issues=[
            QCIssue(
                code="spectral_saturated",
                severity=QCSeverity.ERROR,
                message="Probe signal saturated; reduce gain and retake.",
                message_key="qc.spectral_saturated",
            )
        ],
    )


# --------------------------------------------------------------------------------------------
# Spectral files
# --------------------------------------------------------------------------------------------

CSV_HEADER = [
    "reading_id",
    "site",
    "role",
    "kind",
    "repeat",
    *AS7341_CHANNELS,
    "gain",
    "integration_ms",
    "reference_reading_id",
    "lesion_ref",
    "device_id",
]


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.3f}"


def spectral_csv(
    readings: Sequence[SpectralInput],
    calibration: SpectralCalibration | None = None,
    *,
    explicit_refs: bool = True,
    header: Sequence[str] = CSV_HEADER,
) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(header), extrasaction="ignore")
    writer.writeheader()
    for r in readings:
        for kind, frames in (("dark", r.dark), ("lit", r.lit)):
            for i, f in enumerate(frames):
                row = {
                    "reading_id": r.reading_id,
                    "site": r.site.value,
                    "role": r.role,
                    "kind": kind,
                    "repeat": i,
                    "gain": _fmt(f.gain),
                    "integration_ms": _fmt(f.integration_ms),
                    "reference_reading_id": (r.reference_reading_id or "") if explicit_refs else "",
                    "lesion_ref": r.lesion_ref or "",
                    "device_id": r.meta.device_id or "",
                }
                row.update({c: _fmt(f.channels.get(c)) for c in AS7341_CHANNELS})
                writer.writerow(row)
    if calibration is not None:
        for kind, values in (("white", calibration.white), ("white_dark", calibration.dark)):
            for i in range(2):
                row = {
                    "reading_id": "calibration",
                    "kind": kind,
                    "repeat": i,
                    "gain": _fmt(calibration.gain),
                    "integration_ms": _fmt(calibration.integration_ms),
                }
                row.update({c: _fmt(values.get(c)) for c in AS7341_CHANNELS})
                writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def _frame_json(f: SpectralFrame) -> dict:
    return {c: round(v, 3) for c, v in f.channels.items()}


def spectral_json(
    readings: Sequence[SpectralInput],
    calibration: SpectralCalibration | None = None,
    *,
    device_id: str | None = "confirm-01",
    firmware_version: str | None = "0.1.0",
    extra: dict | None = None,
) -> bytes:
    doc: dict = {"format": "mukhaspectra.spectral.v1"}
    if device_id:
        doc["device_id"] = device_id
    if firmware_version:
        doc["firmware_version"] = firmware_version
    if calibration is not None:
        doc["white_reference"] = {
            "gain": calibration.gain,
            "integration_ms": calibration.integration_ms,
            "white": [dict(calibration.white), dict(calibration.white)],
            "white_dark": [dict(calibration.dark)],
        }
    doc["readings"] = [
        {
            "reading_id": r.reading_id,
            "site": r.site.value,
            "role": r.role,
            "reference_reading_id": r.reference_reading_id,
            "lesion_ref": r.lesion_ref,
            "gain": r.lit[0].gain,
            "integration_ms": r.lit[0].integration_ms,
            "dark": [_frame_json(f) for f in r.dark],
            "lit": [_frame_json(f) for f in r.lit],
        }
        for r in readings
    ]
    doc.update(extra or {})
    return json.dumps(doc).encode("utf-8")


# --------------------------------------------------------------------------------------------
# Tiny ONNX classifier
# --------------------------------------------------------------------------------------------

# Logit weights per (R, G, B) channel mean after ImageNet normalisation, for the 5 labels
# normal, white_patch, red_patch, ulcer_mass, other. "red_patch" responds to red minus green.
DEFAULT_WEIGHTS = np.array(
    [
        [0.0, 1.0, 3.0, -1.0, 0.0],
        [0.0, 1.0, -3.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def build_tiny_onnx(
    path: Path,
    *,
    size: int | str = 32,
    softmax: bool = True,
    n_out: int = 5,
    channels: int = 3,
    weights: np.ndarray | None = None,
) -> Path:
    """input (1, C, S, S) -> GlobalAveragePool -> Flatten -> Gemm(C -> n_out) [-> Softmax]."""
    w = DEFAULT_WEIGHTS[:channels, :n_out] if weights is None else weights.astype(np.float32)
    if w.shape != (channels, n_out):
        w = np.zeros((channels, n_out), dtype=np.float32)
    b = np.zeros((n_out,), dtype=np.float32)
    x = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, channels, size, size])
    out_name = "probs" if softmax else "logits"
    y = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, n_out])
    nodes = [
        helper.make_node("GlobalAveragePool", ["image"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "W", "B"], ["logits"]),
    ]
    if softmax:
        nodes.append(helper.make_node("Softmax", ["logits"], ["probs"], axis=1))
    graph = helper.make_graph(
        nodes,
        "tiny_wl_classifier",
        [x],
        [y],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path = Path(path)
    onnx.save(model, str(path))
    return path


MODEL_CARD = """---
name: wl-tiny-test-model
version: 0.0.1-test
validated: true
---
# Model card (TEST ONLY)
Tiny synthetic model used in unit tests. Not trained on any patient data.
"""


# --------------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------------


def jpeg_with_exif(
    size: tuple[int, int] = (120, 80),
    *,
    orientation: int | None = None,
    exposure: tuple[int, int] | None = (1, 125),
    iso: int | None = 200,
    white_balance: int | None = 1,
    datetime_original: str | None = "2026:10:14 09:15:00",
    offset: str | None = "+05:30",
    make: str | None = "MukhaTest",
    model: str | None = "Scout-1",
    color: tuple[int, int, int] = (200, 80, 90),
) -> bytes:
    img = Image.new("RGB", size, color)
    exif = Image.Exif()
    if make:
        exif[271] = make
    if model:
        exif[272] = model
    if orientation:
        exif[274] = orientation
    sub = exif.get_ifd(0x8769)
    if exposure:
        sub[33434] = TiffImagePlugin.IFDRational(*exposure)
    if iso is not None:
        sub[34855] = iso
    if white_balance is not None:
        sub[41987] = white_balance
    if datetime_original:
        sub[36867] = datetime_original
    if offset:
        sub[36881] = offset
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif, quality=95)
    return buf.getvalue()


def png_bytes(img: Image.Image, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", **kwargs)
    return buf.getvalue()


def image_bytes(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


def dng_tagged_tiff(size: tuple[int, int] = (80, 80)) -> bytes:
    """A plain TIFF carrying the DNGVersion tag (not a decodable raw)."""
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[50706] = bytes([1, 4, 0, 0])
    info.tagtype[50706] = 1
    return image_bytes(Image.new("RGB", size), "TIFF", tiffinfo=info)


def spectral_channel_names() -> tuple[str, ...]:
    return AS7341_SPECTRAL_CHANNELS
