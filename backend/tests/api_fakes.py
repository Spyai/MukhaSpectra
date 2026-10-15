"""Fakes for API tests: a pipeline bridge, a report renderer and upload builders.

The fake analyser returns a fully VALID ``ScreeningResult`` built from the pipeline schema
models, so the API is tested against the real contract while the analysis itself is stubbed.
(Module name is unique on purpose: backend/tests is on pytest's pythonpath.)
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from app.errors import InputDecodeError
from app.services.pipeline_bridge import PipelineBridge
from mukha_pipeline.capture.upload import is_synthetic_file_name
from mukha_pipeline.config import PipelineConfig
from mukha_pipeline.schema.models import (
    AS7341_CHANNELS,
    SYNTHETIC_NOTICE,
    ActionCode,
    CalibrationInfo,
    CalibrationStatus,
    CaptureKind,
    CaptureMetadata,
    CaptureRecord,
    ConsentInfo,
    ContributingFactor,
    DataProvenance,
    FactorSource,
    LesionAssessment,
    LesionOrigin,
    Limitation,
    ModelComponent,
    ModelInfo,
    ModelStatus,
    Notices,
    PatientInfo,
    ProcessingInfo,
    QCResult,
    QCSummary,
    RecommendedAction,
    ReferralUrgency,
    RiskFactors,
    RiskLevel,
    ScoreThresholds,
    ScreeningContext,
    ScreeningResult,
    SiteArtifacts,
    SiteResult,
    SiteRole,
    SourceAdapter,
    TriageResult,
    VersionInfo,
    VisualAppearance,
    VisualFlag,
)
from mukha_pipeline.sites import OralSite, site_label
from mukha_pipeline.types import (
    AnalysisOutput,
    ImageCalibration,
    ImageCapture,
    ScreeningInput,
    SiteInput,
    SpectralCalibration,
    SpectralFrame,
    SpectralInput,
)

FIXED_NOW = datetime(2026, 10, 14, 9, 30, tzinfo=UTC)
PATIENT_NAME = "Asha Ramesh Testpatient"
PATIENT_EXTERNAL_ID = "HOSP-EXT-424242"
PII_MARKERS = ("testpatient", "ramesh", PATIENT_EXTERNAL_ID.lower())  # compare lower-case

# --------------------------------------------------------------------------------------------
# Bytes builders
# --------------------------------------------------------------------------------------------


def png_bytes(
    color: tuple[int, int, int] = (190, 90, 90), size: tuple[int, int] = (24, 16)
) -> bytes:
    """A tiny RGB PNG (width, height)."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    arr = np.zeros((height, width, 3), np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = (color[1] + xx * 3) % 256
    arr[..., 2] = (color[2] + yy * 5) % 256
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _channel_row(base: float) -> dict[str, float]:
    return {c: round(base * (1.0 + 0.1 * i), 3) for i, c in enumerate(AS7341_CHANNELS)}


def spectral_csv_bytes() -> bytes:
    """A small, valid spectral file v1 (CSV): one lesion + one contralateral reference."""
    header = [
        "reading_id",
        "site",
        "role",
        "kind",
        "repeat",
        *AS7341_CHANNELS,
        "gain",
        "integration_ms",
        "reference_reading_id",
        "device_id",
    ]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    for rid, site, role, ref, base in (
        ("s1", "buccal_mucosa_left", "lesion", "s2", 900.0),
        ("s2", "buccal_mucosa_right", "reference", "", 1000.0),
    ):
        for kind, level in (("dark", 0.05), ("lit", 1.0)):
            for repeat in range(3):
                row = {
                    "reading_id": rid,
                    "site": site,
                    "role": role,
                    "kind": kind,
                    "repeat": repeat,
                    "gain": 64,
                    "integration_ms": 50,
                    "reference_reading_id": ref,
                    "device_id": "confirm-01",
                }
                row.update(_channel_row(base * level + repeat))
                writer.writerow(row)
    return out.getvalue().encode("utf-8")


def default_manifest(**overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "patient": {
            "name": PATIENT_NAME,
            "external_id": PATIENT_EXTERNAL_ID,
            "age_years": 52,
            "sex": "female",
        },
        "consent": {
            "consent_recorded": True,
            "method": "checkbox",
            "language": "en",
            "research_use_opt_in": False,
        },
        "risk_factors": {"smokeless_tobacco": {"uses": True, "types": ["gutka"], "years": 12}},
        "context": {"screening_site_name": "PHC Hoskote camp", "health_worker_id": "hw-17"},
        "data_provenance": "real",
        "sites": [
            {
                "site": "buccal_mucosa_left",
                "wl": "wl_bl",
                "af": "af_bl",
                "dark": "dark_bl",
                "visual_flag": {"lesion_seen": True, "appearance": ["white_patch"]},
                "diascopy": "not_done",
                "px_per_mm": None,
                "device_id": "scout-01",
                "firmware_version": "0.1.0",
            },
            {"site": "buccal_mucosa_right", "role": "reference_only", "wl": "wl_br", "af": "af_br"},
        ],
        "spectral_file": "spectrum",
    }
    manifest.update(overrides)
    return manifest


def default_files() -> dict[str, tuple[str, bytes, str]]:
    """Multipart files. Uploaded names deliberately contain the patient's name."""
    return {
        "wl_bl": ("asha_ramesh_wl_left.png", png_bytes((200, 90, 90)), "image/png"),
        "af_bl": ("asha_ramesh_af_left.png", png_bytes((40, 160, 40)), "image/png"),
        "dark_bl": ("asha_ramesh_dark_left.png", png_bytes((2, 2, 2)), "image/png"),
        "wl_br": ("asha_ramesh_wl_right.png", png_bytes((205, 95, 95)), "image/png"),
        "af_br": ("asha_ramesh_af_right.png", png_bytes((45, 170, 45)), "image/png"),
        "spectrum": ("asha_ramesh_spectrum.csv", spectral_csv_bytes(), "text/csv"),
    }


def upload(
    client: Any, manifest: dict[str, Any] | None = None, files: dict[str, Any] | None = None
) -> Any:
    return client.post(
        "/api/v1/screenings",
        data={"manifest": json.dumps(manifest if manifest is not None else default_manifest())},
        files=files if files is not None else default_files(),
    )


# --------------------------------------------------------------------------------------------
# Fake pipeline
# --------------------------------------------------------------------------------------------


def decode_png(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    except Exception as exc:  # noqa: BLE001
        raise InputDecodeError("The image could not be read. Please retake the photo.") from exc


class FakeUploadAdapter:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls

    def _capture(
        self,
        data: bytes,
        kind: CaptureKind,
        site: OralSite,
        name: str | None,
        device_id: str | None,
        firmware: str | None,
        synthetic: bool,
    ) -> ImageCapture:
        image = decode_png(data)
        meta = CaptureMetadata(
            capture_id=str(uuid.uuid4()),
            kind=kind,
            oral_site=site,
            source_adapter=SourceAdapter.UPLOAD,
            device_id=device_id,
            firmware_version=firmware,
            exposure_time_ms=33.0,
            iso=100.0,
            width_px=image.shape[1],
            height_px=image.shape[0],
            file_name=name,
            file_sha256=hashlib.sha256(data).hexdigest(),
            mime_type="image/png",
            synthetic=synthetic,
        )
        return ImageCapture(image=image, meta=meta)

    def site_input(
        self,
        site: Any,
        wl: bytes,
        af: bytes,
        dark: bytes | None = None,
        *,
        filenames: dict[str, str | None],
        source_filenames: dict[str, str | None] | None = None,
        role: Any,
        visual_flag: Any,
        diascopy: Any,
        px_per_mm: float | None,
        device_id: str | None,
        firmware_version: str | None,
        synthetic: bool = False,
    ) -> SiteInput:
        site = OralSite(site)
        sources = dict(source_filenames or {})
        self.calls.append(
            {
                "site": site,
                "filenames": dict(filenames),
                "source_filenames": sources,
                "synthetic": synthetic,
            }
        )

        def flagged(slot: str) -> bool:  # same rule as the real UploadAdapter
            return synthetic or is_synthetic_file_name(sources.get(slot))

        def args(slot: str) -> tuple[str | None, str | None, bool]:
            return (device_id, firmware_version, flagged(slot))

        return SiteInput(
            site=site,
            wl=self._capture(wl, CaptureKind.WHITE_LIGHT, site, filenames.get("wl"), *args("wl")),
            af=self._capture(
                af, CaptureKind.AUTOFLUORESCENCE, site, filenames.get("af"), *args("af")
            ),
            dark=(
                self._capture(
                    dark, CaptureKind.DARK_FRAME, site, filenames.get("dark"), *args("dark")
                )
                if dark
                else None
            ),
            role=SiteRole(role),
            visual_flag=visual_flag,
            diascopy=diascopy,
            px_per_mm=px_per_mm,
        )


def _spectral_meta(
    rid: str, site: OralSite, source: SourceAdapter, device_id: str | None, synthetic: bool
) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=f"spec-{rid}",
        kind=CaptureKind.SPECTRAL,
        oral_site=site,
        source_adapter=source,
        device_id=device_id,
        synthetic=synthetic,
    )


def fake_parse_spectral(
    data: bytes,
    filename: str,
    *,
    source_adapter: SourceAdapter,
    device_id: str | None,
    synthetic: bool,
    now: datetime | None,
) -> Any:
    """Minimal spectral v1 reader (CSV + JSON) with the real parser's call signature."""
    text = data.decode("utf-8", errors="replace")
    groups: dict[str, dict[str, Any]] = {}
    if filename.endswith(".json"):
        doc = json.loads(text)
        device_id = doc.get("device_id") or device_id
        for r in doc["readings"]:
            groups[r["reading_id"]] = {
                "site": r["site"],
                "role": r["role"],
                "ref": r.get("reference_reading_id"),
                "lesion_ref": r.get("lesion_ref"),
                "lit": [
                    SpectralFrame(
                        {k: v for k, v in f.items() if k in AS7341_CHANNELS},
                        f.get("gain"),
                        f.get("integration_ms"),
                    )
                    for f in r["lit"]
                ],
                "dark": [
                    SpectralFrame(
                        {k: v for k, v in f.items() if k in AS7341_CHANNELS},
                        f.get("gain"),
                        f.get("integration_ms"),
                    )
                    for f in r["dark"]
                ],
            }
    else:
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows or "reading_id" not in rows[0] or "kind" not in rows[0]:
            raise InputDecodeError("The spectral file has no 'reading_id' and 'kind' columns.")
        for row in rows:
            g = groups.setdefault(
                row["reading_id"],
                {
                    "site": row["site"],
                    "role": row["role"],
                    "lit": [],
                    "dark": [],
                    "ref": row.get("reference_reading_id") or None,
                    "lesion_ref": None,
                },
            )
            frame = SpectralFrame(
                {c: float(row[c]) for c in AS7341_CHANNELS},
                float(row["gain"]),
                float(row["integration_ms"]),
            )
            g[row["kind"]].append(frame)
            device_id = device_id or row.get("device_id")
    readings = [
        SpectralInput(
            reading_id=rid,
            site=OralSite(g["site"]),
            role=g["role"],
            lit=g["lit"],
            dark=g["dark"],
            meta=_spectral_meta(rid, OralSite(g["site"]), source_adapter, device_id, synthetic),
            reference_reading_id=g["ref"],
            lesion_ref=g["lesion_ref"],
        )
        for rid, g in groups.items()
    ]
    return SimpleNamespace(readings=readings, calibration=None, notes=[])


class FakeClassifier:
    info = ModelInfo(
        component=ModelComponent.WL_LESION_CLASSIFIER,
        name="fake-wl",
        version="0",
        status=ModelStatus.MODEL_UNAVAILABLE,
    )


def build_result(
    inp: ScreeningInput, classifier: Any = None, software_version: str | None = None
) -> ScreeningResult:
    """A valid ScreeningResult derived from the input (no image analysis).

    Like the orchestrator, any capture marked synthetic makes the result synthetic.
    """
    cfg = PipelineConfig.default()
    lesion_sites = [s for s in inp.sites if s.visual_flag.lesion_seen]
    high = bool(lesion_sites)
    level = RiskLevel.HIGH if high else RiskLevel.LOW
    action = RecommendedAction(
        code=ActionCode.URGENT_REFERRAL if high else ActionCode.ROUTINE_RESCREEN,
        summary=(
            "Urgent referral for specialist examination and biopsy."
            if high
            else "Routine re-screening; this does not rule out disease."
        ),
        steps=["Refer for clinical examination / biopsy."] if high else ["Re-screen in 1 year."],
        timeline_days=14 if high else 365,
        referral_urgency=ReferralUrgency.URGENT if high else ReferralUrgency.NONE,
    )
    any_synthetic = any(
        c.meta.synthetic for s in inp.sites for c in (s.wl, s.af, s.dark) if c is not None
    ) or any(r.meta.synthetic for r in inp.spectral)
    provenance = DataProvenance(inp.data_provenance)
    synthetic = provenance == DataProvenance.SYNTHETIC or any_synthetic
    if synthetic and provenance == DataProvenance.REAL:
        provenance = DataProvenance.MIXED
    sites = []
    for s in inp.sites:
        key = f"sites/{s.site.value}"
        caps = [c for c in (s.wl, s.af, s.dark) if c is not None]
        sites.append(
            SiteResult(
                site=s.site,
                site_label=site_label(s.site),
                role=s.role,
                visual_flag=s.visual_flag,
                diascopy=s.diascopy,
                captures=[
                    CaptureRecord(metadata=c.meta, qc=QCResult(passed=True, score=0.9))
                    for c in caps
                ],
                qc=QCResult(passed=True, score=0.9),
                analysis_status="analysed",
                calibration=CalibrationInfo(status=CalibrationStatus.UNCALIBRATED),
                artifacts=SiteArtifacts(
                    wl_image=f"{key}/wl.png",
                    heatmap_overlay=f"{key}/heatmap_overlay.png",
                    relative_af_grid=f"{key}/relative_af_grid.json",
                ),
                site_risk_level=level if s in lesion_sites else RiskLevel.LOW,
            )
        )
    factor = ContributingFactor(
        code="visual.white_patch",
        label="White patch seen by the worker",
        source=FactorSource.CLINICAL_VISUAL,
        points=2.0,
    )
    lesions = [
        LesionAssessment(
            lesion_id=f"L{i + 1}",
            site=s.site,
            origins=[LesionOrigin.VISUAL],
            risk_level=RiskLevel.HIGH,
            score=6.5,
            contributing_factors=[factor],
            recommended_action=action,
            summary=f"Suspicious finding at the {site_label(s.site).lower()}.",
        )
        for i, s in enumerate(lesion_sites)
    ]
    wl_info = getattr(classifier, "info", None) or FakeClassifier.info
    return ScreeningResult(
        result_id=inp.result_id or str(uuid.uuid4()),
        screening_id=inp.screening_id,
        result_version=inp.result_version,
        supersedes_result_id=inp.supersedes_result_id,
        created_at=inp.now or FIXED_NOW,
        data_provenance=provenance,
        synthetic=synthetic,
        notices=Notices(synthetic_notice=SYNTHETIC_NOTICE if synthetic else None),
        versions=VersionInfo(
            software=software_version or "mukhaspectra-test",
            pipeline="0.0.0-fake",
            rule_config=cfg.triage.ref,
            qc_config=cfg.qc.ref,
            analysis_config=cfg.analysis.ref,
            models=[wl_info],
        ),
        context=inp.context,
        patient=inp.patient,
        consent=inp.consent,
        risk_factors=inp.risk_factors,
        sites=sites,
        lesions=lesions,
        triage=TriageResult(
            overall_risk_level=level,
            score=6.5 if high else 0.0,
            thresholds=ScoreThresholds(moderate=3.0, high=6.0),
            contributing_factors=[factor] if high else [],
            recommended_action=action,
            highest_risk_lesion_id="L1" if high else None,
            rule_config=cfg.triage.ref,
            explanation="Risk level from the baseline rules (fake analyser).",
        ),
        qc_summary=QCSummary(passed_all=True),
        calibration_status=CalibrationStatus.UNCALIBRATED,
        limitations=[
            Limitation(
                code="calibration.missing",
                message="Device not calibrated; AF intensities are relative.",
            )
        ],
        processing=ProcessingInfo(working_max_side_px=1024, duration_ms=12.5),
    )


def fake_artifacts(result: ScreeningResult) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for site in result.sites:
        key = f"sites/{site.site.value}"
        artifacts[f"{key}/wl.png"] = png_bytes((200, 100, 100))
        artifacts[f"{key}/heatmap_overlay.png"] = png_bytes((250, 50, 50))
        artifacts[f"{key}/relative_af_grid.json"] = json.dumps(
            {"width": 2, "height": 1, "values": [[1.0, 0.6]], "threshold": 0.75}
        ).encode()
        artifacts[f"{key}/contour.svg"] = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
    return artifacts


@dataclass
class FakePipeline:
    """Records every call; ``fail`` makes the next analyses raise."""

    analyses: list[ScreeningInput] = field(default_factory=list)
    adapter_calls: list[dict[str, Any]] = field(default_factory=list)
    spectral_calls: list[dict[str, Any]] = field(default_factory=list)
    demo_calls: list[str] = field(default_factory=list)
    software_versions: list[str | None] = field(default_factory=list)
    fail: bool = False
    extra_artifacts: dict[str, bytes] = field(default_factory=dict)

    def analyze(
        self, inp: ScreeningInput, *, classifier: Any = None, software_version: str | None = None
    ) -> AnalysisOutput:
        self.analyses.append(inp)
        self.software_versions.append(software_version)
        if self.fail:
            raise RuntimeError("simulated pipeline crash")
        for s in inp.sites:
            assert s.wl.image.dtype == np.float32 and s.wl.image.ndim == 3
        result = build_result(inp, classifier, software_version)
        return AnalysisOutput(
            result=result, artifacts={**fake_artifacts(result), **self.extra_artifacts}
        )

    def parse_spectral(self, data: bytes, filename: str, **kwargs: Any) -> Any:
        self.spectral_calls.append({"filename": filename, **kwargs})
        return fake_parse_spectral(data, filename, **kwargs)

    def bridge(self) -> PipelineBridge:
        return PipelineBridge(
            analyze=self.analyze,
            upload_adapter_factory=lambda: FakeUploadAdapter(self.adapter_calls),
            parse_spectral=self.parse_spectral,
            load_image=lambda data, filename, **kwargs: decode_png(data),
            make_demo_patient=self.make_demo_patient,
            load_classifier=lambda model_dir, cfg: FakeClassifier(),
            spectral_classifier_info=lambda: ModelInfo(
                component=ModelComponent.SPECTRAL_CLASSIFIER,
                name="rules_v1",
                version="1.0.0",
                status=ModelStatus.PLACEHOLDER,
            ),
        )

    def make_demo_patient(self, kind: str) -> ScreeningInput:
        self.demo_calls.append(kind)
        return make_fake_demo_patient(kind)


# --------------------------------------------------------------------------------------------
# Fake demo patient (shape of mukha_pipeline.synthetic.make_demo_patient output)
# --------------------------------------------------------------------------------------------


def _syn_meta(kind: CaptureKind, site: OralSite, device: str) -> CaptureMetadata:
    return CaptureMetadata(
        capture_id=str(uuid.uuid4()),
        kind=kind,
        oral_site=site,
        source_adapter=SourceAdapter.SYNTHETIC,
        device_id=device,
        firmware_version="0.0.0-syn",
        exposure_time_ms=40.0,
        iso=200.0,
        synthetic=True,
    )


def make_fake_demo_patient(kind: str) -> ScreeningInput:
    rng = np.random.default_rng(7)
    sites = []
    for site in (OralSite.BUCCAL_MUCOSA_LEFT, OralSite.BUCCAL_MUCOSA_RIGHT):
        images = {k: rng.random((20, 28, 3), dtype=np.float32) for k in ("wl", "af", "dark")}
        images["dark"] *= np.float32(0.02)
        lesion = kind == "high" and site == OralSite.BUCCAL_MUCOSA_LEFT
        sites.append(
            SiteInput(
                site=site,
                wl=ImageCapture(
                    images["wl"], _syn_meta(CaptureKind.WHITE_LIGHT, site, "scout-syn")
                ),
                af=ImageCapture(
                    images["af"], _syn_meta(CaptureKind.AUTOFLUORESCENCE, site, "scout-syn")
                ),
                dark=ImageCapture(
                    images["dark"], _syn_meta(CaptureKind.DARK_FRAME, site, "scout-syn")
                ),
                visual_flag=VisualFlag(
                    lesion_seen=lesion, appearance=[VisualAppearance.RED_PATCH] if lesion else []
                ),
            )
        )

    def frames(base: float, n: int = 3) -> list[SpectralFrame]:
        return [
            SpectralFrame(_channel_row(base + i), gain=64.0, integration_ms=50.0) for i in range(n)
        ]

    spectral = [
        SpectralInput(
            "syn-l",
            OralSite.BUCCAL_MUCOSA_LEFT,
            "lesion",
            frames(800.0),
            frames(40.0, 1),
            _syn_meta(CaptureKind.SPECTRAL, OralSite.BUCCAL_MUCOSA_LEFT, "confirm-syn"),
            reference_reading_id="syn-r",
        ),
        SpectralInput(
            "syn-r",
            OralSite.BUCCAL_MUCOSA_RIGHT,
            "reference",
            frames(1000.0),
            frames(40.0, 1),
            _syn_meta(CaptureKind.SPECTRAL, OralSite.BUCCAL_MUCOSA_RIGHT, "confirm-syn"),
        ),
    ]
    image_cal = ImageCalibration(
        calibration_id="imgcal-syn",
        device_id="scout-syn",
        created_at=FIXED_NOW,
        expires_at=FIXED_NOW + timedelta(hours=24),
        flat_field=rng.random((6, 8), dtype=np.float32) + np.float32(0.5),
        color_matrix=np.eye(3) * 1.01,
        af_scale=1.25,
        synthetic=True,
    )
    spectral_cal = SpectralCalibration(
        calibration_id="spcal-syn",
        device_id="confirm-syn",
        created_at=FIXED_NOW,
        expires_at=FIXED_NOW + timedelta(hours=24),
        dark=_channel_row(40.0),
        white=_channel_row(2000.0),
        gain=64.0,
        integration_ms=50.0,
        synthetic=True,
    )
    return ScreeningInput(
        screening_id="demo-placeholder",
        patient=PatientInfo(pseudonymous_id=f"SYN-DEMO-{kind.upper()}", age_years=48, sex="male"),
        consent=ConsentInfo(consent_recorded=True, method="checkbox"),
        risk_factors=RiskFactors(),
        context=ScreeningContext(
            screening_site_name="SYNTHETIC DEMO CAMP", health_worker_id="hw-demo"
        ),
        sites=sites,
        spectral=spectral,
        image_calibrations={"scout-syn": image_cal},
        spectral_calibrations={"confirm-syn": spectral_cal},
        data_provenance=DataProvenance.SYNTHETIC,
        now=FIXED_NOW,
    )


# --------------------------------------------------------------------------------------------
# Fake report renderer
# --------------------------------------------------------------------------------------------


@dataclass
class FakeRenderer:
    pdf_calls: list[dict[str, Any]] = field(default_factory=list)
    html_calls: int = 0
    svg_calls: list[dict[str, Any]] = field(default_factory=list)
    error: BaseException | None = None

    def _check(self) -> None:
        if self.error is not None:
            raise self.error

    def render_pdf(
        self, result: ScreeningResult, artifacts: Any, *, identity: Any, options: Any
    ) -> bytes:
        self._check()
        first_key = result.sites[0].artifacts.heatmap_overlay if result.sites else None
        self.pdf_calls.append(
            {
                "identity": identity,
                "options": options,
                "artifact": artifacts(first_key) if first_key else None,
                "traversal": artifacts("../../inputs/x.png"),
            }
        )
        name = identity.get("name") or ""
        return (
            f"%PDF-1.7\n% fake report {options['report_id']} v{result.result_version} "
            f"{name}\n%%EOF\n"
        ).encode()

    def render_html(
        self, result: ScreeningResult, artifacts: Any, *, identity: Any, options: Any
    ) -> str:
        self._check()
        self.html_calls += 1
        return (
            f"<html><body><p>{result.notices.prototype_banner}</p>"
            f"<p>{identity.get('name') or ''}</p><p>{options['record_url']}</p></body></html>"
        )

    def render_site_map_svg(self, result: ScreeningResult, *, width: int, show_legend: bool) -> str:
        self._check()
        self.svg_calls.append({"width": width, "show_legend": show_legend})
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}"></svg>'

    def load_facilities(self, path: Any) -> list[Any]:
        return [{"name": "District Hospital (example)", "path": str(path) if path else None}]

    def make_identity(self, *, name: str | None, external_id: str | None) -> dict[str, Any]:
        return {"name": name, "external_id": external_id}

    def make_options(self, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)
