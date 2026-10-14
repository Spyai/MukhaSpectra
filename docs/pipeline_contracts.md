# Pipeline module contracts

The public functions each `mukha_pipeline` module exposes. Modules are built and tested
independently against these signatures, so treat a change here as an interface change. Types
come from `mukha_pipeline.types` (internal dataclasses) and `mukha_pipeline.schema.models`
(the `ScreeningResult` contract). `cfg` is the relevant section of a loaded YAML config, as a dict.

Conventions: images are float32 RGB in [0, 1], (H, W, 3). Masks are bool (H, W). No file or
network I/O in analysis code.

## io/
```python
# io/images.py
@dataclass
class ExifInfo: exposure_time_ms: float | None; iso: float | None; white_balance: str | None
                captured_at: datetime | None; make: str | None; model: str | None
@dataclass
class LoadedImage: image: np.ndarray; exif: ExifInfo; mime_type: str; sha256: str
                   width_px: int; height_px: int
def load_image_bytes(data: bytes, filename: str | None = None) -> LoadedImage
    # JPEG, PNG, TIFF, HEIC/HEIF (pillow-heif), DNG (rawpy, optional extra). Applies EXIF
    # orientation. Raises ImageDecodeError with a plain-language message on failure.
def to_uint8(img: np.ndarray) -> np.ndarray
def encode_png(img: np.ndarray) -> bytes                      # float [0,1] or uint8, RGB or gray
def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]  # (img, scale)

# io/spectral.py
@dataclass
class ParsedSpectral: readings: list[SpectralInput]; calibration: SpectralCalibration | None
def parse_spectral_file(data: bytes, filename: str, *, source_adapter=SourceAdapter.UPLOAD,
                        device_id: str | None = None, synthetic: bool = False,
                        now: datetime | None = None) -> ParsedSpectral
    # CSV or JSON (format below and in docs/device_protocol.md). Groups rows by reading_id.
    # Raises SpectralParseError with a plain-language message on bad input.
```

### Spectral file format v1 (shared by upload, synthetic data and firmware)

CSV, header required, column order free:

```
reading_id,site,role,kind,repeat,F1_415,F2_445,F3_480,F4_515,F5_555,F6_590,F7_630,F8_680,CLEAR,NIR,gain,integration_ms,reference_reading_id,lesion_ref,device_id
```

* `kind` is `lit`, `dark`, `white` or `white_dark`. `white`/`white_dark` rows measure the white
  reflectance standard in the same session; they use `reading_id = calibration` and leave
  `site`/`role` empty, and produce a `SpectralCalibration` valid for 24 h.
* `role` is `lesion` or `reference`; `site` is an `OralSite` value.
* `repeat` is the 0-based index of the reading within the N repeats.
* `gain`, `integration_ms`, `reference_reading_id`, `lesion_ref`, `device_id` are optional.
* A lesion reading without `reference_reading_id` is paired automatically with the reference
  reading at its contralateral (or analogous) site, if exactly one exists.

JSON (`"format": "mukhaspectra.spectral.v1"`):

```json
{"format": "mukhaspectra.spectral.v1", "device_id": "confirm-01", "firmware_version": "0.1.0",
 "white_reference": {"white": [{"F1_415": 0, "...": 0}], "white_dark": [{"...": 0}]},
 "readings": [{"reading_id": "s1", "site": "buccal_mucosa_left", "role": "lesion",
               "reference_reading_id": "s2", "lesion_ref": null, "gain": 64,
               "integration_ms": 50, "dark": [{"F1_415": 210, "...": 0}],
               "lit": [{"F1_415": 5120, "...": 0}]}]}
```

## capture/
```python
class CaptureSource(Protocol):          # server-side mirror of the PWA DeviceAdapter
    adapter: SourceAdapter
class UploadAdapter:                    # builds SiteInput / SpectralInput from uploaded bytes
    def site_input(self, site, wl: bytes, af: bytes, dark: bytes | None, *, filenames,
                   role, visual_flag, diascopy, px_per_mm, device_id, firmware_version,
                   synthetic=False) -> SiteInput
```

## qc/
```python
def qc_white_light(img: np.ndarray, cfg: dict) -> QCResult
def qc_autofluorescence(img: np.ndarray, cfg: dict) -> QCResult
def qc_dark_frame(img: np.ndarray, cfg: dict) -> QCResult
def qc_registration(reg: RegistrationResult, cfg: dict) -> QCResult
def qc_segmentation(masks: Masks, cfg: dict) -> QCResult
def qc_spectral(lit: list[SpectralFrame], dark: list[SpectralFrame], cfg: dict) -> QCResult
def combine_qc(results: list[QCResult]) -> QCResult
def missing_dark_frame_issue() -> QCIssue                      # warning, not an error
# cfg = the full qc.v1.yaml dict; each function reads its own section.
# Issue codes and plain-language messages live in qc/messages.py (English + i18n key).
```

## calibration/
```python
def image_calibration_status(cal: ImageCalibration | None, now: datetime) -> CalibrationInfo
def spectral_calibration_status(cal: SpectralCalibration | None, now: datetime) -> CalibrationInfo
def compute_flat_field(white_target: np.ndarray, blur_sigma_px: float = 25) -> np.ndarray
def compute_color_matrix(white_target: np.ndarray,
                         chart_measured: np.ndarray | None = None,
                         chart_reference: np.ndarray | None = None) -> np.ndarray
def compute_af_scale(fluorescence_card: np.ndarray, expected_green: float) -> float
def create_image_calibration(device_id, white_target, fluorescence_card, *, now,
                             valid_hours=24, expected_card_green=0.5) -> ImageCalibration
def create_spectral_calibration(device_id, dark: dict, white: dict, *, now, valid_hours=24,
                                gain=None, integration_ms=None) -> SpectralCalibration
def reflectance(sample: dict, dark: dict, white: dict, white_dark: dict | None = None) -> dict
    # (S - D) / (W - D) per channel, clipped to a sane range, NaN-safe
def image_calibration_to_bytes(cal) -> bytes / image_calibration_from_bytes(b) -> ImageCalibration
```

## preprocess/
```python
# correction.py
def subtract_dark(img: np.ndarray, dark: np.ndarray) -> np.ndarray
def apply_flat_field(img: np.ndarray, flat: np.ndarray) -> np.ndarray
def apply_color_matrix(img: np.ndarray, matrix: np.ndarray) -> np.ndarray
def exposure_factor(meta: CaptureMetadata, reference: CaptureMetadata | None) -> float
    # ratio that brings `meta`'s intensities onto `reference`'s exposure (time x ISO); 1.0 if unknown

# registration.py
def register_af_to_wl(wl: np.ndarray, af: np.ndarray, cfg: dict) -> RegistrationResult
    # cfg = analysis.registration. ECC (affine) on gradient-magnitude images, multi-scale;
    # fallback ORB + RANSAC homography; identity if both fail (passed=False).

# segmentation.py
class Segmenter(Protocol):
    name: str; version: str
    def segment(self, wl: np.ndarray) -> Masks: ...
class ClassicalSegmenter:  # cfg = analysis.segmentation
    def __init__(self, cfg: dict): ...
def segmentation_summary(masks: Masks, segmenter: Segmenter, cfg: dict) -> SegmentationSummary
```

## af/
```python
def compute_af_maps(af_registered: np.ndarray, valid: np.ndarray, *, intensity_scale: float = 1.0
                    ) -> AFMaps            # intensity_scale = exposure_factor * calibration af_scale
def reference_green(maps: AFMaps, mucosa: np.ndarray, cfg: dict) -> float | None
    # robust in-image reference: mean of mucosa green between configured percentiles
def resolve_reference(site: OralSite, own_reference: float | None,
                      available: dict[OralSite, float]) -> AFReference
    # contralateral > analogous > in-image auto > none
def detect_fvl(maps: AFMaps, mucosa: np.ndarray, reference: AFReference, cfg: dict, *,
               px_per_mm: float | None = None) -> FVLOutput
def flag_confounders(fvl: FVLOutput, wl: np.ndarray, maps: AFMaps, masks: Masks, cfg: dict
                     ) -> FVLOutput   # cfg = full analysis config; fills regions[].confounders
                                      # and analysis.porphyrin_red_fraction
def render_heatmap_overlay(wl: np.ndarray, relative_af: np.ndarray, labels: np.ndarray,
                           threshold: float, cfg: dict) -> np.ndarray   # uint8 RGB
def relative_af_grid(relative_af: np.ndarray, size: int) -> dict
    # {"width", "height", "values": [[float|None]], "threshold"}: JSON-safe, row-major
```

## wl/
```python
class LesionClassifier(Protocol):
    info: ModelInfo
    def classify(self, wl: np.ndarray, mucosa: np.ndarray | None = None) -> WLAnalysis: ...
class UnavailableLesionClassifier:      # status = model_unavailable, pipeline still works
class OnnxLesionClassifier:             # ONNX Runtime; labels/input size/normalisation from cfg
    def __init__(self, model_path: Path, cfg: dict, model_card: str | None = None): ...
def load_lesion_classifier(model_dir: Path | None, cfg: dict) -> LesionClassifier
    # expects model_dir/model.onnx + model_dir/model_card.md; otherwise Unavailable
def occlusion_saliency(predict: Callable[[np.ndarray], np.ndarray], img: np.ndarray,
                       target_index: int, cfg: dict) -> np.ndarray   # float32 (H, W) in [0,1]
```

## spectral/
```python
def summarise_reading(inp: SpectralInput, cal: SpectralCalibration | None, now: datetime
                      ) -> tuple[dict[str, float], dict[str, float], str, CalibrationInfo]
    # (mean values, std values, signal_type, calibration info)
def compute_features(values: dict[str, float]) -> SpectralFeatures
def delta_features(lesion: SpectralFeatures, reference: SpectralFeatures) -> SpectralFeatures
class SpectralClassifier(Protocol):
    info: ModelInfo
    def classify(self, features: SpectralFeatures, delta: SpectralFeatures | None
                 ) -> SpectralClassifierOutput: ...
class RuleBasedSpectralClassifier:      # status "placeholder", method "rules_v1"
def analyze_spectral(inputs: list[SpectralInput], calibrations: dict[str, SpectralCalibration],
                     qc_cfg: dict, analysis_cfg: dict, now: datetime,
                     classifier: SpectralClassifier | None = None
                     ) -> list[SpectralReadingResult]
```

## fusion/
```python
@dataclass
class TriageOutput:
    lesions: list[LesionAssessment]
    triage: TriageResult
    site_levels: dict[OralSite, RiskLevel]
    limitations: list[Limitation]
def run_triage(sites: list[SiteResult], spectral: list[SpectralReadingResult],
               patient: PatientInfo, risk_factors: RiskFactors, rules: LoadedConfig
               ) -> TriageOutput
```

## synthetic/
```python
def generate_wl_image(rng: np.random.Generator, size=(768, 1024), *, site: OralSite,
                      lesion: LesionSpec | None = None) -> tuple[np.ndarray, np.ndarray]
    # (wl image, lesion truth mask)
def generate_af_from_wl(wl, rng, *, tissue_mask=None, fvl_mask=None, fvl_depth=0.5,
                        porphyrin_spots=0, bloom=0.6, misalign=True) -> np.ndarray
def generate_dark_frame(shape, rng, level=0.01) -> np.ndarray
def generate_spectral_reading(rng, *, hb_dip: float, n=5, reading_id, site, role, ...
                              ) -> SpectralInput
def watermark_synthetic(img: np.ndarray) -> np.ndarray       # visible "SYNTHETIC" stamp
def make_demo_patient(kind: Literal["low", "moderate", "high"], seed: int = 0) -> ScreeningInput
def write_sample_data(out_dir: Path) -> None                  # sample_data/synthetic/...
```

## orchestrator
```python
def analyze_screening(inp: ScreeningInput, config: PipelineConfig | None = None, *,
                      classifier: LesionClassifier | None = None,
                      segmenter: Segmenter | None = None,
                      spectral_classifier: SpectralClassifier | None = None
                      ) -> AnalysisOutput
```
