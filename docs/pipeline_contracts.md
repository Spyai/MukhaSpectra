# Pipeline module contracts

The public functions each `mukha_pipeline` module exposes. Modules are built and tested
independently against these signatures, so treat a change here as an interface change. Types
come from `mukha_pipeline.types` (internal dataclasses) and `mukha_pipeline.schema.models`
(the `ScreeningResult` contract). `cfg` is the relevant section of a loaded YAML config, as a dict.

Conventions: images are float32 RGB in [0, 1], (H, W, 3). Masks are bool (H, W). No file or
network I/O in analysis code.

**Encoded vs linear light.** Decoded images hold the file's sRGB-encoded values. White light stays
encoded throughout (its colour rules and QC thresholds are defined on encoded values). AF and dark
frames are converted to linear light (`srgb_to_linear`, exact IEC 61966-2-1) before any
radiometric step: dark subtraction, AF flat field, exposure scaling and every AF ratio
(relative AF, confounders, references) work on linear light. QC judges the captures as stored
(encoded). Images shown to people (`af.jpg`, `af_registered.jpg`, overlays) are encoded again.

## io/
```python
# io/images.py
@dataclass
class ExifInfo: exposure_time_ms: float | None; iso: float | None; white_balance: str | None
                captured_at: datetime | None; make: str | None; model: str | None
@dataclass
class LoadedImage: image: np.ndarray; exif: ExifInfo; mime_type: str; sha256: str
                   width_px: int; height_px: int          # as decoded
                   bit_depth: int; notes: list[str]
                   original_width_px: int | None; original_height_px: int | None
                   decode_scale: float                    # decoded width / original width
                   orientation: int; synthetic_markers: list[str]
MAX_IMAGE_PIXELS = 50_000_000                              # larger images are refused
def load_image_bytes(data: bytes, filename: str | None = None, *,
                     max_pixels: int = MAX_IMAGE_PIXELS, max_side: int | None = None,
                     raw_white_balance: Literal["camera", "unity"] = "camera") -> LoadedImage
    # JPEG, PNG, TIFF, WebP, BMP, HEIC/HEIF (pillow-heif), DNG (rawpy, optional extra). Applies
    # EXIF orientation. Values are sRGB-encoded (ICC converted to sRGB; DNG rendered with the
    # exact sRGB curve (gamma 2.4, slope 12.92) and no auto-brightening). `max_side` shrinks
    # WHILE decoding (JPEG draft, Pillow reduce, then area resampling; DNG half_size), so a
    # large photo never exists at full size in memory. `raw_white_balance="unity"` (AF and dark
    # frames) keeps the sensor's own channel balance instead of the camera's white balance.
    # SYNTHETIC markers in EXIF / PNG text / comments / XMP are reported in synthetic_markers.
    # Raises ImageDecodeError with a plain-language message on failure.
def to_uint8(img: np.ndarray) -> np.ndarray
def encode_png(img: np.ndarray, *, bit_depth: Literal[8, 16] = 8) -> bytes  # float [0,1] or uint8
def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]  # (img, scale)

# io/sanitize.py
@dataclass
class SanitizedImage: data: bytes; removed: tuple[str, ...]; sanitised: bool; orientation: int
def strip_identifying_metadata(data: bytes, *, orientation: int | None = None) -> SanitizedImage
    # JPEG: drops APP1 (EXIF/XMP), APP13 (IPTC), other APPn, COM and bytes after EOI; PNG: drops
    # eXIf/tEXt/iTXt/zTXt and trailing bytes. Compressed image data, ICC and Adobe segments are
    # untouched (decoded pixels identical); a minimal EXIF with only the orientation is written
    # back. Other formats (HEIC, DNG, TIFF, WebP, BMP) are returned unchanged (sanitised=False):
    # their metadata is still stored as uploaded.

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
UPLOAD_DECODE_MAX_SIDE = 2048           # uploads are decoded at most this size (longest side)
class UploadAdapter:                    # builds SiteInput / SpectralInput from uploaded bytes
    def site_input(self, site, wl: bytes, af: bytes, dark: bytes | None = None, *,
                   filenames=None, source_filenames=None, role=SiteRole.EXAMINED,
                   visual_flag=None, diascopy=Diascopy.NOT_DONE, px_per_mm=None,
                   device_id=None, firmware_version=None, synthetic=False) -> SiteInput
    # Decodes at most UPLOAD_DECODE_MAX_SIDE; WL with the camera white balance, AF and dark with
    # unity white balance. CaptureMetadata.width_px/height_px keep the ORIGINAL capture size;
    # extra carries decoded_width_px/decoded_height_px/decode_scale/exif_orientation and
    # synthetic_marker. px_per_mm is given for the original pixels and converted to the decoded
    # ones. A capture is synthetic when the caller says so, its source file name starts with
    # SYNTHETIC, or the file carries a SYNTHETIC marker.
```

## qc/
```python
def qc_white_light(img: np.ndarray, cfg: dict) -> QCResult
def qc_autofluorescence(img: np.ndarray, cfg: dict, *, white_balance: str | None = None
                        ) -> QCResult
    # error af_not_green when green is not dominant (auto white balance, greyscale, red over the
    # whole field); warning af_auto_white_balance when capture metadata says AWB was on
def qc_dark_frame(img: np.ndarray, cfg: dict) -> QCResult
def qc_registration(reg: RegistrationResult, cfg: dict) -> QCResult
def qc_segmentation(masks: Masks, cfg: dict, *, status: str | None = None) -> QCResult
    # status "failed" -> error segmentation_failed (site not analysed);
    # "low_confidence" -> warning segmentation_low_confidence (site analysed, cannot give LOW)
def qc_spectral(lit: list[SpectralFrame], dark: list[SpectralFrame], cfg: dict) -> QCResult
def combine_qc(results: list[QCResult]) -> QCResult
def missing_dark_frame_issue() -> QCIssue                      # warning, not an error
# cfg = the full qc.v1.yaml dict; each function reads its own section.
# Issue codes and plain-language messages live in qc/messages.py (English + i18n key).
# QC sees the captures as stored (sRGB-encoded file values); its thresholds are defined there.
# A greyscale photo saved as three equal channels is an error (image_grayscale).
```

## calibration/
```python
def image_calibration_status(cal: ImageCalibration | None, now: datetime) -> CalibrationInfo
def spectral_calibration_status(cal: SpectralCalibration | None, now: datetime) -> CalibrationInfo
def compute_flat_field(white_target: np.ndarray, blur_sigma_px: float = 25) -> np.ndarray
def compute_color_matrix(white_target: np.ndarray,
                         chart_measured: np.ndarray | None = None,
                         chart_reference: np.ndarray | None = None) -> np.ndarray
def compute_af_flat_field(fluorescence_card: np.ndarray, blur_sigma_px: float = 50.0, *,
                          encoding: Literal["srgb", "linear"] = "srgb") -> np.ndarray
    # violet-LED illumination profile: card green channel, linearised, blurred, mean 1
def compute_af_scale(fluorescence_card: np.ndarray, expected_green: float) -> float
def create_image_calibration(device_id, white_target, fluorescence_card, *, now,
                             valid_hours=24, expected_card_green=0.5, synthetic=False,
                             chart_measured=None, chart_reference=None,
                             flat_blur_sigma_px=25, af_flat_blur_sigma_px=50.0,
                             card_encoding="srgb") -> ImageCalibration
    # ImageCalibration: calibration_id, device_id, created_at, expires_at, flat_field (white
    # LED, WL only), color_matrix (WL only), af_scale (measured in linear light),
    # af_flat_field (violet LED, linear AF only; None for calibrations made before 1.1.0 ->
    # limitation af_flat_field_missing). AF is never divided by the white-light flat field.
def create_spectral_calibration(device_id, dark: dict, white: dict, *, now, valid_hours=24,
                                gain=None, integration_ms=None) -> SpectralCalibration
def reflectance(sample: dict, dark: dict, white: dict, white_dark: dict | None = None) -> dict
    # (S - D) / (W - D) per channel, clipped to a sane range, NaN-safe
def image_calibration_to_bytes(cal) -> bytes / image_calibration_from_bytes(b) -> ImageCalibration
    # round-trips af_flat_field (absent in older files -> None)
```

## preprocess/
```python
# correction.py
def srgb_to_linear(img: np.ndarray) -> np.ndarray     # exact IEC 61966-2-1 decoding curve
def linear_to_srgb(img: np.ndarray) -> np.ndarray     # its inverse (for images shown to people)
def subtract_dark(img: np.ndarray, dark: np.ndarray | None) -> np.ndarray   # linear light only
def apply_flat_field(img: np.ndarray, flat: np.ndarray | None) -> np.ndarray
def apply_color_matrix(img: np.ndarray, matrix: np.ndarray) -> np.ndarray
def exposure_factor(meta: CaptureMetadata, reference: CaptureMetadata | None) -> float
    # ratio that brings `meta`'s intensities onto `reference`'s exposure (time x ISO); 1.0 if
    # unknown. Valid for linear light only.
# AF order in the orchestrator: decode (encoded) -> QC -> srgb_to_linear(AF), srgb_to_linear(dark)
# -> subtract_dark -> apply_flat_field(af_flat_field) -> intensity_scale = exposure_factor x
# af_scale. Registration runs on the corrected AF re-encoded to sRGB (same value space as WL);
# the registered result is linearised again for compute_af_maps.

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
    # Fills holes enclosed by mucosa up to fill_holes_max_fraction (0.35) of the image, so a
    # large white patch stays in the analysis; a hole that is mostly dark cavity shadow (the
    # mouth opening) keeps the 0.10 limit. Specular = small near-white blobs only.
def refine_masks_with_af(masks: Masks, af_green: np.ndarray, valid: np.ndarray, cfg: dict
                         ) -> tuple[Masks, float]
    # Colour rules call a white patch "teeth"; enamel fluoresces brightly, a white patch does
    # not. Whitish areas touching the mucosa whose AF (LINEAR registered green, WL frame) is not
    # brighter than the mucosa (teeth_af_min_ratio) go back into the mucosa. Returns the refined
    # masks and the image share still excluded as teeth (-> limitation white_area_not_analysed).
    # Run after registration and before compute_af_maps / reference_green.
def segmentation_summary(masks: Masks, segmenter: Segmenter, cfg: dict) -> SegmentationSummary
    # status ok | low_confidence | failed ("failed" whenever there is no mucosa pixel)
```

## af/
```python
def compute_af_maps(af_registered: np.ndarray, valid: np.ndarray | None, *,
                    intensity_scale: float = 1.0) -> AFMaps
    # af_registered is LINEAR light; intensity_scale = exposure_factor * calibration af_scale
def reference_green(maps: AFMaps, mucosa: np.ndarray, cfg: dict) -> float | None
    # robust in-image reference: mean of mucosa green between the 50th and 95th percentiles,
    # after setting aside highlights above reference_outlier_ratio x the 90th percentile; FVL
    # covering up to ~45% of the mucosa does not drag it down
def resolve_reference(site: OralSite, own_reference: float | None,
                      available: dict[OralSite, float], *, min_paired_ratio: float = 0.67
                      ) -> AFReference
    # contralateral > analogous > in-image auto > none. The level used is
    # max(paired, own): a paired site can only RAISE the reference (more sensitive), never hide
    # a loss because the other side read dimmer (exposure, angle, or change on both sides).
    # The type stays contralateral / analogous; the note says when the own level was used.
    # min_paired_ratio is accepted for compatibility and ignored. The orchestrator offers only
    # sites with no recorded lesion (and no recorded appearance at an examined site).
def detect_fvl(maps: AFMaps, mucosa: np.ndarray, reference: AFReference, cfg: dict, *,
               px_per_mm: float | None = None) -> FVLOutput
    # per region: border_irregularity = 1 - convex hull perimeter / perimeter (outline
    # simplified with a 1 px tolerance): 0 for any smooth outline at any elongation
def flag_confounders(fvl: FVLOutput, wl: np.ndarray, maps: AFMaps, masks: Masks, cfg: dict, *,
                     intensity_scale: float = 1.0) -> FVLOutput
    # cfg = full analysis config; fills regions[].confounders and analysis.porphyrin_red_fraction.
    # porphyrin: smoothed red >= porphyrin_red_rel_min x median mucosa red AND red/green >= 0.85
    #   (a red EXCESS; in deep FVL red/green rises with no red emission, which is not plaque)
    # blood/red: WL L* <= 45 and a* >= 25 (never discounted; triage red-area floor)
    # pigmentation: dark brown with a* < 20 (reddish pixels are never pigment)
    # glare: AF saturation judged on unscaled values (intensity_scale divided out)
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
def compute_features(values: dict[str, float]) -> SpectralFeatures   # absolute features
def quotient_spectrum(lesion: dict[str, float], reference: dict[str, float]) -> dict[str, float]
    # q_c = lesion_c / reference_c per channel
def reference_delta(lesion: dict[str, float], reference: dict[str, float]
                    ) -> SpectralFeatures | None
    # delta_vs_reference = features(q) - features(all ones over the same channels): ratios become
    # the relative change ratio_l / ratio_r - 1, hb_absorption_proxy the 540-590 nm dip of q,
    # slopes slope_l - slope_r. The probe's per-channel gain (LED x filter x sensor) cancels in
    # q, so calibrated and uncalibrated readings of the same tissue give the same delta. None
    # when the reference has no usable F4-F7 signal (the reading is then "indeterminate", never
    # reassuring). A feature whose channels are missing scores 0, not -1.
def delta_features(lesion: SpectralFeatures, reference: SpectralFeatures) -> SpectralFeatures
    # plain feature difference; kept for compatibility, NOT used for delta_vs_reference
class SpectralClassifier(Protocol):
    info: ModelInfo
    def classify(self, features: SpectralFeatures, delta: SpectralFeatures | None
                 ) -> SpectralClassifierOutput: ...
class RuleBasedSpectralClassifier:      # status "placeholder" (never "ok"), method "rules_v2",
                                        # version 2.0.0; first reason says placeholder heuristic,
                                        # not a trained model, not validated
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
    # Every probe reading with role "lesion" becomes a lesion whatever its label. A site whose
    # segmentation status is low_confidence or failed cannot give LOW. A site not analysed
    # because of a software error (analysis_status "not_analysed") is never sent for a retake.
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
                      spectral_classifier: SpectralClassifier | None = None,
                      software_version: str | None = None, strict: bool = False
                      ) -> AnalysisOutput
    # software_version is recorded in versions.software (the backend passes "<version>+<build>").
    # strict=True re-raises a crash; otherwise a crash in either pass makes that site
    # "not_analysed" (software error, limitation analysis_error), never a QC failure.
    # working_resolution.scale_from_original = working width / ORIGINAL capture width (the
    # decode-time downscaling of uploads is folded in).
```

## Limitation codes

`ScreeningResult.limitations[]` (one entry per code; the orchestrator's entry wins on a clash):

| Code | Source | Meaning |
|---|---|---|
| `synthetic_data` | orchestrator | synthetic inputs |
| `uncalibrated` / `calibration_expired` / `calibration_partial` | orchestrator | device calibration state |
| `no_dark_frame` | orchestrator | room light not checked or subtracted |
| `analysis_error` | orchestrator, fusion | software error at a site; examine clinically |
| `af_flat_field_missing` | orchestrator | calibrated site without a violet-LED flat field |
| `possible_bilateral_change` | orchestrator | contralateral level < `paired_reference_dimmer_ratio` (0.9) x own |
| `white_area_not_analysed` | orchestrator | > 3% of the image excluded as teeth after the AF cross-check |
| `sites_failed_qc` | fusion | examined sites whose captures failed QC (retake) |
| `device_analysis_unavailable` | fusion | no examined site could be analysed; result rests on observation and symptoms |
| `segmentation_unreliable` | fusion | segmentation low_confidence or failed at an analysed site |
| `reference_in_image` / `reference_site_fvl` | fusion | reference quality |
| `spectral_no_reference` / `spectral_qc_failed` | fusion | probe reading limits |
| `wl_model_unavailable` / `triage_rules_unvalidated` | fusion | model and rule status |
