"""SYNTHETIC data: WL/AF intraoral images, AS7341 spectra and demo patients.

Everything this package produces is labelled synthetic (a visible stamp in the pixels,
``SYNTHETIC_`` file names, ``synthetic=True`` metadata and SYNTHETIC provenance). It exists to
demo and test the software. It is not evidence about any patient and must never be used to
support a performance claim.
"""

from mukha_pipeline.synthetic.af import (
    generate_af_from_wl,
    generate_dark_frame,
    random_misalignment,
)
from mukha_pipeline.synthetic.lesions import LESION_KINDS, LesionKind, LesionSpec
from mukha_pipeline.synthetic.patients import (
    DEMO_KINDS,
    EXPECTED_LEVEL,
    DemoBundle,
    DemoPatientSpec,
    demo_patient_spec,
    make_demo_bundle,
    make_demo_patient,
)
from mukha_pipeline.synthetic.sample_data import build_manifest, write_sample_data
from mukha_pipeline.synthetic.spectra import (
    generate_spectral_reading,
    generate_white_reference,
    make_spectral_calibration,
    spectral_inputs_to_csv,
)
from mukha_pipeline.synthetic.watermark import stamp_mask, watermark_synthetic
from mukha_pipeline.synthetic.wl import (
    DEFAULT_SIZE,
    SyntheticScene,
    generate_wl_image,
    generate_wl_scene,
)

__all__ = [
    "DEFAULT_SIZE",
    "DEMO_KINDS",
    "EXPECTED_LEVEL",
    "LESION_KINDS",
    "DemoBundle",
    "DemoPatientSpec",
    "LesionKind",
    "LesionSpec",
    "SyntheticScene",
    "build_manifest",
    "demo_patient_spec",
    "generate_af_from_wl",
    "generate_dark_frame",
    "generate_spectral_reading",
    "generate_white_reference",
    "generate_wl_image",
    "generate_wl_scene",
    "make_demo_bundle",
    "make_demo_patient",
    "make_spectral_calibration",
    "random_misalignment",
    "spectral_inputs_to_csv",
    "stamp_mask",
    "watermark_synthetic",
    "write_sample_data",
]
