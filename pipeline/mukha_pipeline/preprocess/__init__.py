"""Image preprocessing in the white-light (WL) frame, before any autofluorescence (AF) analysis.

Submodules (import them directly; this package deliberately imports nothing so that each
submodule stays independently importable and testable):

* ``correction``   dark-frame subtraction, flat-field and colour correction, exposure factor.
* ``registration`` AF -> WL alignment (ECC affine on gradient structure, ORB + RANSAC fallback,
  identity with ``passed=False`` when neither can be trusted).
* ``segmentation`` mucosa / teeth / background / specular masks (``Segmenter`` interface).

Order used by the orchestrator: dark subtraction -> flat-field -> colour (WL only) ->
registration -> segmentation. Corrections never raise on bad pixels; they sanitise and clip.
"""
