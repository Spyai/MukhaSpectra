---
name: <model name, e.g. wl-lesion-classifier>
version: <semver, e.g. 0.1.0>
component: <wl_lesion_classifier | spectral_classifier | segmenter | fusion>
status: <placeholder | research | candidate>
validated: false
---

# Model card: <model name> v<version>

> Research prototype — not for clinical use. This model supports screening and triage. It does
> not diagnose. Final diagnosis requires a clinician and histopathology.

## Intended use
- **Task:** <e.g. classify a white-light oral-site photo into normal / white patch / red patch /
  ulcer-mass / other>
- **Users:** <community health workers, dental screening camps; outputs reviewed by a clinician>
- **Setting:** <device, capture protocol version, oral sites covered>
- **Out of scope:** <diagnosis; paediatric patients; images not captured with the Scout protocol>

## Training data
- **Source(s):** <dataset name, version, licence, URL; or MukhaSpectra flywheel export id>
- **Provenance:** <real / phantom / ex vivo / volunteer>. Synthetic data used? <no / yes: how,
  and confirmation that no synthetic image is in any evaluation set>
- **Labels:** <who labelled, how; histopathology-linked or visual-only>
- **Size and balance:** <counts per class, per site, per sex and age band, per skin tone>
- **De-identification and consent:** <method; research-use opt-in only>

## Evaluation
- **Evaluation data:** <held-out set description; never synthetic>
- **Metrics:** <report with confidence intervals and the evaluation protocol. Leave empty until a
  documented evaluation exists. Do not copy numbers into UI text or marketing.>
- **Subgroup results:** <by site, sex, age band, tobacco habit, device>

## Limitations and known failure modes
- <e.g. keratinised sites (gingiva, hard palate) produce more false-positive FVL>
- <e.g. red porphyrin fluorescence from plaque mimics FVL in the green channel>
- <e.g. pigmentation and blood absorb excitation light>

## Versions and artefacts
- **Architecture:** <MobileNetV3-Small / EfficientNet-B0, input size, preprocessing>
- **Training code:** <commit hash, script, config>
- **Exported ONNX:** `model.onnx`, sha256 `<hash>`, opset <n>
- **Changelog:** <what changed since the previous version and why>

## Ethics and safety
- Outputs are shown with reasons and always routed to clinician review.
- A LOW output never says "no disease"; it recommends routine re-screening.
