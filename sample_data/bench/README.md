# Bench data slots (TRL 3)

> **Research prototype — not for clinical use.** Bench data verifies the measurement chain
> (optics, firmware, calibration, analysis). It is not clinical evidence and supports no
> performance claim about patients.

This folder is where real bench captures from our TRL-3 experiments go, so they can be replayed
through the same upload path as the [SYNTHETIC demo patients](../README.md). Nothing here is
synthetic; nothing here is a patient. Three slots, one per provenance:

| Slot | `data_provenance` | What goes in |
|---|---|---|
| [`phantom/`](phantom/) | `phantom` | Gelatin phantoms with intralipid (scattering), whole blood or haemoglobin (absorption) and a fluorophore (e.g. FAD or fluorescein) standing in for mucosal autofluorescence; inclusions with extra blood and less fluorophore mimic a fluorescence-loss region. Also white-target and fluorescence-card calibration shots. |
| [`ex_vivo/`](ex_vivo/) | `ex_vivo` | Ex vivo animal tissue (e.g. porcine buccal mucosa or tongue from a food-chain supplier), with time since slaughter and storage recorded. |
| [`volunteer/`](volunteer/) | `volunteer` | Healthy oral mucosa of **consenting team volunteers** only. No lesions are sought; anything unexpected seen during capture is referred for clinical examination, never interpreted by the team. |

## Rules

* **Consent first (volunteers).** Written informed consent on the current form version, recorded
  in the sidecar (`consent_recorded`, form version, date, research-use opt-in) **before** any
  capture. A volunteer may withdraw at any time; delete their files and sidecars when they do.
* **No identities.** Volunteers appear only as a pseudonymous id (`VOL-001`). No names, faces,
  phone numbers or other identifiers in file names, sidecars, EXIF or image content (crop to the
  mouth). Strip GPS from EXIF.
* **Volunteer images stay out of public repositories.** Commit only phantom and ex vivo data
  here. Keep volunteer data in the team's access-controlled storage with the same layout; this
  folder documents the convention.
* **Never mix provenances** in one session folder, and never relabel bench data as `real`.
* **No performance claims.** Bench data checks signal levels, repeatability, registration,
  calibration drift and phantom contrast. It does not estimate how the screen performs on people.

## Layout and naming

```
bench/<slot>/<session_id>/
    <date>_<sample_id>_<target>_wl.jpg          white light
    <date>_<sample_id>_<target>_af.jpg          405 nm autofluorescence
    <date>_<sample_id>_<target>_dark.png        dark frame (LEDs off, AF exposure)
    <date>_<sample_id>_spectrum.csv             Confirm probe, spectral CSV v1
    <date>_<sample_id>.json                     metadata sidecar (template below)
    <date>_calibration_white.jpg                optional: white target for flat field
    <date>_calibration_card.jpg                 optional: fluorescence reference card
```

* `session_id`: `BENCH-<YYYY-MM-DD>-<letter>`, one per bench sitting with fixed settings.
* `date`: `YYYYMMDD`.
* `sample_id`: phantom `PH-<recipe>-<n>` (e.g. `PH-GB050-1` = gelatin with 0.50 % blood,
  sample 1); ex vivo `EV-<species>-<tissue>-<n>` (e.g. `EV-PORC-BUCCAL-2`); volunteer
  `VOL-<nnn>`.
* `target`: an `OralSite` value (`buccal_mucosa_left`, `lateral_tongue_right`, ...) for
  volunteers and ex vivo tissue; `surface` or `inclusion` for phantoms.
* Lower case, no spaces. Spectra follow `docs/pipeline_contracts.md` (white-reference rows use
  `reading_id = calibration`).

## Metadata sidecar

One JSON file per sample, next to its images. The template is
[`sidecar.template.json`](sidecar.template.json): copy it, keep the block that matches the
slot (`phantom`, `ex_vivo` or `volunteer`), set the others to `null`, and fill every field you
know. Unknown values stay `null`; never guess.

| Field | Meaning |
|---|---|
| `schema` | `mukhaspectra.bench_sidecar.v1` |
| `data_provenance` | `phantom`, `ex_vivo` or `volunteer` (must match the slot) |
| `sample_id`, `session_id`, `captured_at`, `operator_id` | identity of the capture (ISO 8601 time with offset) |
| `devices` | Scout / Confirm ids and firmware, phone model |
| `capture_settings` | exposure, ISO and white balance actually used (locked), working distance |
| `calibration` | image and spectral calibration ids, white standard used |
| `files` | the image and spectrum file names of this sample |
| `site` | `OralSite` value, or `null` for phantoms |
| `phantom` / `ex_vivo` / `volunteer` | provenance-specific details (recipe, tissue source, consent) |
| `expected` | what the sample was built or chosen to show (e.g. "inclusion should read darker in AF"); a bench expectation, not a clinical label |
| `notes` | anything unusual: glare, movement, bubbles, temperature |

To replay a session through the pipeline, build an upload manifest (`docs/architecture.md` §6)
with `"data_provenance"` set to the slot's value; the result and report then carry that
provenance.
