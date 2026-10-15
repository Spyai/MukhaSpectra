# MukhaSpectra device protocol

> Research prototype, not for clinical use. MukhaSpectra is a screening and triage aid; it
> flags suspicious findings and assigns a risk level. It is not a diagnostic device.

This document is the shared contract between the **Confirm** probe firmware (ESP32 + AS7341),
the PWA device adapters, the synthetic data generator and the pipeline parser
(`pipeline/mukha_pipeline/io/spectral.py`). Change it only together with the parser and its tests.

## 1. Confirm probe sensor: AS7341 channels

The AS7341 is an 11-channel spectral sensor. MukhaSpectra uses its eight visible bands (`F1`–`F8`)
plus the broadband `CLEAR` and the `NIR` channel. Channel names in every file and API are exactly
the names below (`mukha_pipeline.schema.models.AS7341_CHANNELS`).

| Name | Peak (nm) | FWHM (nm, approx.) | Role in oral mucosa reflectance |
|---|---|---|---|
| `F1_415` | 415 | 26 | Haemoglobin Soret band: strongest blood absorption. White LEDs emit little here, so counts are low. |
| `F2_445` | 445 | 30 | Blue; also the white LED's pump peak. |
| `F3_480` | 480 | 36 | Start of the 480 → 555 nm log-slope (absorption rising into the Q bands). |
| `F4_515` | 515 | 39 | Short-wavelength anchor of the haemoglobin-dip baseline. |
| `F5_555` | 555 | 39 | Between the oxy-haemoglobin Q bands (542 / 577 nm) and on deoxy-Hb's 555 nm peak: darkens with blood volume. |
| `F6_590` | 590 | 40 | Falling edge of the 577 nm Q band. |
| `F7_630` | 630 | 50 | Little haemoglobin absorption; long-wavelength anchor of the baseline. |
| `F8_680` | 680 | 52 | Red; end of the 590 → 680 nm log-slope. |
| `CLEAR` | broadband | — | Total signal; used for the tissue-contact and ambient-leak checks. |
| `NIR` | 910 | — | Near infrared; high dark counts mean ambient light is leaking past the probe tip. |

Peaks and widths are from the AS7341 datasheet and vary between parts by a few nm; nothing in
the pipeline depends on the exact values beyond the nominal peak used for slopes
(`AS7341_WAVELENGTH_NM`).

Counts are raw 16-bit ADC values (0–65535). They scale linearly with **gain** (0.5x–512x) and
**integration time** (`integration_ms`, set by ATIME/ASTEP: (ATIME + 1) × (ASTEP + 1) × 2.78 µs).
Readings near full scale are saturated and fail QC.

## 2. Measurement sequence

For each probed spot (a lesion or its reference):

1. Probe tip in contact with the mucosa, white LED **off**: one or more `dark` readings
   (ambient light plus sensor offset).
2. White LED **on**: N `lit` readings (N ≥ 3 recommended; QC checks repeatability across them).
3. Same gain and integration time for dark and lit readings of one spot.

Reference readings are taken on clinically normal mucosa at the paired site: the contralateral
site where one exists (left vs right buccal mucosa, lateral tongue, retromolar area), otherwise
the analogous site (upper vs lower lip mucosa or gingiva). Sites without a pair (tongue dorsum,
palate, floor of mouth) need an explicit `reference_reading_id` if a comparison is wanted.

Daily white calibration: hold the probe on the white reflectance standard, take `white_dark`
(LED off) then `white` (LED on) readings. The pipeline computes

```
reflectance = (S − D) / (W − Dw)
```

where S/D are a reading's lit/dark counts and W/Dw the white standard's lit/dark counts. If the
white standard was read at a different gain or integration time, W − Dw is scaled by the ratio
of gain × integration time. Without a valid calibration the pipeline reports a **relative
signal** (each channel's dark-subtracted counts divided by the sum over F1–F8). Absolute levels
are lost, and the reading's own band ratios still carry the probe's per-channel gain
`K_c` (LED output × filter transmission × sensor response in that band), so a relative-signal
reading's absolute features depend on the probe and must not be compared with fixed thresholds
or with another probe's readings. A calibration is valid for 24 h; an expired one is still
applied and flagged `expired` in the report.

### 2.1 Lesion-versus-reference comparison

The comparison that the triage scores (`delta_vs_reference`, `spectral/features.py`
`reference_delta`) is computed on the **quotient spectrum**: each channel of the lesion reading
divided by the same channel of the same patient's reference reading,

```
q_c = lesion_c / reference_c
    = (K_c R_lesion,c / S_lesion) / (K_c R_reference,c / S_reference)     (relative signal)
    = (R_lesion,c / R_reference,c) × (S_reference / S_lesion)
```

`K_c` cancels exactly, leaving the tissue reflectance quotient times one constant shared by every
channel (for calibrated reflectance `q_c = R_lesion,c / R_reference,c` directly). Every feature
is unchanged by an overall scale, so calibrated and uncalibrated readings of the same tissue give
the same comparison, and the delta is defined as the features of `q` minus the features of a
flat (all-ones) spectrum over the same channels:

* `ratio_*` deltas are relative changes, `ratio_lesion / ratio_reference − 1` (−0.10 = 10 % lower);
* `hb_absorption_proxy` is the 540–590 nm haemoglobin dip of `q` against a straight 515–630 nm
  baseline (> 0 = a deeper dip in the lesion);
* slopes are `slope_lesion − slope_reference`.

Both readings must use the same probe, gain and integration time (step 3 above). If the
reference has no usable signal in F4–F7 there is no delta and the reading is reported as
indeterminate, never as reassuring. The spectral classifier is a **placeholder heuristic**
(`rules_v2`, not trained, not validated); its cut-offs (dip ≥ 0.025 / ratio ≤ −4 % first tier,
0.06 / −10 % top tier) were set on synthetic readings and are kept equal to the triage tiers.
A "no spectral difference detected" result does not rule out disease.

## 3. Spectral file format v1

Shared by upload, synthetic data and firmware. Copied from `docs/pipeline_contracts.md`.

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

### 3.1 Parser rules (`parse_spectral_file`)

Required, or the file is rejected with a plain-language `SpectralParseError` naming the row:

* CSV header with `reading_id`, `kind` and `F1_415` … `F8_680`. Row numbers in messages count
  the header as row 1, as a spreadsheet does. JSON messages name the reading and frame
  (e.g. `Reading 2 ('s2'), lit frame 3`).
* Every `lit`/`dark` row has a known `site` and `role`; every channel value is a finite number.
* Each reading has at least one `lit` and one `dark` row, and all its rows agree on site, role,
  `reference_reading_id`, `lesion_ref` and `device_id`.
* `reference_reading_id`, when given, names a `reference` reading in the same file.
* White calibration needs both `white` and `white_dark` rows. `reading_id = calibration` is
  reserved for them.

Accepted for robustness (firmware and spreadsheet exports differ):

* Column names are case-insensitive; unknown columns are ignored and listed in `notes`.
* Delimiter `,`, `;` or tab; UTF-8 with or without BOM; blank lines and lines starting with `#`.
* `site` values in any case, with `-` or spaces instead of `_`; `gain` written as `64x`.
* `CLEAR` and `NIR` may be empty or absent.
* Optional ISO 8601 `captured_at` (CSV column, JSON reading or top level).
* JSON may set `"synthetic": true`; a file can add the synthetic flag, never remove it.
* JSON frames may carry their own `gain` / `integration_ms`, overriding the reading's.
* Files up to 5 MB.

Output: one `SpectralInput` per reading (frames ordered by `repeat`), each with
`CaptureMetadata` (`kind = spectral`, `led_mode = WHITE`, source adapter, device and firmware,
file name and SHA-256, `calibration_id` when the file carries a white calibration, and
`extra.reference_pairing` = `explicit`, `auto_contralateral`, `auto_analogous` or `none`), plus
an optional `SpectralCalibration` (24 h validity).

## 4. BLE GATT (Phase 3)

Placeholder: to be specified with the firmware in Phase 3. Planned scope:

* **Scout clip service:** LED control (`OFF` / `WHITE` / `VIOLET`, PWM brightness), capture-sync
  command (dark → WL → AF), battery level, device ID and firmware version. Safety: violet LED
  auto-off after ≤ 10 s per exposure and a hard maximum duty cycle enforced in firmware.
* **Confirm probe service:** commands `DARK_READ`, `LIT_READ`, `READ_N`, white calibration
  routine; responses as JSON frames using the channel names in §1, so a session can be saved
  directly as a spectral file v1 (§3). The same JSON frames are sent over USB serial.
* Service and characteristic UUIDs, MTU and chunking, and error codes: TBD.
