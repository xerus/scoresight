# Numeric padding follow-up

Validated 2026-10-13. Reducing numeric recognition padding from 320 to 128 pixels
retained accepted accuracy on additional frames from the supplied Camera clips,
and reduced warmed selective-TTA latency by approximately 2.5×. The fresh frames
also revealed an accepted **7→1** error at both widths: the original development
set's 34/34 result did not establish reliable handling of that confusion.

## Data and frozen policy

The new [manifest](../tools/ocr_benchmark/validation_manifest.json) has 71 complete
numeric crops manually checked against the images before running either model:

- 68 score fields at 10% and 90% of frame count from 17 clips, excluding prior
  midpoint frames. Four moving-camera ROIs in clips 5/6 were corrected after
  inspecting their full frames; every included crop was visually checked.
- Two isolated score `7` crops at frames 3600 and 5400 of clip 3, separate from its
  previous midpoint frame 4500.
- One natural isolated `1` from the complete Away foul indicator in clip 5's
  early frame. This is a different scoreboard field, explicitly labeled as
  `away_fouls`; the recordings did not provide an isolated score `1` at the
  inspected time points. A score `11` and two three-digit `108` crops are included.

The [crop sheet](../results/ocr_video_benchmark/validation_crops.jpg) permits
label/geometry review. Eight heavy-glare fields at the 10%/90% frames of clips 9
and 11 were excluded because their ground truth remains uncertain in those
individual frames; [exclusions](../tools/ocr_benchmark/validation_exclusions.json)
retain filenames, frame indices and reasons. Some included frames have glare
but remain legible, including clip 7's early `27`/`37`.

These are additional frames from the **same recordings/session/display** as
development. Many scores repeat. They provide a fixed-policy sanity check,
not independent session validation, and cannot establish robustness to another
camera, font or old laptop. There are no true blank or clock labels.

Both widths use the same bundled model, BGR resize to height 48, normalization,
greedy CTC decoding, native confidence trigger 0.85, and fallback inversion/two
shears with two-vote agreement. Width 128 was selected before examining these
additional frames; no thresholds or transforms were retuned. Recognition preserves
actual resized width when it exceeds the padding minimum. Results normalize
leading zeros only for complete numeric strings. Acceptance below mirrors the
app confidence filter of 0.5, not a probability of correctness.

## Results

| Numeric minimum width | Raw correct | Accepted correct | Accepted wrong | Rejected at 0.5 | Fallback fields | Mean ms | Median ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 320 | 68/71 | 66 | 1 | 4 | 22/71 | 38.81 | 22.09 | 90.24 |
| 128 | 68/71 | 68 | 1 | 2 | 22/71 | 15.48 | 8.43 | 34.40 |

Warmed complete `recognize()` calls include preprocessing, TTA and decoding,
with three repeats per crop on the AMD Ryzen 7 PRO 5850U, one CPU thread. Model
load/video decode and the rest of the app are excluded. Model/manifest hashes
and dependency versions are in
[raw results](../results/ocr_video_benchmark/padding_validation.json). Timing
varies with load and CPU frequency; these figures do not certify old-laptop
deadlines. A direct comparison of the actual candidate app recognizer found all
71 strings, confidence values and variant details identical to the width-128
diagnostic: [app check](../results/ocr_video_benchmark/padding_validation_app.json).

Width 128 recovered two legitimate zero fields whose width-320 fallback
confidences were just below the app's 0.5 acceptance filter. It introduced no
additional accepted wrong value on this sample. Both widths accepted the same
incorrect `1` for a natural `7` at frame 3600, rejected the `11`, and rejected the
isolated foul `1`. The other isolated `7` at frame 5400 was correct at both widths.

| Case | Width 320 | Width 128 |
| --- | --- | --- |
| Clip 3 frame 3600, expected `7` | `1`, confidence 0.5701; accepted wrong | `1`, confidence 0.5908; accepted wrong |
| Clip 3 frame 5400, expected `7` | `7`, confidence 0.7112 | `7`, confidence 0.7491 |
| Clip 3 late Away, expected `11` | Unknown | Unknown |
| Clip 5 early foul indicator, expected `1` | Unknown | Unknown |

The wrong `7` has two fallback variants voting for `1` and only one for `7`;
TTA agreement can reinforce a shared error. This is a concrete reason to test
complete-string temporal decisions next. Raising confidence just to remove this
sample would tune on validation and could reject more legitimate digits.

## Recommendation

The app now uses width 128 for numeric fields within the existing experimental
PP-OCRv4 + TTA option; clock/text fields retain width 320. Focused tests cover
wide-input pixels, all fallback transforms, and mixed-field concurrent calls.

The measurements support numeric width 128 as a useful experimental CPU
optimization, with no observed accepted regression and fewer rejected zeros.
They do **not** justify claiming production-safe accuracy across displays.
Retain width 320 for clocks/general text and complete wide-input geometry.
Before presenting the numeric change as broadly validated, run the packaged
Windows app on representative numeric crops and an actual older laptop, and
collect independently recorded genuine score `1`s, `7`s, blanks, glare and score
transitions. Report rejection rate and accepted errors alongside speed. In
particular, selective TTA remains experimental even with reduced padding.

## Reproduce

The benchmark can decode the manifest's frames directly from original videos,
without relying on temporary exported images. The model and runner require only
the existing DNN dependencies. The candidate's `recognize(NUMBER)` currently
selects width 128; `--verify-app-width 128` checks exact strings, confidences and
variant details against that implementation and records its source hash.

```sh
.venv/bin/python tools/ocr_benchmark/performance_options.py \
  --videos /home/pgrunt@inno.innovatrics.net/Videos/Camera \
  --manifest tools/ocr_benchmark/validation_manifest.json \
  --model models/en_PP-OCRv4_rec_mobile.onnx \
  --output /tmp/padding-validation.json \
  --widths 320 128 --repeats 3 --verify-app-width 128
```

Repeat with `video_manifest.json` for the original 34 development crops; its
follow-up raw results are retained in
[development check](../results/ocr_video_benchmark/padding_validation_development.json).
