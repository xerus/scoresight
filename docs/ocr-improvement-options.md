# OCR robustness: code review and next experiments

Reviewed 2026-09-25. This document records code findings and proposed experiments;
it does not claim measured accuracy improvements. See the benchmark report for
actual measurements. No application code was changed during this review.

## What the new examples establish

| Image | Home | Away | Clock |
| --- | --- | --- | --- |
| `home_1_away_0_2:58.png` | 1 | 0 | 2:58 |
| `home_1_away___2:54.png` | 1 | **blank** | 2:54 |
| `home_2_away_2_2:52.png` | 2 | 2 | 2:52 |
| `home_4_away_2_2:47.png` | 4 | 2 | 2:47 |

Blank means that neither Away digit is illuminated. Unlit red LEDs remain
visible, so a recognizer must separate emitted light from the physical segment
pattern. Blank is different from zero, recognition failure, and retaining an
older value. The complete score field must include both digit positions even
when only the right digit is illuminated.

These four pictures provide eight score fields, one blank. They do not cover
illuminated two-digit scores, most digits, multiple cameras, or lighting changes.
Tuning on them is diagnostic, not evidence of general accuracy. Earlier fixed
video crops clipped or missed digits; their model ranking should not guide the
choice of recognizer.

## Defects and limitations visible in the current code

1. **Blank can become zero.** In `src/tesseract.py`, the
   `remove_leading_zeros` branch maps any empty result after `lstrip("0")` to
   `"0"`, including an originally empty OCR result. Apply that conversion only
   to nonempty digit strings. Preserve a distinct blank state.
2. **Blank needs an explicit output policy.** `Empty` is now accepted by text
   file output, the visible table, OBS, vMix, UNO, and pivoted HTTP JSON/XML when
   `skip_empty` is off. A `FailedFilter` does not clear an old value. CSV/XML
   full-result outputs retain their existing structured state behavior. The
   setting and consumers should remain aligned if more output types are added.
3. **Dilation depends on array aliasing.** `src/tesseract.py` writes the
   dilated array into `binary`, without assigning it to `imagecrop`. Global
   crops are views of `binary` and see the update; Local/Adaptive crops and
   crops produced by vertical scaling/skew are separate arrays and can send
   undilated pixels to OCR. Explicitly assign the processed crop at each step.
4. **Preview and recognition can use different geometry.** In
   `src/camera_thread.py`, `preview_frame` is saved before crop, stabilization,
   and homography. With crop enabled, color preview emits this full original
   frame; OCR uses the cropped and subsequently transformed frame. Binary
   preview emits the cropped frame, while target rectangles retain full-frame
   coordinates. Subtracting crop offsets alone cannot account for all these
   transforms. This is a concrete route to the reported shifted overlays.
   Use one explicit coordinate transform chain for preview, OCR, and boxes.
5. **Whole-frame Otsu sees the room.** Global binarization in
   `src/camera_thread.py` derives its threshold from the entire frame, including
   windows and furniture. Local mode in `src/tesseract.py` derives Otsu from
   the field. Neither explicitly distinguishes lit red LEDs from unlit red
   segments. Global and Local are meaningfully different benchmark variants.
6. **Cleanup can remove the signal.** Cleanup deletes individual contours
   smaller than a fraction of crop area, before dilation. A dot-matrix digit
   consists of many small LED components. Increasing cleanup can delete the
   actual digit, especially when a wide two-position field increases the area.
7. **Autocrop guesses polarity from one corner.** Any nonzero sum in the
   top-left 5×5 pixels makes `autocrop()` invert its working mask. This is
   particularly unsuitable for grayscale No Binarization input. It also uses
   inclusive last-pixel coordinates as exclusive slice endpoints, dropping a
   boundary pixel when padding reaches the image edge.
8. **Scaling is fixed and can discard detail.** `rescale_patch` forces the
   crop to 35 pixels high with `INTER_AREA`, even when the original contains
   more useful LED detail. Compare native resolution and controlled larger
   heights; preserve aspect ratio and document border/polarity choices.
9. **Smoothing invents strings.** `OCRResultPerCharacterSmoother` in
   `src/text_detection_target.py` votes separately by position, using the
   oldest string's length. Mixed-length observations can yield truncations or
   combinations never observed. Ties use set iteration order. Prefer a
   deterministic whole-value vote or consecutive-agreement policy, with an
   explicit transition delay and reset on source/geometry/model changes.
10. **Confidence is not a correctness percentage.** The confidence control
    compares Tesseract mean text confidence to a threshold. It does not measure
    LED activity or prevent confident hallucinations. Report accepted accuracy
    together with coverage/rejection rate; high thresholds can hide errors by
    refusing most readings.

## Fair model evaluation

Keep raw model text, normalized value, acceptance/rejection reason, and latency
separate. An invalid prediction must not become a correct blank simply because
the model wrapper returns `None`. Count blank false positives separately from
errors on visible digits. Disable temporal smoothing in independent-image tests.

| Candidate | Required handling |
| --- | --- |
| Bundled Tesseract and official English models | Record engine/library version, model hash, OEM and PSM; test PSM 7/8/13 on complete score fields. Use PSM 10 only for separately segmented single digits. |
| `7seg`, `ssd`, `ssd_plus`, `ssd_int` | Cached traineddata headers contain LSTM entries; `7seg` here is not a legacy-only model. Use a compatible LSTM engine, not an assumed OEM 0. |
| OICWS CRNN | Published RGB transform is resize to 64×256 and normalization to [-1,1]. Its numeric wrapper can reject zero via minimum-value settings and turns blank/invalid/range failure into the same `None`; retain raw CTC decoding and allow score zero. |
| Leander's 17 Keras checkpoints | Repository preprocessing uses inverted Otsu, `cv2.resize` default linear interpolation, and raw 0–255 inputs. Read input shape per checkpoint. Training labels include a leading `+` for positive values; normalize one leading plus only. Do not delete minus signs or internal punctuation to turn an invalid score into a match. |
| Renjith TFLite | Use grayscale 200×31 input divided by 255. Output is decoded indices, not logits. Both `predict.py` and training notebook specify digits + lowercase letters + decimal point, so index 10 is `a`, not a decimal. Runtime needs supported Flex operations. |
| OpenCV segment baseline | Use the same complete field ROIs and include blank detection. Keep manually supplied digit-cell geometry explicit: it is additional calibration, not learned general recognition. |

The external preprocessing findings above are from the cached repositories
`/tmp/lcd-digit-recognition/ocr_reader.py`, `/tmp/leander-sparse/dataset.py`,
`/tmp/leander-sparse/image_filtering.py`, `/tmp/renjith2/predict.py`, and
`/tmp/renjith2/keras_ocr_7_seg.ipynb`. These temporary paths are inspection
references; reproducible benchmark artifacts should record model hashes and
repository revisions rather than depend on these paths surviving reboot.

## Recommended implementation order

1. Fix geometry, blank handling, and inconsistent preprocessing first. Expose
   the exact patch sent to OCR beside its raw text, confidence, and rejection
   reason. This makes errors diagnosable instead of requiring slider guessing.
2. Add an LED-specific preprocessing option. Preserve color, combine brightness
   with color/chroma evidence, and measure activity relative to the local dark
   background. Red-only hue rules are unsuitable for green clocks or saturated
   LED centers; make field color configurable or infer it from calibration.
   Add a blank/activity check before recognition and report uncertain activity
   as unknown rather than zero.
3. Compare Tesseract variants and an OpenCV segment reader with controlled
   preprocessing. For a fixed scoreboard, calibrated digit cells plus seven
   segment occupancy are a plausible baseline. Dot spacing, perspective,
   reflections, and partially illuminated frames must remain rejection cases.
   Use a runner-up margin to reject ambiguous segment patterns.
4. Add deterministic temporal decisions after measuring raw recognition. A
   short confirmation window can suppress flicker, but must not enforce
   monotonic scores: resets and corrections are valid. Clock rules should be
   separate from score rules. Measure delay as well as accuracy on video.
5. Collect held-out clips covering all digits, two-digit scores, blanks,
   resets, angles, exposure changes, and motion. Split by clip/session before
   tuning; adjacent frames are not independent validation examples.
6. Train or fine-tune only if the measured gap remains. Training data should
   include this display's dotted LEDs, dark unlit segments, glare, perspective,
   blur, and blank fields. A newer model trained on another type of meter is
   not automatically a better fit. GPU acceleration affects throughput, not
   the information available in a clipped or incorrectly thresholded crop.

The immediate goal is a reliable complete pipeline, with explicit abstention
when evidence is insufficient, rather than forcing a digit on every frame.
