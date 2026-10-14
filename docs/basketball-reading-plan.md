# Basketball scoreboard reading plan

Improve the existing readers using measured failures from the camera recordings.
Temporal score/countdown confirmation already exists; reuse it.

## 1. Configurable numeric LED reader

- Support one, two, and three equal-width digit cells, including scores over 99.
- Add per-box warm, green, and all-saturated LED color choices and brightness.
- Expose settings in the box UI and preserve them in saved/exported boxes.
- Retain the current two-cell warm-LED defaults for existing configurations.
- Reject unknown patterns and missing digits after the first lit digit; distinguish
  physical blanks from zero and uncertain readings.
- Validate with regression tests, settings round trips, and an offscreen UI check.

## 2. Calibrated clocks and geometry

- Support explicit digit positions and separator regions for MM:SS and SS.t.
- Read observed punctuation and blank states rather than infer a running clock.
- Label real game-clock and shot-clock crops before tuning thresholds.
- Test stopped/ticking clocks, final-minute precision changes and resets.

## 3. Per-box reader selection and basketball preset

- Allow each box to inherit the global reader or select an available reader.
- Keep model instances loaded; reset confirmation when reader/settings change.
- Add a basketball preset for scores, game clock, shot clock, period and fouls.
- Use existing score/countdown confirmation; allow score corrections and resets.

## 4. Recognition and sequence validation

- Label held-out sequences from another game/session, including real 1 and 7,
  three-digit scores, glare, movement, blanks and transitions.
- Compare PP-OCR + TTA with the calibrated LED reader on complete matching crops.
- Measure accepted errors, rejections, update delay and processing time.
- Investigate a segment-based ambiguity check or digit classifier if repeated
  7-to-1 errors remain. Do not assume repeated agreement establishes correctness.

## Progress

- Numeric LED cells, color/brightness, calibrated clock cells, saved per-box
  reader selection, and basketball presets are implemented and tested.
- Six full camera frames and reviewed numeric/clock labels are committed. The
  original diagnostic remains an audit of baseline failures, including the
  uncalibrated equal-width LED reader's poor results.
- Optional PP-OCR LED checks now recover demonstrated missing/confused digits.
  Explicit Clock input formatting supports clocks whose separators are obscured.
  Settings scroll so the controls remain reachable on laptop screens.
- Seven chronological sequences contain 203 reviewed lossless crops, with source
  frame/timestamp/ROI/hash metadata and pixel-verifying re-extraction. Actual
  OCR plus confirmation accepts 40 wrong baseline values and zero new wrong
  checked values; three stale score outputs remain during confirmation.
- Existing 94 additional fields pass as regression; ten fresh score crops pass
  both policies after freezing the revised reader. All are from one session.
- See [recognition and measurement details](led-ambiguity-check.md),
  [field settings](basketball-settings.md), and [LED calibration](calibrated-led-clocks.md).

- Final checks: 122 regression tests, including actual ONNX camera replay, pass.
  Configured Ruff lint/format hooks pass; source crop re-extraction matches pixels.

## Remaining external validation

Another game/camera session and physical transitions to tenths are not present
in the available labeled data. Validate those before enabling the LED check by
default or claiming general accuracy. Automatic physical layout switching is
not implemented. The standalone calibrated LED clock reader has synthetic
coverage but still needs legible real separator examples. Temporal confirmation
cannot establish correctness of a persistent wrong reading.
