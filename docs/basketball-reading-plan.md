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

- Milestone 1 complete: one/two/three numeric cells, LED color and brightness
  controls, normalized and persisted per box, generated UI, and regression tests.
- Offscreen UI edits and reloads checked. Full suite: 82 tests passed, including
  real ONNX inference; configured Ruff lint/format checks passed.
- Six original 1920×1080 lossless camera frames and 11 reviewed numeric labels
  committed separately in `261bc8c`, with a pixel-verifying re-extraction tool.
- Those fixtures reproduce the PP-OCR/TTA accepted 7-to-1 error and rejected
  isolated foul 1 / score 11. No recognition accuracy improvement is claimed
  for those cases.
- Milestone 3 complete: cached per-box readers with global inheritance,
  isolated model-load failures and basketball settings for selected standard
  fields. Mixed-reader/confirmation regression cases and offscreen UI checks
  pass; full suite now has 88 passing tests. See [usage](basketball-settings.md).
- Milestones 2 and 4 remain to do.
- Existing local benchmarks cover mostly scores from one session. Synthetic
  regression tests verify behavior, not accuracy on unseen basketball footage.
