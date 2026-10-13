# Confirm score and clock changes

Select an OCR box and choose **Confirm changes**: **Numbers / scores**,
**Countdown clock**, or **Count-up clock**. The default is **Off**. This applies
to Tesseract, PP-OCRv4 + TTA, and the OpenCV reader after confidence/format checks.
For a numeric shot-clock box, select Countdown explicitly. Settings are saved
with each box; confirmation history is runtime state and is not saved.

## Scores and other numbers

The first value needs two matching readings. Once confirmed, unchanged readings
are accepted immediately. Increases of 1, 2, or 3 need two matching readings.
Decreases, resets, and larger increases need three. These are soft continuity
checks: every repeated numeric change can eventually be accepted, including
manual score corrections and multiple baskets between observations. There is
no permanent monotonic-score rule or maximum-score restriction.

For example, after confirming `7`, one or two readings of `1` are held as
unconfirmed. A return to `7` cancels them. Three repeated readings of `1` accept
the correction. After confirming `24`, two readings of `27` accept a three-point
basket; three readings of `108` accept a large correction/jump.

## Clocks

Two consistent observations establish a clock. They may be different successive
ticks, so confirmation also works at one OCR reading per second. Once confirmed,
the clock may remain stopped or move in its configured direction by an amount
compatible with elapsed time and displayed precision. Such ordinary ticks are
accepted immediately. A jump, reversal, or period reset needs three consistent
clock observations; those observations may continue ticking.

Only text actually observed by OCR is published. The filter never fills in an
unread clock from a predicted countdown. Video files use playback timestamps,
with frame position/FPS as a fallback, so acceleration and backward seeks do
not rely on wall-clock time. Pause/resume, reconnect, model changes, ROI coordinates,
and settings changes reset history. Duplicate video timestamps do not count as
additional confirmation. Rejected readings interrupt a pending confirmation;
a clock must be reconfirmed after a rejected reading.

## Output, delay, and limits

While a change is pending, its raw text is marked `FailedFilter`. The visible
accepted value and vMix/OBS outputs retain their last accepted value; pivoted
JSON omits the pending field. Full JSON exposes its raw text and rejected state.
Confirmed physical blanks need two readings when empty output is allowed.
Blank, zero, and rejected OCR remain different states.

At 5 readings/s, ordinary score changes add approximately 200 ms after the
first observation; unusual changes add approximately 400 ms. Acquisition and
processing add their own delay. Lower OCR cadence increases confirmation delay.
Fresh OCR is required, so **Skip Similar Image** and the older per-character
**Average Output** vote are bypassed for boxes using confirmation. Template
fields calculated from other boxes are outside this OCR filter.

This reduces isolated or alternating errors; it cannot prove that a persistently
misread digit is correct. Three repeated wrong readings can still be accepted,
and an incorrect initial value can be confirmed. Clock continuity does not prove
game state or recognize a stopped timer independently of OCR. The existing
confidence/format filters and a complete, well-exposed crop remain necessary.

Regression tests cover score increments, corrections, resets, 7/1 transients,
whole-value decisions, blank/zero handling, stopped/fractional clocks, clock
resets, one-Hz startup, unknown interruptions, duplicate frames, source/context
resets, persistence, and camera-local target copies. These are behavioral tests,
not an independently measured accuracy gain on another game or camera session.

A chronological diagnostic also checked 30 frames around the previously found
seven/one problem (clip 3, frames 3480–3654, six-frame steps, complete Home ROI).
Every crop is visually a `7`: see the
[crop sheet](../results/ocr_video_benchmark/temporal_sequence.jpg) and
[raw decisions](../results/ocr_video_benchmark/temporal_sequence.json).
Several consecutive OCR `1`s established a wrong initial value and later kept
it stable; the final sustained `7` readings recovered it through confirmation.
This example demonstrates a limitation, not a measured accuracy improvement.
The feature is therefore explicitly **confirmation**, rather than a guarantee
of correcting seven/one recognition. Independent sequences with actual score
changes, corrections and clock resets are still needed for tuning the policy.
