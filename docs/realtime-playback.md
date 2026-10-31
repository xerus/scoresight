# File playback and fresh OCR output

File playback now follows source timestamps against a monotonic clock. Slow OCR
causes overdue frames to be skipped rather than accumulating playback delay.
Pause/resume, speed changes, reconnects and looping reset the playback anchor.
Playback speed multiplies source time instead of adding an extra frame jump.
When timestamps are unavailable, frame position and source FPS provide a fallback.

Preview updates no longer wait for OCR cadence and target up to 30 updates/s.
Actual preview rate still depends on decoding, transforms and OCR, which share
one worker. Binary preview updates require OCR. Camera/stream capture is not
restructured by this file-playback change. Processing-rate measurements now use
total elapsed time, including intervals longer than a second.

For the tested scoreboard, select PP-OCRv4 + TTA, enable LED checks, set Time to
MSS input and request 5 OCR updates/s. Two updates/s, as in the saved user
configuration, can add up to half a second of sampling delay before recognition.
The Basketball button does not change cadence or reader selection.

A 10-second replay with three fields requested at 5 Hz showed:

| Camera loop | Wall time | Source time advanced | Last OCR completion age |
| --- | --- | --- | --- |
| Previous | 9.64 s | 4.20 s | 5.44 s |
| Timestamp-paced | 9.84 s | 9.60 s | 0.24 s |

With timestamp pacing, median age was 0.19 s and maximum 0.35 s on this machine.
This measures processing completion against the source playback clock, excluding
startup, confirmation holds and GUI/network delivery. Rejected reads can still
leave the last confirmed output visible; these measurements are not a guarantee
that every displayed clock value is current. No timer values are extrapolated.
Raw observations and the same-session sequence regression summaries are in
[validation results](../results/ocr_video_benchmark/realtime_glare_validation.json).

## Software glare suppression

For Time fields with LED checks enabled, failed green-core retries can try two
less aggressive brightness masks, 220 and 200, after the original 240 mask.
Both extra neural passes must agree, reach 0.80 confidence each, and satisfy the
original digit-shape constraints. Only spaces in these numeric mask readings
are removed; missing digits and letters are never substituted. Two extra native
passes bound the additional work; scores and general text retain the old policy.

Two reviewed development crops now recover 8:01 and 7:27. The original 203
sequence samples retain 140 accepted correct, zero accepted wrong and 63 rejected
readings. These repeated development examples are not independent accuracy
validation. Exposure saturation and physically missing LED segments can still
be unrecoverable. Increasing OCR retries without bounding their cost would
undermine output freshness.

Reproduce actual worker timing:

```sh
PYTHONPATH=src .venv/bin/python tools/ocr_benchmark/realtime_playback.py --video /path/to/recording.mp4 --output /tmp/playback.json
```

The benchmark's default ROIs and starting frame are for the tested clip
`Recording from 2026-09-26 18-02-24.926611.mp4`. Use the fixture re-extraction tool
with `tests/fixtures/basketball/glare/manifest.json` to verify the new crops.
