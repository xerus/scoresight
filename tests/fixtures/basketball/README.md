# Basketball camera frames

Six full-resolution (1920×1080), lossless PNG frames extracted from the user's
September 26, 2026 recordings in `Videos/Camera`. About 10.4 MiB total. Pixels
are decoded BGR frames written directly as PNG, without resizing, enhancement,
or another lossy compression pass. Full frames retain clocks and scene context
for future crop, tracking, and perspective tests.

`manifest.json` records original video filenames, zero-based frame indices, FPS,
image dimensions and PNG SHA-256 hashes. Each field has a manually reviewed
expected value and an XYWH rectangle in original frame coordinates. Labels were
selected from the existing video/validation manifests and visually rechecked
after extraction. Crops include the complete number area and may include unlit
leading cells; they are not individual-digit calibration boxes.

| Frame | Labeled fields | Purpose |
| --- | --- | --- |
| zero.png | Home 0, Away 0 | Genuine zero, distinct from an unlit display |
| seven_ambiguous.png | Home 7 | Known repeated PP-OCR/TTA 7-to-1 error |
| seven.png | Home 7 | Another observation of the same digit |
| eleven.png | Home 9, Away 11 | Repeated thin digits |
| foul_one.png | Home 24, Away 32, Away fouls 1 | Natural isolated 1, not a score |
| three_digit.png | Home 96, Away 108 | Three-digit basketball score |

The current bundled PP-OCR + TTA reader accepts `1` for the ambiguous `7`, and
rejects `11` and the isolated foul `1`. Their ground-truth labels remain `7`,
`11`, and `1`; wrong model predictions are not labels. The real-model tests
exercise working numeric cases and ensure the thin-digit cases either reject or
read correctly. The ambiguous seven remains an evaluation case for future fixes.

`clock_manifest.json` adds seven reviewed clock crops from these same frames:
five green game clocks and two shot clocks. It records visible digit strings
separately from semantic time values. Green punctuation is not independently
legible under glare, so these labels must not be used to claim separator-reading
accuracy. The main numeric manifest and PNG hashes are unchanged.

These frames come from the existing development session, not an independent
held-out camera/game. They contain no labeled physical blanks or fractional clocks
and are sparse observations rather than a continuous transition sequence.

Re-extract using the original recordings:

```sh
.venv/bin/python tools/ocr_benchmark/extract_fixture_frames.py \
  --videos /path/to/Videos/Camera --output /tmp/basketball-frames
```

The extraction tool verifies decoded frame dimensions and pixels against the
committed PNGs. PNG byte hashes may change with encoder versions.
