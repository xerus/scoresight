# Reader diagnostic on committed camera frames

Measured October 13, 2026 using the six committed lossless camera frames. This
is an 11-field, same-session diagnostic, not an independent accuracy estimate.
The PP-OCR policy and LED thresholds were unchanged for this comparison.

| Reader | Accepted correct | Accepted wrong | Rejected |
| --- | --- | --- | --- |
| PP-OCRv4 + selective TTA, confidence >= 0.5 | 8 | 1 | 2 |
| LED reader, default color/brightness, equal-width cells | 0 | 7 | 4 |

The LED comparison uses the existing complete field rectangles, with one cell
for the foul indicator, three for Away 108, and two for other numeric fields.
These crops were checked for complete numbers but were not calibrated to the
LED reader's sample zones. Exposure, margins and geometry produce wrong patterns,
including Home 0 read as 18 and both 24/32 read as 88. This does not measure the
new explicit-cell reader after calibration. It does establish that the default
LED reader cannot serve as an automatic verifier for these neural crops.

PP-OCR still accepts 1 for the ambiguous 7 and rejects score 11 and foul 1.
Confirmation cannot reliably correct a consistently wrong accepted value.

Seven clock fields were also visually labeled in `clock_manifest.json`: five
green game-clock crops and two red shot clocks. Raw observed digit labels and
semantic time labels are separate because green punctuation is not independently
legible in these crops. PP-OCR returns `720`, `642`, and `553` for three game
clocks; these lack the separators required by the basketball Time format. Two
game clocks and shot clock 17 are rejected. Shot clock 14 is read as 4 at about
0.916 confidence. A high-confidence result alone does not establish completeness.

Raw results include predictions, latency, package versions and model/manifest
hashes in [fixture_readers.json](../results/ocr_video_benchmark/fixture_readers.json).
Warmed median timings exclude model loading, video/image decoding and full app
overhead; they are diagnostic timings on this host.

```sh
.venv/bin/python tools/ocr_benchmark/fixture_readers.py \
  --output /tmp/fixture-readers.json
```

Next recognition work should calibrate cells/crops on a development subset,
freeze those settings, and evaluate additional frames and transitions. Capture
all digits, real isolated 1/7, fractional clocks, blank displays, glare and
motion. Another game/camera session is needed for independent evaluation. If
calibrated geometry remains unreliable, use the reviewed crops to develop a
digit recognizer rather than add unmeasured model substitutions.
