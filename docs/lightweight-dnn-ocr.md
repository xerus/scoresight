# Lightweight neural OCR and TTA on the camera recordings

Exploration date: 2026-10-12. The app now offers **PP-OCRv4 + TTA (experimental)**
in the OCR model dropdown. It bundles the model and uses CPU ONNX Runtime directly.

## Recommendation

Try an optional **English PP-OCRv4 mobile recognizer through CPU ONNX Runtime**,
reading existing user-selected fields. The downloaded ONNX file is 7.3 MiB.
Detection and orientation classification are disabled. Keep the model loaded and
limit inference to one CPU thread initially. Recognition takes about 20 ms per
score field on this host; that makes a CPU experiment plausible, but an older
laptop still needs its own measurements.

[MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR) is a useful survey of document
parsers, but its 1.2B/3B models, document layout work, and GPU benchmarks do not
fit small live score fields. Its comparison includes PP-StructureV3, Docling,
MinerU, GOT-OCR, and other document-oriented pipelines; their document rankings
do not establish scoreboard accuracy. Even the newer
[MonkeyOCRv2](https://github.com/Yuliang-Liu/MonkeyOCRv2) is intended for documents.

[PaddleOCR's recognition catalog](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html)
lists English PP-OCRv4 and v5 mobile models at 7.5 MB, general PP-OCRv5 mobile at
16 MB, and PP-OCRv6 tiny at 4.4 MB. The English v4 ONNX model was tested here;
v5 and v6 remain candidates, without local measurements. Published CPU timings
use different hardware and settings. [RapidOCR](https://github.com/RapidAI/RapidOCR)
provides ONNX Runtime and other inference backends. Explicitly select the model:
its current defaults use a different, larger recognition model.

[EasyOCR](https://github.com/JaidedAI/EasyOCR) supports CPU inference, but uses
PyTorch. It is a secondary comparison candidate because of the additional
runtime packaging. A small digit-only CNN/CTC model trained for dotted LEDs is
another possibility if the pretrained model's remaining errors justify collecting
training data. Neither was tested in this exploration.

## What was inspected

The Camera directory has **19 MP4 clips**, all 1920×1080, totaling **1592.157 seconds**
(26 minutes 32 seconds). Three frames per clip were exported at 10%, 50%, and 90%
of frame count. These are sampled observations, not an exhaustive review of every
frame. See the [inventory](../results/ocr_video_benchmark/inventory.json). The
sampler generates a contact sheet locally; full-scene video samples are not
included in the repository.

The scenes contain red scores, a green game clock, another clock/shot-clock display,
camera movement, glow, and strong exposure changes. The final clip contains a
three-digit Away score of **108**, so a fixed two-cell recognizer cannot cover
all these recordings without another geometry mode.

For the initial recognition trial, the midpoint frames of 17 clips were manually
labeled and their full score crops checked visually: **34 fields**, including
zero, one-, two-, and three-digit scores. Clips 9 and 11 in filename order were
excluded because their midpoint frames have heavy glare and uncertain labels.
The labels/rectangles are in [video_manifest.json](../tools/ocr_benchmark/video_manifest.json);
the [crop sheet](../results/ocr_video_benchmark/score_crops.jpg) supports review.
Boxes are per sampled frame, not tracked boxes validated across each video.

No true blanks or clock readings were evaluated. This is one scoreboard/session,
with repeated scores across clips, not 34 independent test cases or a held-out
accuracy estimate. The existing [four-image comparison](ocr-model-benchmark.md)
has different inputs and preprocessing, and should not be compared numerically.

## Baseline recognition

PP-OCRv4 on native BGR crops: **31/34 correct**. Errors:

| Expected | Raw output | Confidence |
| --- | --- | --- |
| 7 | 1 | 0.62315 |
| 37 | 31 | 0.73777 |
| 50 | 5O | 0.61048 |

No character substitution was applied to manufacture a match. Only entirely
numeric strings have leading zeros normalized; invalid strings remain invalid.
Confidence is a model score, not a probability of correctness.

Warmed recognition, including input transforms and decoding, was measured with
10 calls per field: median **19.95 ms**, p95 **23.65 ms**. Model loading, video
decoding, and full app overhead are excluded. Hardware: AMD Ryzen 7 PRO 5850U;
Python 3.11.16, RapidOCR 3.10.0, ONNX Runtime 1.31.0; CPU only, one inference thread.
The result JSON records versions, hashes, raw predictions, and confidence.

Untuned Tesseract reference runs with ROI Otsu/inversion/white border, OEM1/PSM8,
digits whitelist produced 0/34 for bundled scoreboard_general at native and
35-pixel height, and bundled eng at 35-pixel height. These are fixed reference
settings, not an app-wide comparison or a tuned Tesseract optimum. Earlier
7seg plus LED preprocessing results remain relevant; that model was not available
in the current temporary cache and was not rerun here.

## Test-time augmentation (TTA)

All variants use the same model and the same complete crops. Shear transforms
change the apparent slant of the vertical strokes without changing labels;
they can make a vertical seven-segment 7 more recognizable to a general text model.

| Input variant | Correct / 34 |
| --- | --- |
| Native BGR | 31 |
| Inverted grayscale | 33 |
| Inverted grayscale with 8px white border | 23 |
| Native with 8px gray border | 23 |
| Mild unsharp filter | 32 |
| Horizontal stretch ×1.5 | 29 |
| Horizontal shear −0.25 | 33 |
| Horizontal shear +0.25 | 34 |
| Rotation −3° | 32 |
| Rotation +3° | 30 |
| Strict majority across all ten variants | 32, with 1 rejection |

Stretching and both shears recover the isolated **7**; both shears recover **37**.
Ordinary padding, sharpening, inversion, and small rotations do not recover the
isolated 7. Voting across all variants still predicts 1 for it: those errors are
correlated, not independent evidence. The +0.25 shear's perfect result on this
small development set does not justify always applying it to other cameras/fonts.

A selective policy was derived **after examining these results**:

1. Read the native crop.
2. If the result is not entirely numeric, or confidence is below 0.85, run
   inverted grayscale and both shear variants.
3. Require at least two of those three alternatives to agree on a numeric value;
   otherwise return unknown. Do not interpret unknown as zero or blank.

Replaying this policy recovers all three original errors, yielding **34/34** with
fallback on **11/34** fields. Estimated cost from the individual measured calls:
median **20.14 ms**, mean **37.67 ms**, p95 **80.31 ms** per field. This cost is a
sum of calls from the TTA sweep, not an independently timed adaptive implementation.
Running every variant costs about **195 ms** per field and is a poor default for
an older laptop. At ScoreSight's default 5 OCR updates/s, four fields at the
selective policy's development-set mean would consume approximately 753 ms of
inference per second (75% of one core), excluding the rest of the application.
Two score fields would consume about 38% of one core at that cadence. Start
with fewer fields or a lower OCR cadence on an older laptop; actual workloads
may trigger fallback more often than this sample set.

The confidence trigger and transform choices need freezing before evaluation
on fresh clips. Include genuine isolated **1** examples to measure the reverse
1→7 mistake, blanks, clocks, heavy glare, motion, and score transitions. Measure
accepted errors, rejection rate, whole-field accuracy, and update delay. A small
number of samples cannot establish a safe export policy.

## Reproduce

Use an isolated Python 3.11 environment for the neural dependencies; the app's
requirements need no changes. The measured environment used RapidOCR 3.10.0,
ONNX Runtime 1.31.0, NumPy 1.26.4, OpenCV Python/headless 4.11.0.86.

Download the [exact English v4 ONNX model](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx).
Its SHA-256 is recorded in ppocrv4.json and tta.json. Weights are kept outside
the repository.

```sh
VIDEOS=/home/pgrunt@inno.innovatrics.net/Videos/Camera
SAMPLES=/tmp/scoresight-dnn
MODEL=/tmp/scoresight-dnn/en_PP-OCRv4_rec_mobile.onnx
PYTHON=/tmp/scoresight-dnn/venv/bin/python

.venv/bin/python tools/ocr_benchmark/sample_videos.py "$VIDEOS" "$SAMPLES"
"$PYTHON" tools/ocr_benchmark/video_recognition.py \
  --videos "$VIDEOS" --manifest tools/ocr_benchmark/video_manifest.json \
  --engine rapidocr --model "$MODEL" \
  --output results/ocr_video_benchmark/ppocrv4.json
"$PYTHON" tools/ocr_benchmark/video_tta.py \
  --samples "$SAMPLES" --manifest tools/ocr_benchmark/video_manifest.json \
  --model "$MODEL" --output results/ocr_video_benchmark/tta.json
```

The labeled manifest names original video files and frame indices, so the main
recognition runner does not depend on exported images. TTA reads exported images
from the sampler. Raw results are retained in `results/ocr_video_benchmark/`.
Both runners are diagnostic. The app uses the same selective transform policy,
with whole-string agreement and existing confidence/format filters. It reads
original color crops and bypasses Tesseract-specific patch transforms, smoothing,
and similar-image skipping. Empty neural decoding is rejected rather than
exported as a confirmed blank. Model choice is saved in the existing settings.
The direct ONNX app integration also read all 34 development crops correctly;
raw results are in `results/ocr_video_benchmark/app_integration.json`.
