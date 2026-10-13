# Next performance experiments

Explored 2026-10-12 against the current PP-OCRv4 + selective TTA implementation.
The initial investigation added a benchmark and measurements. The follow-up
[numeric validation](numeric-padding-validation.md) checked 71 fresh crops from
the supplied recordings; the app now uses width 128 for numeric fields and keeps
width 320 for clocks/text. For accuracy, collect independent transition and blank
examples before tuning another model or changing the TTA policy.

## Measured CPU options

The original recognizer resized each score to height 48 while preserving aspect
ratio, then pads its input to at least width 320. Most tested score crops use only
a small part of that width. The ONNX model exposes dynamic batch/width dimensions.
The diagnostic runner changes only the minimum padding width or thread count,
retaining the model, normalization, greedy decoding, confidence trigger (0.85),
three fallback transforms and two-vote acceptance policy.

Full warmed `recognize()` calls were timed, including transforms and decoding.
Model loading and video decoding were excluded. Hardware: AMD Ryzen 7 PRO 5850U,
Linux x64, Python 3.11; dependency versions and model/manifest hashes are recorded
in [raw results](../results/ocr_video_benchmark/performance_options.json).
Five calls per crop were measured after a warm-up, in the table's order.

| Minimum width | CPU threads | Native correct | With TTA | Fallback fields | Mean ms | Median ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 320 (baseline) | 1 | 31/34 | 34/34 | 11/34 | 39.53 | 21.49 | 91.01 |
| 256 | 1 | 32/34 | 34/34 | 11/34 | 34.62 | 19.37 | 76.82 |
| 192 | 1 | 32/34 | 34/34 | 10/34 | 23.04 | 12.55 | 50.04 |
| 160 | 1 | 33/34 | 34/34 | 10/34 | 19.02 | 10.71 | 42.64 |
| 128 | 1 | 33/34 | 34/34 | 12/34 | 16.16 | 8.62 | 32.38 |
| 320 | 2 | 31/34 | 34/34 | 11/34 | 27.41 | 14.19 | 61.16 |

Width 128 reduced the measured mean latency by about **59% (2.45× faster)**.
The reverse-order run reproduced the ranking: width 320/one thread averaged
37.16 ms, width 128 averaged 14.93 ms (2.49× faster), and all configurations again
read 34/34 correctly with TTA. Absolute timings varied between runs, so their
precision should not be interpreted as a guaranteed deadline.
At 5 OCR updates/s with four fields, the width-320 mean corresponds to about
791 ms of sequential recognition per second; width 128 corresponds to 323 ms.
These are workload estimates, excluding video capture, GUI and outputs, rather
than whole-app CPU measurements. Two threads reduce elapsed latency but can use
more CPU time across cores; they are not the first choice for an older laptop.
[ONNX Runtime's threading guidance](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
describes the tradeoff between thread pools, spinning and CPU contention.

**This does not establish a production-safe width.** These are the same 34
development crops previously used to choose TTA, from one scoreboard/session.
There are no clock, blank or isolated genuine `1` labels. Changing padding also
changes native predictions and confidence, so even preserving 34/34 can alter
acceptance on other inputs. More repeated timing calls do not create independent
accuracy samples. The current maximum of minimum width and actual resized width
must be preserved for longer fields. Keep width 320 for general text until its
own validation passes. PaddleOCR's
[recognition preprocessing](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/infer/predict_rec.py)
uses a model-specific recognition shape and batch aspect ratio; a smaller width
here is an experiment with this particular ONNX export.

## Prioritized follow-ups

| Priority | Option | Why it could help | Evidence and required check |
| --- | --- | --- | --- |
| 1 | Short numeric input width of 128–160 | Avoid processing unused right padding on every native/TTA call | Measured above; freeze width before fresh validation, repeat packaged Windows inference and test an actual old laptop |
| 2 | Recognize unchanged scores less often | Scores typically change much less often than the default 5 Hz cadence | Current DNN path bypasses `skip_similar_image`; expected saving depends on video, unmeasured |
| 3 | Whole-value temporal confirmation | Suppress single-frame glare/multiplexing errors and disputed 7/1 readings | DNN currently clears the existing character smoother and makes independent decisions; unmeasured accuracy and transition delay |
| 4 | Explicit LED activity / blank check | Distinguish an unlit field from OCR failure and confidently wrong text | Current empty DNN decoding is rejected; calibration must cover red scores, green clocks, saturated centers and glare |
| 5 | Batch the three fallback variants | Fewer inference calls and potentially better kernel throughput | ONNX input permits dynamic batch; no batched latency/accuracy measurement in this investigation |
| 6 | INT8 model experiment | Potential smaller weights and faster CPU kernels | Not measured; needs representative calibration, unsupported-operator checks and Windows smoke coverage |
| 7 | Fine-tune a digit/clock recognizer | Improve dotted LED domain fit if errors remain after pipeline changes | Needs independent labeled data; training solely on these 34 crops would overfit |

For unchanged-image reuse, cache only a successful reading alongside its crop,
confidence and timestamp. A crop difference check must not extend an unknown or
failed reading indefinitely. Compare LED activity and local structure, since
small mean pixel differences can conceal the extra segment distinguishing `7`
from `1`. Force periodic OCR, invalidate on model/source/ROI/settings changes,
and expose the age of reused data. Test real score transitions, corrections,
blanking and camera motion; measure both missed transitions and saving. Lowering
cadence is available now and directly reduces inference work, with increased
response delay. Scores and clocks need separate cadence policies.

For temporal confirmation, vote over complete observed strings, with a bounded
window or consecutive agreement. Do not synthesize values by per-character
majorities and do not enforce monotonic scores: corrections and resets occur.
At 5 Hz, requiring two consecutive observations can add approximately 200 ms
after the first observation; acquisition timing adds up to another interval.
Unknown must not become zero, and stale output age needs an explicit policy.

For quantization, ONNX Runtime distinguishes static calibration from dynamic
scaling and warns that accuracy may change; dynamic scaling also adds inference
work. Start with a representative static INT8 comparison rather than assuming
a file-size reduction equals a latency reduction. Compare raw output,
confidence, TTA trigger frequency, accepted errors and update latency against
FP32 on the actual old CPU.
[ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
also describes debugging activation/weight differences and CPU instruction-set
constraints. A new inference backend or larger general document model has lower
priority than the measured padding improvement for this workload.

## Validation before enabling an option

1. Freeze the candidate width, TTA policy and thresholds using these development
   crops. Split future data by recording/session or camera before tuning; nearby
   frames from the same score must stay in one split.
2. Label complete score/clock ROIs in separate clips, including isolated `1`
   and `7`, `17`/`71`, three-digit scores, zero, true blank, clock punctuation,
   glare, motion and score/clock transitions. Record uncertainty explicitly.
3. Report whole-field accepted errors, rejection/coverage, blank false positives,
   7→1 and 1→7 counts separately. Retain raw strings; leading-zero normalization
   applies only to numeric strings. Include confidence changes and TTA frequency.
4. Measure warmed median/p95 and complete-app responsiveness on an old Windows
   laptop for two and four active fields; include output update delay and stale
   readings. Timings on this Ryzen cannot certify old-laptop responsiveness.
5. Exercise the packaged executable outside the source directory using the same
   bundled model and a representative crop suite. Width changes do not require
   another model dependency, but their numerical behavior still needs checking
   on the shipped ONNX Runtime and Windows CPU provider.

## Reproduce

The sampler output used here is `/tmp/scoresight-dnn`; it can be regenerated from
the original Camera clips using the existing sampler. No new dependency is
required beyond the app's DNN dependencies.

```sh
.venv/bin/python tools/ocr_benchmark/sample_videos.py \
  /home/pgrunt@inno.innovatrics.net/Videos/Camera /tmp/scoresight-dnn
.venv/bin/python tools/ocr_benchmark/performance_options.py \
  --samples /tmp/scoresight-dnn \
  --manifest tools/ocr_benchmark/video_manifest.json \
  --model models/en_PP-OCRv4_rec_mobile.onnx \
  --output results/ocr_video_benchmark/performance_options.json
```

Use `--reverse --repeats 3` with another output filename to check configuration
order effects. This investigation also retains that
[reverse-order timing run](../results/ocr_video_benchmark/performance_options_reverse.json).
