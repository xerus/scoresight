# OCR comparison on complete scoreboard crops

Date: 2026-09-25

## Findings

The earlier six-frame video comparison is **superseded**: fixed rectangles clipped or missed digits as the camera moved, so its rankings were not reliable. This run uses the four supplied screenshots with individually checked, complete two-slot score regions. The empty Away field is labeled `""`, never zero.

On this development set, **Tesseract `7seg` with LED masking and the OpenCV segment prototype both read 8/8 fields correctly**, including the blank. Switching model files alone is insufficient: preprocessing and geometry substantially change the results. These settings were explored on the same four images, so this is not a held-out accuracy estimate.

## Inputs

| Image | Home | Away | Clock (not benchmarked) |
|---|---:|---:|---|
| home_1_away_0_2:58.png | 1 | 0 | 2:58 |
| home_1_away___2:54.png | 1 | blank | 2:54 |
| home_2_away_2_2:52.png | 2 | 2 | 2:52 |
| home_4_away_2_2:47.png | 4 | 2 | 2:47 |

The manifest in `tools/ocr_benchmark/manifest.json` stores labels and XYWH rectangles. Each score retains both digit positions and excludes neighboring indicators. All recognizers receive the same regions. Only four distinct illuminated digits occur (0,1,2,4); there are no illuminated two-digit scores. Eight fields comprise seven numeric values and one blank.

## Model results

Baseline is a fixed reference configuration, not the current app pipeline: for Tesseract, OEM1/PSM8, ROI Otsu, original crop size, inverted to dark text with a white border. Neural baselines follow their published input transforms. Best explored is selected across the listed settings on these same images; it is optimistically biased. A correct blank contributes one point, so 1/8 can mean recognizing only the blank.

| Model | Baseline correct / 8 | Best explored / 8 | Blank correct? | Best setting |
|---|---:|---:|---|---|
| daktronics | 1 | 5 | no | `oem1/psm8/rgb/scale3` |
| scoreboard_general | 5 | 6 | no | `oem1/psm13/otsu/scale1` |
| scoreboard_general_large | 3 | 7 | no | `oem1/psm13/otsu/scale3` |
| eng | 2 | 6 | yes | `oem1/psm8/led160/scale3` |
| tessdata_best_eng | 3 | 5 | yes | `oem1/psm8/led160/scale1` |
| tessdata_fast_eng | 1 | 3 | yes | `oem1/psm8/led160/scale1` |
| 7seg | 6 | 8 | yes | `oem1/psm8/led160/scale1` |
| ssd | 0 | 7 | no | `oem1/psm7/led160/scale1` |
| ssd_plus | 1 | 7 | no | `oem1/psm7/led160/scale1` |
| ssd_int | 0 | 7 | no | `oem1/psm7/led160/scale1` |
| OICWS_CRNN | 2 | 2 | no | `rgb` |
| Renjith_TFLite | 0 | 1 | no | `rgb/invertTrue` |
| Leander/old/sequence_model_combined.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/old/sequence_model_combined_multi.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/old/sequence_model_combined_multi_v2.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/old/sequence_model_combined_temp_v2.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/old/sequence_model_combined_temp_v3.keras | 0 | 1 | no | `otsu/invertFalse` |
| Leander/old/sequence_model_combined_temp_v4.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/old/sequence_model_multimeter.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/old/sequence_model_sample1_meter_1.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/old/sequence_model_sample2_meter_2.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/old/sequence_model_synthetic_v1.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/old/sequence_model_temp_v5.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/sequence_model_synthetic_V.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/sequence_model_synthetic_V_v2.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/sequence_model_synthetic_mV.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/sequence_model_synthetic_mV_v2.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/sequence_model_synthetic_temp.keras | 0 | 0 | no | `otsu/invertTrue` |
| Leander/sequence_model_temp_v6.keras | 0 | 0 | no | `otsu/invertTrue` |

The OpenCV prototype scores **8/8**, with the blank correct. Its segment zones were adjusted using these images. See [OpenCV details](opencv-benchmark.md).

OICWS **YOLO + CRNN scores 0/8** in each of four explored detector configurations (image size640/1280, confidence0.3/0.05). Detections are assigned to score fields by their centers using the supplied regions; the highest-confidence candidate is recognized with five pixels of padding. A missing detection is rejection (`null`), not a confirmed blank. This is localization plus recognition with known field assignment, not automatic scoreboard interpretation.

## Settings and measurement

- Tesseract: 10 traineddata files, OEM0 and OEM1 attempted; unsupported legacy engines skipped. Four segmentation modes:7 (line),8 (word),10 (character, diagnostic only),13 (raw line). Three input variants and two scales (original/3×). **264 supported configurations**, all predictions saved. Per-image adaptive state reset; dictionaries disabled at initialization; digits-only whitelist. No smoothing, confidence rejection or blank gate.
- `rgb` for Tesseract means grayscale conversion from the RGB source, inverted. `otsu` uses per-ROI Otsu, inverted. `led160` masks illuminated warm LEDs in OpenCV HSV: `(H<40 or H>165) and S>80 and V>160`, then inverts. All Tesseract inputs receive a12-pixel white border after resizing. The app now crops the selected boxes first; Global shares an Otsu threshold computed from those boxes, and the per-box Rescale Input option resizes only that crop to35 pixels high before Tesseract reads it.
- OICWS CRNN: native RGB, Otsu and LED-mask inputs; published64×256 normalization. Raw CTC decoded text is retained, avoiding the upstream validator's default rejection of zero.
- Renjith: grayscale200×31, float0..1, plus Otsu/LED variants and both polarities (6 configurations). Full TensorFlow Lite Flex delegate; output is already decoded indices, mapped with the alphabet used in its training notebook (`digits + lowercase + '.'`).
- Leander: all17 checkpoints, their native input dimensions, unscaled0..255 images, linear resize, Otsu or LED masks, both polarities (68 configurations). Fixed-position output heads; class13 is unmapped in the published alphabet and is retained as `?` (unknown), not credited as blank. This conservative policy differs from upstream silently ignoring unmapped heads; any unknown-containing prediction is invalid for these score labels. An optional leading plus and trailing `.0` normalize numerically; minus signs and malformed/internal signs are retained, never stripped to manufacture a match.
- Total neural configurations:77. Each output retains raw text, normalized prediction, expected value, exact-match flag and exception if any. Blank is `""`; exceptions/missing detections are `null`. Numeric normalization treats `01` and `1.0` as1 but does not turn an empty string into0.
- Scores above are CPU recognition results. App environment: Python3.14, tesserocr linked to Tesseract5.5.1. Neural environment: Python3.11, TensorFlow2.15.1, CPU PyTorch. Recorded sweep durations include Python/preprocessing overhead and are not controlled speed comparisons.

## Reproduce and inspect

From the repository root (external weights and repos currently cached under `/tmp`):

```sh
.venv/bin/python tools/ocr_benchmark/run.py --engine tesseract \
  --manifest tools/ocr_benchmark/manifest.json --output results/ocr_benchmark/tesseract.json
/tmp/ocrbench-venv/bin/python tools/ocr_benchmark/run.py --engine neural \
  --manifest tools/ocr_benchmark/manifest.json --output results/ocr_benchmark/neural.json
/tmp/ocrbench-venv/bin/python tools/ocr_benchmark/detector.py
python tools/ocr_benchmark/summarize.py
.venv/bin/python tools/opencv_benchmark/benchmark.py
```

`--cache-root` changes the root of external model directories for the main runner. The detector script currently uses `/tmp` paths. The JSON result files contain every prediction rather than only the winning configurations. `results/ocr_benchmark/table.md` contains the complete generated model table. The inputs contain colons in filenames and need renaming with manifests updated for Windows.

## Sources and scope

- Bundled ScoreSight `daktronics`, `scoreboard_general`, `scoreboard_general_large`, `eng`: binaries added in the initial upstream commit; their training provenance is not documented.
- [Shreeshrii tessdata_ssd](https://github.com/Shreeshrii/tessdata_ssd): all four supplied models, including7seg. Their age does not predict their suitability for this display.
- Official [tessdata_best](https://github.com/tesseract-ocr/tessdata_best) and [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast): English models.
- [OICWS](https://github.com/OICWS/lcd-digit-recognition): v1.1.0 CRNN and YOLO weights; code is imported from the external checkout, not copied into the app.
- [Leander](https://github.com/2leander2/seven-segment-ocr): all17 cached Keras checkpoints, each listed above.
- [Renjith](https://github.com/renjithsasidharan/seven-segment-ocr): published float16 TFLite model.
- [MiXaiLL76/7SEG_OCR](https://huggingface.co/datasets/MiXaiLL76/7SEG_OCR) is a training dataset, not a ready model; no training was attempted. MATLAB-only examples are not executed. Other OCR families without downloaded models are not covered; “all” here means the collected candidates and the requested OpenCV baseline.

## Next changes supported by the evidence

1. Correct frame/display geometry and confirm blank-output behavior before tuning confidence. See [code findings](ocr-improvement-options.md).
2. Expose an LED intensity/color preprocessing option with preview. Keep a full two-digit region; suppress unlit segments rather than shrinking the box to one character.
3. Try the opt-in **OpenCV Seven-Segment (2 digits, experimental)** model for a calibrated two-cell warm-LED display. It returns separate blank and uncertain states. Fixed brightness and alignment assumptions still fail under some perturbations.
4. Collect fresh, held-out examples of every digit, two-digit scores, true blanks, reflections, exposure changes and camera movement. Freeze settings before evaluating those examples.

The benchmark runner does not alter OCR settings. The app now includes a selectable OpenCV reader; see the OpenCV report for limitations and its focused tests.

## Speed and robustness checks

Warmed CPU measurements (800 calls per engine, including preprocessing, excluding file I/O/imports/model loading): OpenCV median **0.183 ms**, p95 **0.252 ms**;7seg median **4.181 ms**, p95 **7.486 ms** per field. Tesseract used `OMP_THREAD_LIMIT=1`. These are diagnostic measurements on this host, not production throughput guarantees.

| Perturbation of development images | OpenCV correct /8 | 7seg correct /8 |
|---|---:|---:|
| Original |8|8|
| Brightness60% |1|1|
| Brightness80% |8|7|
| Brightness120% |8|8|
| ROI shifted right2px |6|8|
| ROI shifted down2px |8|8|

At60% brightness OpenCV rejects seven fields;7seg returns blank for all eight. Neither should silently export those outcomes as valid current scores. These synthetic perturbations are not independent test data. Raw comparison and timing data are in `tools/opencv_benchmark/results/comparison.json`.

Weight SHA-256 hashes and Python package versions are recorded under `results/ocr_benchmark/` for reproducibility. Third-party weights remain in their external caches and are not redistributed with these scripts.
