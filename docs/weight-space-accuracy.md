# Scoreboard weight editing and final-layer fine-tuning

Experiment: 2026-11-10. **Final-layer fine-tuning recovered two additional
October crops; the tested direct bias edits did not improve validation.**
This is preliminary evidence for adapting the recognizer to scoreboard imagery.
The bundled application model has not been replaced.

The [weight-space programming article](https://rikiyat.github.io/blog/weight-space-programming/)
motivated the comparison. Our direct edit tests one narrow hypothesis: an output
bias against `7` relative to `1`. It does not reproduce the article's circuit
analysis or establish the limits of more sophisticated weight editing.

## Models and selection

- Baseline: bundled English PP-OCRv4 ONNX, SHA-256
  `e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318`.
- Direct edit: add 0.25, 0.5, 1, 2, or 4 to the `7` output bias and subtract the
  same amount from `1`. Retain the original if no edit improves validation.
  The original won: the selected candidate changes zero parameters.
- Fine-tuned head: freeze the visual backbone; train the existing 120-by-97
  classification matrix and 97 biases using CTC sequence loss. All 11,737 head
  parameters changed. This is ordinary gradient-based fine-tuning, not
  gradient-free weight-space programming, and not full-network fine-tuning.

Training uses 47 crops from two September source videos, with four image
variants per crop (188 training examples): native, inverted grayscale, and the
existing two shears. Repeated sequence frames are capped at four per value per
sequence. Labels include genuine `11`, `7`, multiple-digit scores, compact
clock digits, and blank shot-clock frames. Clock targets contain the observed
digits; application clock layout supplies punctuation, rather than training
the network to invent a colon.

Validation uses 72 crops from seven other September source videos. Sources
never cross the training/validation/evaluation split; September is still the
same recording session. Selection maximizes source-video macro-average correct
whole fields, then minimizes macro-average wrong fields. Each source has equal
weight in selection, so the 54-frame clock sequences do not outweigh six other
videos. Native predictions use a 0.5 confidence threshold and the application's
explicit clock formatting. Search: learning rates 0.001/0.003, checkpoints at
5/15/40 epochs, seed 20261110, Adam, batches of 16, gradient clipping, and a
0.01 penalty anchoring weights to the original head.

Selected checkpoint: learning rate 0.001, epoch 40. Selection is written before
October crop inference. October labels/fixtures were already reviewed during
earlier development: this is a session-separated exploratory evaluation, not
a newly collected blind benchmark. An initial diagnostic run was corrected to
use compact-clock digit targets, application clock formatting, and selection
balanced by source video. The retained JSON describes the corrected experiment;
neither the search grid nor image transforms were tuned to October outcomes.

## Results

All tables count complete field values, not individual characters. Scores are
normalized only for leading zeros. Clock values use the explicit layout from
their fixture. A valid nonempty value with confidence at least 0.5 is accepted.
Unknown/empty output is rejected, never converted to zero or a confirmed blank.
These are per-crop recognition results, without temporal publication filtering.

October evaluation, **with the existing selective TTA and LED checks**:

| Model | Correct | Wrong | Rejected | Total |
| --- | ---: | ---: | ---: | ---: |
| Original | 19 | 1 | 3 | 23 |
| Selected bias edit (original retained) | 19 | 1 | 3 | 23 |
| Fine-tuned head | 21 | 1 | 1 | 23 |

The two recovered crops are `panel_scoreboard/away_0080.png` and
`away_0327.png`, both displaying **17**. Both improvements belong to one
recording and one unchanged score; they are not two independent generalization
successes. The blurred phone **37 still becomes 87**, and the blurred **12
remains rejected**. The other October phone examples are unchanged.

| October source | Original: correct / wrong / rejected | Fine-tuned: correct / wrong / rejected |
| --- | --- | --- |
| Panel 17:48:28, 14 crops | 12 / 0 / 2 | 14 / 0 / 0 |
| Phone 16:20:12, 3 crops | 3 / 0 / 0 | 3 / 0 / 0 |
| Phone 17:29:51, 3 crops | 1 / 1 / 1 | 1 / 1 / 1 |
| Phone 17:30:08, 3 crops | 3 / 0 / 0 | 3 / 0 / 0 |

September validation, **with TTA and LED checks**:

| Model | Correct | Wrong | Rejected | Total |
| --- | ---: | ---: | ---: | ---: |
| Original | 42 | 0 | 30 | 72 |
| Fine-tuned head | 53 | 0 | 19 | 72 |

The validation gain consists of repeated clock observations from one source.
Both models also recover both glare stress crops with LED checks. Stress crops
are excluded from training and candidate selection, but share a training source
video and are not independent evaluation data.

**Keep the LED checks.** On September validation, the fine-tuned head with
ordinary TTA but without LED checks accepts 13 wrong values (baseline: 25).
Its native-only predictions accept 14 wrong values. On October native-only
recognition, the tuned model accepts one wrong value while the original rejects
that crop. Better training loss does not eliminate confident recognition errors.
General English text accuracy was not evaluated; this candidate is intended for
scores/clocks, not as a blanket replacement for text-field recognition.

## Verification and artifacts

- CTC implementation checked against exhaustive alignment enumeration and
  finite-difference gradients, including blanks and repeated characters.
- Extracted final-layer features reconstruct original ONNX probabilities within
  absolute tolerance 0.00002; exported candidates pass the ONNX checker.
- Full application regression suite: **147 tests, OK, one expected failure**
  (the existing blurred `37` case). Ruff lint and formatting checks pass.
- Candidate substituted into the 15 real-camera/sequence tests: 12 pass,
  one expected failure, two failures asserting particular fallback paths.
  Those failures occur after the correct `7`/`741` value and confidence checks
  pass: the candidate no longer needs `tight_`/`core_` variants. The second
  test's later thin-digit loop is consequently not reached, but those crops
  are included in the experiment's field-by-field evaluation. The candidate
  passes ROI-shift, score-transition, stopped-clock, and blank-reset checks.

The [full result](../results/ocr_video_benchmark/weight_space_accuracy.json)
includes every prediction, acceptance outcome, source, split, confidence, search
candidate, and input/code/model hash. Candidate-test details are in
[weight_space_candidate_checks.json](../results/ocr_video_benchmark/weight_space_candidate_checks.json).
The tuned model is retained locally at
`/tmp/scoresight-weight-accuracy/ctc_head.onnx`, SHA-256
`b419dffe2be0db20aef52bdceaecba8cc3fad8b8dcb302d19b255ccb13a8e7c4`.
Temporary model files need regeneration after cleanup/reboot.

## Reproduce

Use the normal app environment plus the experiment-only `onnx` dependency.
The phone scoreboard fixtures must be available as well as the basketball and
panel fixtures; the result records exact manifest/image hashes.

```sh
uv pip install --python .venv/bin/python onnx==1.23.2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tools/ocr_benchmark/weight_space_accuracy.py \
  --output results/ocr_video_benchmark/weight_space_accuracy.json \
  --artifacts /tmp/scoresight-weight-accuracy
.venv/bin/python -m unittest discover -s tests -p test_ctc_head.py -v
PYTHONPATH=src QT_QPA_PLATFORM=offscreen .venv/bin/python \
  -m unittest discover -s tests -v
.venv/bin/prek run --all-files
```

The next accuracy experiment should add independently labeled sessions with
blurred `3`/`8`, genuine isolated `1`/`7`, different display colors/fonts,
transitions, and true blanks. Freeze this checkpoint before collecting and
scoring those examples. That will establish whether to expand head training
or fine-tune the visual backbone. The current experiment supports that direction
but does not yet justify changing the bundled default.
