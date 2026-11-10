# Scoreboard accuracy experiment artifacts

These versioned artifacts preserve the 2026-11-10 comparison independently of
temporary-directory cleanup. They are research candidates; the application
continues to use `models/en_PP-OCRv4_rec_mobile.onnx`.

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `ctc_head.onnx` | Fine-tuned final classification layer; 11,737 changed parameters | `b419dffe2be0db20aef52bdceaecba8cc3fad8b8dcb302d19b255ccb13a8e7c4` |
| `bias_edit.onnx` | Selected bias-edit candidate; original parameter values retained | `9b05c3056f6752f75d0f682adff602fada9b1a631f218c14f5df2ab3cc7d89c9` |
| `selection.json` | Candidate search and selection frozen before October inference | `d31725115ec8d13cf7ed99388789d67e035722a8ccebf4cd8d04afc33b0c3bc1` |

The bias-edit file has a different serialized hash from the bundled model
because the export rewrites the head's Constant tensors; its numerical weights
are unchanged. Keeping the exact export preserves the original run's provenance.

Both ONNX files derive from the PaddleOCR English PP-OCRv4 mobile recognizer
distributed by RapidAI/RapidOCR. The original model's attribution is in
[`models/NOTICE.md`](../../NOTICE.md), and its Apache-2.0 license is preserved in
[`models/LICENSE`](../../LICENSE). The fine-tuned candidate modifies only the
final linear weights and bias; the visual backbone is unchanged.

See the [experiment report](../../../docs/weight-space-accuracy.md) for training,
evaluation, limitations, and reproduction instructions. The gains concern two
views of the same score; the blurred `37` to `87` error remains. These candidates
are not validated as replacements for general English text recognition.
