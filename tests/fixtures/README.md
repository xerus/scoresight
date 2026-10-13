# PP-OCRv4 Windows smoke-test input

`ppocr_seven.png` is a complete Home score crop from the supplied Camera video
`Recording from 2026-09-26 18-02-24.926611.mp4`, at the frame index recorded for
`clip03_1.png` in `tools/ocr_benchmark/video_manifest.json` (XYWH 140,445,120,90).

The native PP-OCRv4 English mobile model reads this seven as `1`; selective TTA
recovers `7`. Windows CI supplies the crop to the frozen app's `--check-imports`
probe and requires both `7` and four recognition passes. This verifies actual
ONNX inference and the TTA path, rather than only loading the DLL or model.
