# Chronological basketball crops

203 lossless field crops from seven sequences in the original September 26
camera session. Labels were reviewed from contact sheets; these sequences were
subsequently used during recognition development. They are regression fixtures,
not independent game/camera validation. Source timestamps come from playback;
frame indices are zero-based. The manifest stores source video, fixed ROI,
expected displayed value, and each PNG checksum. Camera movement can clip digits.

Re-extract with decoded-pixel verification:

```sh
.venv/bin/python tools/ocr_benchmark/extract_sequence_fixtures.py --videos /path/to/Videos/Camera --output /tmp/sequence-crops
```

Run tests without the source videos via `test_basketball_sequences.py` (requires
ONNX Runtime and the bundled model). The benchmark reports accepted readings,
retained outputs, unavailable values, update delays and processing timings.
See [measurement details](../../../../docs/led-ambiguity-check.md).
