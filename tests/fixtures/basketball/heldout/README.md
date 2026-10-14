# Frozen-policy score samples

Ten new crops from five distinct videos, labeled
visually before inference after freezing the LED policy. All are from the same
display/session and near previously reviewed frames; they cannot establish
another-camera accuracy. Manifest hashes identify the frozen implementation.

```sh
.venv/bin/python tools/ocr_benchmark/heldout_led_check.py --output /tmp/heldout.json
.venv/bin/python tools/ocr_benchmark/extract_sequence_fixtures.py --manifest tests/fixtures/basketball/heldout/manifest.json --videos /path/to/Videos/Camera --output /tmp/heldout-crops
```
