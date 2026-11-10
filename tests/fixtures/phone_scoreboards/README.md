# Phone scoreboard samples

Twelve lossless, unscaled crops from four user-supplied phone videos in
`~/Videos/scoresight`, selected after reviewing all 15 photos and six videos.
The original media stays outside the repository. The manifest records zero-based
frame indices, source dimensions, XYWH coordinates, source SHA-256 checksums,
and visually reviewed labels. Coordinates refer to OpenCV-decoded frames.

These samples add white scores and dark digits on a green panel, two leading-zero
clocks, perspective and motion blur, and a distant red LED three-digit score.
Equipment photos and photographs of computer screens were omitted from OCR
fixtures. They offer little clean ground truth compared with the direct views.

Initial PP-OCRv4 results with `check_led=True`:

| Source (frame 30) | Home | Away | Clock |
| --- | --- | --- | --- |
| `20261031_162012.mp4` | 0 | 0 | 39:19 |
| `20261031_172951.mp4` | **37 read as 87** | **12 rejected** | 05:22 |
| `20261031_173008.mp4` | 39 | 12 | 05:10 |
| `20260926_200212.mp4` | 92 | 100 | 3:54 |

All labels come from visual inspection, independently of OCR predictions.
The blurred 37 is retained as an explicit expected-failure regression: a correct
reading or rejection will produce an unexpected success, prompting removal of
the decorator. The blurred 12 may be rejected but must never become another
number. The remaining ten crops require correct recognition with confidence
at least 0.5. These are development examples, not an independent accuracy score.

Run the tests without original media:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_basketball_camera_ocr.py -v
```

Verify source checksums and reproduce every crop in a separate directory:

```sh
.venv/bin/python tools/ocr_benchmark/extract_phone_fixtures.py \
  --videos ~/Videos/scoresight --output /tmp/scoresight-phone-fixtures
```

Different video decoders may produce slightly different pixels; the extractor
reports differences instead of silently replacing the committed fixtures.
