"""Regression comparison on previously reviewed same-session labeled frames."""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dnn_ocr import DnnOcr  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    fixtures = json.loads(
        (ROOT / "tests/fixtures/basketball/manifest.json").read_text()
    )["frames"]
    development = {
        (f["video"], f["frame"], field["name"])
        for f in fixtures
        for field in f["fields"]
    }
    rows = {}
    for name in ("video_manifest.json", "validation_manifest.json"):
        for row in json.loads((ROOT / "tools/ocr_benchmark" / name).read_text()):
            key = (row["video"], row["frame"], row["field"])
            if key not in development:
                rows[key] = row
    reader = DnnOcr(str(ROOT / "models/en_PP-OCRv4_rec_mobile.onnx"))
    results, frame, previous = [], None, None
    for key, row in sorted(rows.items()):
        frame_key = key[:2]
        if frame_key != previous:
            capture = cv2.VideoCapture(str(args.videos / row["video"]))
            try:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame"]):
                    raise ValueError(f"Cannot seek to {frame_key}")
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode {frame_key}")
            finally:
                capture.release()
            previous = frame_key
        x, y, w, h = row["roi"]
        if (
            min(x, y) < 0
            or min(w, h) < 1
            or x + w > frame.shape[1]
            or y + h > frame.shape[0]
        ):
            raise ValueError(f"Invalid crop: {row}")
        crop = frame[y : y + h, x : x + w]
        result = dict(row)
        for mode, enabled in (("baseline", False), ("checked", True)):
            reader.recognize(crop, check_led=enabled)
            durations = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                text, confidence, readings = reader.recognize(crop, check_led=enabled)
                durations.append((time.perf_counter() - start) * 1000)
            if text and text.isascii() and text.isdigit():
                text = text.lstrip("0") or "0"
            result[mode] = {
                "text": text,
                "confidence": confidence,
                "accepted": text is not None and confidence >= 0.5,
                "median_ms": round(statistics.median(durations), 3),
                "readings": readings,
            }
        results.append(result)
    summary = {}
    for mode in ("baseline", "checked"):
        accepted = [r for r in results if r[mode]["accepted"]]
        summary[mode] = {
            "accepted_correct": sum(r[mode]["text"] == r["expected"] for r in accepted),
            "accepted_wrong": sum(r[mode]["text"] != r["expected"] for r in accepted),
            "rejected": len(results) - len(accepted),
            "median_field_ms": round(
                statistics.median(r[mode]["median_ms"] for r in results), 3
            ),
        }
    output = {
        "scope": "Same-session regression corpus used during policy development; excludes the 11 original numeric fixture fields. Not independent validation.",
        "policy": {
            "revision": 4,
            "white_core_min": {"warm": 160, "green": 240},
            "reject_horizontal_border_clipping": True,
            "minimum_height_fraction": 0.4,
            "thin_aspect_max": 0.32,
            "wide_aspect_max": 0.85,
            "segment_occupancy": 0.25,
            "tight_padding": 3,
            "require_neural_shape_agreement": True,
            "enclosed_holes": {"0": 1, "6": 1, "8": 2, "9": 1},
        },
        "sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/led_digit_evidence.py",
                "src/dnn_ocr.py",
                "src/seven_segment.py",
                "models/en_PP-OCRv4_rec_mobile.onnx",
            )
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("onnxruntime", "opencv-python", "numpy")
        },
        "repeats": args.repeats,
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
