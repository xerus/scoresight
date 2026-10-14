"""Compare bundled readers on committed camera crops, without source videos.

This is a development-set diagnostic. LED geometry is deliberately untuned:
numeric ROIs use equal-width cells and the existing brightness/color defaults.
Clock results report observed strings separately from semantic time labels.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from defaults import FieldType  # noqa: E402
from dnn_ocr import DnnOcr  # noqa: E402
from seven_segment import read_score  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    fixtures = ROOT / "tests/fixtures/basketball"
    frames = json.loads((fixtures / "manifest.json").read_text())["frames"]
    clocks = json.loads((fixtures / "clock_manifest.json").read_text())["fields"]
    images = {row["image"]: cv2.imread(str(fixtures / row["image"])) for row in frames}
    if any(image is None for image in images.values()):
        raise ValueError("Cannot load a camera fixture")
    reader = DnnOcr(str(ROOT / "models/en_PP-OCRv4_rec_mobile.onnx"))

    def timed(call):
        call()  # warm up; exclude model load, image decode and first call
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = call()
            durations.append((time.perf_counter() - start) * 1000)
        return result, round(statistics.median(durations), 3)

    numeric = []
    for row in frames:
        for field in row["fields"]:
            x, y, w, h = field["roi"]
            crop = images[row["image"]][y : y + h, x : x + w]
            (text, confidence, _), latency = timed(lambda: reader.recognize(crop))
            # Explicit physical digit counts for these reviewed fields; not a
            # decoder-selected count based on its own predicted string.
            count = 1 if field["name"] == "away_fouls" else 2
            if row["image"] == "three_digit.png" and field["name"] == "away":
                count = 3
            (led_text, _, _), led_latency = timed(
                lambda: read_score(crop, digit_count=count)
            )
            text = (
                text.lstrip("0") or "0"
                if text and text.isascii() and text.isdigit()
                else text
            )
            numeric.append(
                {
                    "image": row["image"],
                    "field": field["name"],
                    "expected": field["expected"],
                    "ppocr": {
                        "text": text,
                        "confidence": confidence,
                        "accepted": text is not None and confidence >= 0.5,
                        "median_ms": latency,
                    },
                    "led": {
                        "text": led_text,
                        "digit_count": count,
                        "median_ms": led_latency,
                    },
                }
            )
    clock_readings = []
    for field in clocks:
        x, y, w, h = field["roi"]
        crop = images[field["image"]][y : y + h, x : x + w]
        (text, confidence, _), latency = timed(
            lambda: reader.recognize(crop, FieldType.TIME)
        )
        clock_readings.append(
            dict(
                field,
                text=text,
                confidence=confidence,
                median_ms=latency,
                exact_displayed_time_match=text == field["displayed_time"],
            )
        )
    summary = {}
    for engine in ("ppocr", "led"):
        accepted = [
            r
            for r in numeric
            if r[engine]["text"] is not None and r[engine].get("accepted", True)
        ]
        summary[engine] = {
            "correct": sum(r[engine]["text"] == r["expected"] for r in accepted),
            "wrong": sum(r[engine]["text"] != r["expected"] for r in accepted),
            "rejected": len(numeric) - len(accepted),
        }
    output = {
        "scope": "Same-session development fixtures; LED geometry/default thresholds untuned. Not an independent accuracy estimate.",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("onnxruntime", "numpy", "opencv-python")
        },
        "sha256": {
            name: hashlib.sha256(file.read_bytes()).hexdigest()
            for name, file in {
                "model": ROOT / "models/en_PP-OCRv4_rec_mobile.onnx",
                "numeric_manifest": fixtures / "manifest.json",
                "clock_manifest": fixtures / "clock_manifest.json",
            }.items()
        },
        "repeats": args.repeats,
        "summary": summary,
        "numeric": numeric,
        "clocks": clock_readings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
