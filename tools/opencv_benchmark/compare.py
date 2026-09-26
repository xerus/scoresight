"""Compare development-selected OpenCV and Tesseract readers on identical crops."""

import argparse
import json
import os
from pathlib import Path
import time

# Bound OpenMP before importing the OCR library; this is a CPU diagnostic.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2
import numpy as np
from PIL import Image
import tesserocr

from benchmark import ROOT, read_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-directory", type=Path, default=Path("/tmp/scoresight-ssd-models")
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/comparison.json"
    )
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
    frames = {
        sample["image"]: cv2.imread(str(ROOT / "images" / sample["image"]))
        for sample in manifest
    }
    api = tesserocr.PyTessBaseAPI(init=False)
    api.InitFull(
        path=str(args.model_directory),
        lang="7seg",
        oem=1,
        configs=[],
        variables={"load_system_dawg": "0", "load_freq_dawg": "0"},
    )
    api.SetVariable("tessedit_char_whitelist", "0123456789")
    api.SetPageSegMode(8)

    def tess_read(crop):
        h, s, v = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
        led = (((h < 40) | (h > 165)) & (s > 80) & (v > 160)).astype(np.uint8) * 255
        image = cv2.copyMakeBorder(
            255 - led, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255
        )
        api.ClearAdaptiveClassifier()
        api.SetImage(Image.fromarray(image))
        return api.GetUTF8Text().strip()

    readers = {
        "opencv_geometry": lambda crop: read_score(crop)[0],
        "tesseract_7seg_oem1_psm8_led160_scale1": tess_read,
    }
    conditions = [
        ("original", 1, 0, 0),
        ("brightness_60pct", 0.6, 0, 0),
        ("brightness_80pct", 0.8, 0, 0),
        ("brightness_120pct", 1.2, 0, 0),
        ("roi_right_2px", 1, 2, 0),
        ("roi_down_2px", 1, 0, 2),
    ]
    result = {
        "repeats_per_field": args.repeats,
        "warmup_passes": 10,
        "timing_scope": "preprocessing+recognition+decoding; excludes imports, model load, IO, crop extraction and perturbation",
        "omp_thread_limit": os.environ["OMP_THREAD_LIMIT"],
        "tesseract": tesserocr.tesseract_version(),
        "latency": {},
        "sensitivity": [],
    }
    original_crops = []
    for condition, gain, dx, dy in conditions:
        records = {name: [] for name in readers}
        for sample in manifest:
            frame = frames[sample["image"]]
            for field in ("home", "away"):
                spec = sample[field]
                x, y, w, h = spec["roi"]
                crop = np.clip(
                    frame[y + dy : y + dy + h, x + dx : x + dx + w].astype(float)
                    * gain,
                    0,
                    255,
                ).astype(np.uint8)
                if condition == "original":
                    original_crops.append(crop)
                for name, reader in readers.items():
                    prediction = reader(crop)
                    records[name].append(
                        {
                            "image": sample["image"],
                            "field": field,
                            "expected": spec["expected"],
                            "prediction": prediction,
                            "correct": prediction == spec["expected"],
                        }
                    )
        for name, readings in records.items():
            result["sensitivity"].append(
                {
                    "engine": name,
                    "condition": condition,
                    "correct": sum(r["correct"] for r in readings),
                    "total": len(readings),
                    "rejected": sum(r["prediction"] is None for r in readings),
                    "blank_false_positives": sum(
                        r["expected"] == "" and r["prediction"] not in ("", None)
                        for r in readings
                    ),
                    "readings": readings,
                }
            )
    for name, reader in readers.items():
        for _ in range(10):
            for crop in original_crops:
                reader(crop)
        timings = []
        for _ in range(args.repeats):
            for crop in original_crops:
                start = time.perf_counter_ns()
                reader(crop)
                timings.append((time.perf_counter_ns() - start) / 1e6)
        result["latency"][name] = {
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "calls": len(timings),
        }
    api.End()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["latency"], indent=2))
    for row in result["sensitivity"]:
        print(
            row["condition"],
            row["engine"],
            f"{row['correct']}/{row['total']}",
            "rejected",
            row["rejected"],
        )


if __name__ == "__main__":
    main()
