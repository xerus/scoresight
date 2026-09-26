"""Original OpenCV-only, calibrated two-cell seven-segment benchmark.

Blank is an empty string; rejected/unknown is None. No temporal smoothing.
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Seven rectangular occupancy regions: top, upper-right, lower-right,
# bottom, lower-left, upper-left, middle. Coordinates relative to each slot.
ZONES = [
    (0.35, 0.04, 0.65, 0.18),
    (0.68, 0.16, 0.98, 0.43),
    (0.60, 0.56, 0.94, 0.86),
    (0.35, 0.81, 0.65, 0.97),
    (0.01, 0.56, 0.34, 0.86),
    (0.07, 0.16, 0.40, 0.43),
    (0.35, 0.40, 0.65, 0.60),
]
PATTERNS = {
    "1111110": "0",
    "0110000": "1",
    "1101101": "2",
    "1111001": "3",
    "0110011": "4",
    "1011011": "5",
    "1011111": "6",
    "1110000": "7",
    "1111111": "8",
    "1111011": "9",
}


def read_score(crop, brightness=140, occupancy=0.09):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (
        ((hsv[:, :, 0] <= 35) | (hsv[:, :, 0] >= 165))
        & (hsv[:, :, 1] >= 90)
        & (hsv[:, :, 2] >= brightness)
    ).astype(np.uint8) * 255
    digits, details = [], []
    for cell in np.array_split(mask, 2, axis=1):
        h, w = cell.shape
        densities = [
            float(
                np.mean(cell[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] > 0)
            )
            for x0, y0, x1, y1 in ZONES
        ]
        pattern = "".join("1" if p >= occupancy else "0" for p in densities)
        # Require no bright evidence for blank; isolated spots become rejection.
        value = "" if np.count_nonzero(cell) < 3 else PATTERNS.get(pattern)
        digits.append(value)
        details.append({"pattern": pattern, "densities": densities, "value": value})
    # A lit tens digit followed by a blank units cell is ambiguous, not a number.
    result = (
        None if None in digits or (digits[0] and not digits[1]) else "".join(digits)
    )
    return result, mask, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tools/opencv_benchmark/results"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
    records = []
    sensitivity = []
    for sample in manifest:
        frame = cv2.imread(str(ROOT / "images" / sample["image"]))
        if frame is None:
            raise FileNotFoundError(sample["image"])
        for field in ("home", "away"):
            spec = sample[field]
            x, y, w, h = spec["roi"]
            crop = frame[y : y + h, x : x + w]
            start = time.perf_counter()
            prediction, mask, details = read_score(crop)
            elapsed = (time.perf_counter() - start) * 1000
            stem = sample["image"].replace(":", "_").removesuffix(".png") + "_" + field
            cv2.imwrite(str(args.output / (stem + "_crop.png")), crop)
            cv2.imwrite(str(args.output / (stem + "_mask.png")), mask)
            records.append(
                {
                    "image": sample["image"],
                    "field": field,
                    "expected": spec["expected"],
                    "prediction": prediction,
                    "correct": prediction == spec["expected"],
                    "milliseconds": elapsed,
                    "slots": details,
                }
            )
    for name, gain, dx, dy in [
        ("original", 1, 0, 0),
        ("brightness_60pct", 0.6, 0, 0),
        ("brightness_80pct", 0.8, 0, 0),
        ("brightness_120pct", 1.2, 0, 0),
        ("roi_right_2px", 1, 2, 0),
        ("roi_down_2px", 1, 0, 2),
    ]:
        correct = rejected = 0
        for sample in manifest:
            frame = cv2.imread(str(ROOT / "images" / sample["image"]))
            for field in ("home", "away"):
                spec = sample[field]
                x, y, w, h = spec["roi"]
                crop = frame[y + dy : y + dy + h, x + dx : x + dx + w]
                crop = np.clip(crop.astype(float) * gain, 0, 255).astype(np.uint8)
                predicted, _, _ = read_score(crop)
                correct += predicted == spec["expected"]
                rejected += predicted is None
        sensitivity.append(
            {"condition": name, "correct": correct, "total": 8, "rejected": rejected}
        )
    result = {
        "sensitivity": sensitivity,
        "opencv": cv2.__version__,
        "samples": records,
        "correct": sum(r["correct"] for r in records),
        "total": len(records),
        "blank_false_positives": sum(
            r["expected"] == "" and r["prediction"] not in ("", None) for r in records
        ),
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for r in records:
        print(
            r["image"],
            r["field"],
            repr(r["expected"]),
            "=>",
            repr(r["prediction"]),
            [s["pattern"] for s in r["slots"]],
        )
    print(json.dumps(sensitivity, indent=2))
    print(
        f"Correct: {result['correct']}/{result['total']}; blank false positives: {result['blank_false_positives']}"
    )


if __name__ == "__main__":
    main()
