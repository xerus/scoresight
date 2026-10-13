"""Diagnostic CPU sweep; changed padding is experimental, not app behavior."""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dnn_ocr import DnnOcr  # noqa: E402


class ExperimentalOcr(DnnOcr):
    def __init__(self, model_path, minimum_width, threads):
        super().__init__(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.minimum_width = minimum_width

    def predict(self, crop, minimum_width=320):
        height, width = crop.shape[:2]
        resized_width = max(1, math.ceil(48 * width / height))
        padded_width = max(self.minimum_width, resized_width)
        image = cv2.resize(crop, (resized_width, 48)).astype(np.float32)
        image = image.transpose(2, 0, 1) / 127.5 - 1
        tensor = np.zeros((1, 3, 48, padded_width), dtype=np.float32)
        tensor[0, :, :, :resized_width] = image
        probabilities = self.session.run(None, {self.input_name: tensor})[0][0]
        indices = probabilities.argmax(axis=1)
        selected = np.ones(len(indices), dtype=bool)
        selected[1:] = indices[1:] != indices[:-1]
        selected &= indices != 0
        text = "".join(self.characters[i] for i in indices[selected])
        scores = probabilities.max(axis=1)[selected]
        confidence = float(np.mean(np.round(scores, 5))) if len(scores) else 0.0
        return text, round(confidence, 5)


def normalized(text):
    return (text.lstrip("0") or "0") if text and text.isdigit() else text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples", type=Path)
    source.add_argument("--videos", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--widths", type=int, nargs="+", help="Widths to compare on one CPU thread"
    )
    parser.add_argument(
        "--verify-app-width",
        type=int,
        help="Verify app numeric readings match this experimental width",
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Reverse configuration order"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    rows = json.loads(args.manifest.read_text())
    crops = []
    decoded = {}
    for row in rows:
        if args.samples:
            image = cv2.imread(str(args.samples / row["image"]))
        else:
            key = (row["video"], row["frame"])
            if key not in decoded:
                cap = cv2.VideoCapture(str(args.videos / row["video"]))
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f"Cannot decode {key}")
                    decoded[key] = image
                finally:
                    cap.release()
            image = decoded[key]
        if image is None:
            raise ValueError(f"Missing image: {row['image']}")
        x, y, width, height = row["roi"]
        crop = image[y : y + height, x : x + width].copy()
        if crop.shape[:2] != (height, width):
            raise ValueError(f"Invalid ROI: {row}")
        crops.append(crop)
    results = []
    configurations = [
        (320, 1),
        (256, 1),
        (192, 1),
        (160, 1),
        (128, 1),
        (320, 2),
    ]
    if args.widths:
        if min(args.widths) < 1:
            parser.error("--widths must be positive")
        configurations = [(width, 1) for width in args.widths]
    if args.reverse:
        configurations.reverse()
    for minimum_width, threads in configurations:
        engine = ExperimentalOcr(str(args.model), minimum_width, threads)
        engine.recognize(crops[0])
        readings = []
        timings = []
        for row, crop in zip(rows, crops):
            native = engine.predict(crop)
            durations = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                text, confidence, variants = engine.recognize(crop)
                durations.append((time.perf_counter() - start) * 1000)
            timings.extend(durations)
            readings.append(
                dict(
                    row,
                    native_text=native[0],
                    native_confidence=native[1],
                    text=text,
                    confidence=confidence,
                    correct=normalized(text) == row["expected"],
                    variants=variants,
                    median_ms=float(np.median(durations)),
                )
            )
        result = {
            "minimum_width": minimum_width,
            "threads": threads,
            "correct": sum(r["correct"] for r in readings),
            "native_correct": sum(
                normalized(r["native_text"]) == r["expected"] for r in readings
            ),
            "fallbacks": sum(len(r["variants"]) > 1 for r in readings),
            "mean_ms": float(np.mean(timings)),
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "readings": readings,
        }
        result["acceptance_at_0_5"] = {
            "accepted_correct": sum(
                r["correct"] and r["text"] is not None and r["confidence"] >= 0.5
                for r in readings
            ),
            "accepted_wrong": sum(
                not r["correct"] and r["text"] is not None and r["confidence"] >= 0.5
                for r in readings
            ),
            "rejected": sum(
                r["text"] is None or r["confidence"] < 0.5 for r in readings
            ),
        }
        results.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "readings"}), flush=True
        )
    output = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "reversed_order": args.reverse,
        "timings": "Warmed full recognize calls; video decoding and model loading excluded",
        "configurations": results,
    }
    if args.verify_app_width is not None:
        configuration = next(
            (r for r in results if r["minimum_width"] == args.verify_app_width), None
        )
        if configuration is None:
            parser.error("--verify-app-width must be one of the measured widths")
        engine = DnnOcr(str(args.model))
        matches = []
        for crop, reading in zip(crops, configuration["readings"]):
            actual = engine.recognize(crop)
            expected = (reading["text"], reading["confidence"], reading["variants"])
            matches.append(actual == expected)
        output["app_equivalence"] = {
            "width": args.verify_app_width,
            "matching_readings": sum(matches),
            "total_readings": len(matches),
            "source_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[2] / "src/dnn_ocr.py").read_bytes()
            ).hexdigest(),
        }
        if not all(matches):
            raise ValueError(
                f"App predictions do not match width {args.verify_app_width}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
