"""CPU recognition on manually labeled video crops, without temporal smoothing.

Model loading and video decoding are excluded from warmed per-field timings.
RapidOCR reads native BGR crops; Tesseract reads inverted ROI Otsu with a border.
Missing results/errors are null, distinct from a model decoding an empty string.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine", choices=("rapidocr", "tesseract"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional Tesseract crop height; 0 preserves native size",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.height < 0:
        parser.error("--height cannot be negative")
    if args.engine == "rapidocr" and args.height:
        parser.error("--height applies only to Tesseract")
    rows = json.loads(args.manifest.read_text())
    crops = []
    decoded = {}
    for row in rows:
        key = (row["video"], row["frame"])
        if key not in decoded:
            cap = cv2.VideoCapture(str(args.videos / row["video"]))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode {key}")
                decoded[key] = frame
            finally:
                cap.release()
        frame = decoded[key]
        x, y, w, h = row["roi"]
        if (
            min(x, y) < 0
            or min(w, h) < 1
            or x + w > frame.shape[1]
            or y + h > frame.shape[0]
        ):
            raise ValueError(f"Invalid ROI: {row}")
        crops.append(frame[y : y + h, x : x + w].copy())

    if args.engine == "rapidocr":
        from rapidocr import RapidOCR

        engine = RapidOCR(
            params={
                "Global.use_det": False,
                "Global.use_cls": False,
                "Global.use_preprocess_img": False,
                "Global.text_score": 0.0,
                "Rec.model_path": str(args.model.resolve()),
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )

        def predict(crop):
            result = engine(crop)
            if result.txts is None or len(result.txts) != 1:
                return None, None
            return result.txts[0], float(result.scores[0])

        packages = ("rapidocr", "onnxruntime", "numpy", "opencv-python")
        preprocessing = (
            "native BGR; recognizer's resize/normalization; no detector or classifier"
        )
    else:
        from PIL import Image
        from tesserocr import PyTessBaseAPI

        api = PyTessBaseAPI(
            path=str(args.model.resolve().parent),
            lang=args.model.stem,
            oem=1,
            psm=8,
            variables={"load_system_dawg": "F", "load_freq_dawg": "F"},
        )
        api.SetVariable("tessedit_char_whitelist", "0123456789")

        def predict(crop):
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            if args.height:
                width = max(1, round(binary.shape[1] * args.height / binary.shape[0]))
                binary = cv2.resize(
                    binary, (width, args.height), interpolation=cv2.INTER_AREA
                )
            binary = cv2.copyMakeBorder(
                255 - binary, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255
            )
            api.ClearAdaptiveClassifier()
            api.SetImage(Image.fromarray(binary))
            return api.GetUTF8Text().strip(), float(api.MeanTextConf())

        packages = ("tesserocr", "numpy", "opencv-python")
        preprocessing = (
            "ROI grayscale/Otsu/invert/12px white border; native size; OEM1/PSM8/digits"
        )
        if args.height:
            preprocessing = preprocessing.replace(
                "native size", f"resize height {args.height}"
            )

    for crop in crops[:3]:
        predict(crop)
    readings = []
    latencies = []
    for row, crop in zip(rows, crops):
        times = []
        raw = score = error = None
        try:
            for _ in range(args.repeats):
                start = time.perf_counter()
                text, confidence = predict(crop)
                times.append((time.perf_counter() - start) * 1000)
                if len(times) == 1:
                    raw, score = text, confidence
        except Exception as exc:
            error = repr(exc)
        value = (
            str(int(raw)) if raw is not None and re.fullmatch(r"[0-9]+", raw) else raw
        )
        readings.append(
            dict(
                **row,
                raw=raw,
                predicted=value,
                confidence=score,
                error=error,
                correct=error is None and value == row["expected"],
                median_ms=float(np.median(times)) if times else None,
            )
        )
        latencies.extend(times)
    versions = {name: importlib.metadata.version(name) for name in packages}
    result = dict(
        engine=args.engine,
        model=args.model.name,
        model_bytes=args.model.stat().st_size,
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        python=platform.python_version(),
        platform=platform.platform(),
        packages=versions,
        preprocessing=preprocessing,
        repeats=args.repeats,
        correct=sum(r["correct"] for r in readings),
        total=len(readings),
        median_ms=float(np.median(latencies)),
        p95_ms=float(np.percentile(latencies, 95)),
        readings=readings,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            key: result[key]
            for key in ("model", "correct", "total", "median_ms", "p95_ms")
        }
    )
    if args.engine == "tesseract":
        api.End()


if __name__ == "__main__":
    main()
