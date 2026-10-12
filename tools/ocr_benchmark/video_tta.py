"""Explore recognition TTA on the labeled crops exported by sample_videos.py.

Variants and majority policy are diagnostic, not calibrated production settings.
The vote requires a strict majority of all variants; invalid strings cannot vote.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
from rapidocr import RapidOCR


def variants(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    inverted = cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)
    yield "native", crop
    yield "inverted_gray", inverted
    yield (
        "inverted_gray_border",
        cv2.copyMakeBorder(
            inverted, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        ),
    )
    yield (
        "native_border",
        cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(45, 45, 45)),
    )
    yield (
        "sharpen",
        cv2.addWeighted(crop, 1.5, cv2.GaussianBlur(crop, (0, 0), 1), -0.5, 0),
    )
    h, w = crop.shape[:2]
    yield "stretch_1.5", cv2.resize(crop, (round(w * 1.5), h))
    for shear in (-0.25, 0.25):
        transform = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
        yield (
            f"shear_{shear}",
            cv2.warpAffine(crop, transform, (w, h), borderMode=cv2.BORDER_REPLICATE),
        )
    for angle in (-3, 3):
        transform = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        yield (
            f"rotate_{angle}",
            cv2.warpAffine(crop, transform, (w, h), borderMode=cv2.BORDER_REPLICATE),
        )


def adaptive_summary(readings):
    """Replay a policy chosen using this development set; estimate call costs."""
    decisions = []
    for row in readings:
        predictions = {p["variant"]: p for p in row["variants"]}
        original = predictions["native"]
        value = original["value"]
        cost = original["ms"]
        triggered = value is None or original["confidence"] < 0.85
        if triggered:
            alternatives = [
                predictions[name]
                for name in ("inverted_gray", "shear_-0.25", "shear_0.25")
            ]
            counts = Counter(p["value"] for p in alternatives if p["value"] is not None)
            winner, votes = counts.most_common(1)[0] if counts else (None, 0)
            value = winner if votes >= 2 else None
            cost += sum(p["ms"] for p in alternatives)
        decisions.append(
            dict(
                image=row["image"],
                field=row["field"],
                predicted=value,
                correct=value == row["expected"],
                triggered=triggered,
                estimated_ms=cost,
            )
        )
    return dict(
        development_policy=True,
        confidence_trigger=0.85,
        alternatives=["inverted_gray", "shear_-0.25", "shear_0.25"],
        required_votes=2,
        correct=sum(r["correct"] for r in decisions),
        total=len(decisions),
        rejected=sum(r["predicted"] is None for r in decisions),
        triggered=sum(r["triggered"] for r in decisions),
        estimated_median_ms=float(np.median([r["estimated_ms"] for r in decisions])),
        estimated_mean_ms=float(np.mean([r["estimated_ms"] for r in decisions])),
        estimated_p95_ms=float(
            np.percentile([r["estimated_ms"] for r in decisions], 95)
        ),
        decisions=decisions,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    rows = json.loads(args.manifest.read_text())
    readings = []
    for row in rows:
        frame = cv2.imread(str(args.samples / row["image"]))
        if frame is None:
            raise ValueError(f"Cannot read {row['image']}")
        x, y, w, h = row["roi"]
        crop = frame[y : y + h, x : x + w]
        if crop.shape[:2] != (h, w):
            raise ValueError(f"Invalid ROI: {row}")
        if not readings:
            engine(crop)  # warm model before timing
        predictions = []
        for name, image in variants(crop):
            start = time.perf_counter()
            result = engine(image)
            elapsed = (time.perf_counter() - start) * 1000
            raw = (
                result.txts[0]
                if result.txts is not None and len(result.txts) == 1
                else None
            )
            value = (
                str(int(raw))
                if raw is not None and re.fullmatch(r"[0-9]+", raw)
                else None
            )
            predictions.append(
                dict(
                    variant=name,
                    raw=raw,
                    value=value,
                    confidence=float(result.scores[0]) if raw is not None else None,
                    ms=elapsed,
                    correct=value == row["expected"],
                )
            )
        counts = Counter(p["value"] for p in predictions if p["value"] is not None)
        winner, votes = counts.most_common(1)[0] if counts else (None, 0)
        accepted = winner if votes > len(predictions) / 2 else None
        readings.append(
            dict(
                **row,
                variants=predictions,
                vote=accepted,
                correct=accepted == row["expected"],
                total_ms=sum(p["ms"] for p in predictions),
            )
        )
    names = [p["variant"] for p in readings[0]["variants"]]
    summary = {
        name: dict(
            correct=sum(
                next(p for p in r["variants"] if p["variant"] == name)["correct"]
                for r in readings
            ),
            total=len(rows),
        )
        for name in names
    }
    summary["majority"] = dict(
        correct=sum(r["correct"] for r in readings),
        total=len(rows),
        rejected=sum(r["vote"] is None for r in readings),
        median_ms=float(np.median([r["total_ms"] for r in readings])),
    )
    output = dict(
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        summary=summary,
        adaptive=adaptive_summary(readings),
        readings=readings,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
