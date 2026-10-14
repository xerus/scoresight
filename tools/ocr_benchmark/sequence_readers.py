"""Run committed chronological crops through the actual app OCR/filter path."""

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from defaults import (
    basketball_settings_for_box,
    default_info_for_box_name,
    normalize_settings_dict,
)  # noqa: E402
from dnn_ocr import DnnOcr  # noqa: E402
from tesseract import TextDetector  # noqa: E402
from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult  # noqa: E402


def evaluate_sequence(sequence, fixture_root, neural, check_led, clock_input):
    settings = normalize_settings_dict(
        dict(
            basketball_settings_for_box(sequence["field"]),
            check_led=check_led,
            clock_input=clock_input,
            box_ocr_model=-1,
        ),
        default_info_for_box_name(sequence["field"]),
    )
    reader = TextDetector.__new__(TextDetector)
    reader.api_lock = threading.Lock()
    reader.api = None
    reader.dnn = neural
    reader.ocr_model_index = TextDetector.OcrModelIndex.PPOCR_TTA
    reader.reset_temporal()
    width, height = sequence["roi"][2:]
    target = TextDetectionTarget(0, 0, width, height, sequence["field"], settings)
    first = cv2.imread(str(fixture_root / sequence["frames"][0]["image"]))
    neural.recognize(
        first, settings["type"], check_led=check_led, integer_seconds=clock_input == 5
    )  # warm without changing confirmation
    states = TextDetectionTargetWithResult.ResultState
    published, previous_expected = None, None
    events, observations, durations = [], [], []
    counters = {
        name: 0
        for name in (
            "accepted_correct",
            "accepted_wrong",
            "rejected",
            "correct_published_samples",
            "wrong_published_samples",
            "retained_mismatch_samples",
            "unavailable_samples",
        )
    }
    for row in sequence["frames"]:
        crop = cv2.imread(str(fixture_root / row["image"]))
        if crop is None or crop.shape[:2] != (height, width):
            raise ValueError(f"Invalid fixture {row['image']}")
        if row["expected"] != previous_expected:
            events.append(
                {
                    "expected": row["expected"],
                    "first_observed_at": row["timestamp"],
                    "first_correct_publication_at": None,
                    "delay_seconds": None,
                }
            )
            previous_expected = row["expected"]
        start = time.perf_counter()
        result = reader.detect_multi_text(
            None, None, [target], color=crop, observation_time=row["timestamp"]
        )[0]
        durations.append((time.perf_counter() - start) * 1000)
        accepted = result.state == states.Success or (
            result.state == states.Empty and not settings["skip_empty"]
        )
        if accepted:
            published = result.text
            counters[
                "accepted_correct" if published == row["expected"] else "accepted_wrong"
            ] += 1
        else:
            counters["rejected"] += 1
        if published is None:
            counters["unavailable_samples"] += 1
        elif published == row["expected"]:
            counters["correct_published_samples"] += 1
            if events[-1]["first_correct_publication_at"] is None:
                events[-1]["first_correct_publication_at"] = row["timestamp"]
                events[-1]["delay_seconds"] = round(
                    row["timestamp"] - events[-1]["first_observed_at"], 6
                )
        else:
            counters["wrong_published_samples"] += 1
            if not accepted:
                counters["retained_mismatch_samples"] += 1
        observations.append(
            {
                "frame": row["frame"],
                "timestamp": row["timestamp"],
                "expected": row["expected"],
                "text": result.text,
                "state": result.state.name,
                "published": published,
                "processing_ms": round(durations[-1], 3),
                "extra": result.extra,
            }
        )
    return {
        "settings": settings,
        "summary": dict(
            counters,
            median_processing_ms=round(statistics.median(durations), 3),
            p95_processing_ms=round(
                sorted(durations)[max(0, int(len(durations) * 0.95) - 1)], 3
            ),
        ),
        "events": events,
        "observations": observations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = ROOT / "tests/fixtures/basketball/sequences"
    manifest = json.loads((fixtures / "manifest.json").read_text())
    neural = DnnOcr(str(ROOT / "models/en_PP-OCRv4_rec_mobile.onnx"))
    results = []
    for sequence in manifest["sequences"]:
        row = {"name": sequence["name"], "sample_count": len(sequence["frames"])}
        for mode, check in (("baseline", False), ("led_check", True)):
            row[mode] = evaluate_sequence(
                sequence, fixtures, neural, check, sequence["clock_input"]
            )
        if sequence["clock_input"]:
            row["observed_separators_only"] = evaluate_sequence(
                sequence, fixtures, neural, False, 0
            )
        if sequence["field"] == "Shot Clock":
            row["integer_seconds_optimized"] = evaluate_sequence(
                sequence, fixtures, neural, True, 5
            )
        results.append(row)
        print(
            sequence["name"],
            json.dumps(
                {mode: row[mode]["summary"] for mode in ("baseline", "led_check")}
            ),
            flush=True,
        )
    output = {
        "scope": "Same-session chronological crops, not another-camera/game validation. Delay is relative to first sampled visible value; capture and model-loading delays are excluded.",
        "sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/dnn_ocr.py",
                "src/led_digit_evidence.py",
                "src/clock_digits.py",
                "src/tesseract.py",
                "src/temporal_values.py",
                "models/en_PP-OCRv4_rec_mobile.onnx",
                "tests/fixtures/basketball/sequences/manifest.json",
            )
        },
        "sequences": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
