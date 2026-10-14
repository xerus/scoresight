"""Evaluate the frozen LED check on newly reviewed same-session score crops."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dnn_ocr import DnnOcr  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = ROOT / "tests/fixtures/basketball/heldout"
    manifest = json.loads((fixtures / "manifest.json").read_text())
    for name, expected in manifest["policy_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Frozen policy changed: {name}; report this as regression"
            )
    reader = DnnOcr(str(ROOT / "models/en_PP-OCRv4_rec_mobile.onnx"))
    results = []
    summary = {
        mode: dict(correct=0, wrong=0, rejected=0) for mode in ("baseline", "checked")
    }
    for field in manifest["fields"]:
        path = fixtures / field["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != field["sha256"]:
            raise ValueError(f"Fixture changed: {path}")
        crop = cv2.imread(str(path))
        result = dict(field)
        for mode, check in (("baseline", False), ("checked", True)):
            text, confidence, readings = reader.recognize(crop, check_led=check)
            accepted = text is not None and confidence >= 0.5
            normalized = (text.lstrip("0") or "0") if text else text
            outcome = (
                "rejected"
                if not accepted
                else "correct"
                if normalized == field["expected"]
                else "wrong"
            )
            summary[mode][outcome] += 1
            result[mode] = dict(
                text=text, confidence=confidence, outcome=outcome, readings=readings
            )
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                scope=manifest["note"],
                policy_sha256=manifest["policy_sha256"],
                summary=summary,
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
