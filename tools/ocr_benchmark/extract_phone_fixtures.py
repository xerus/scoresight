"""Reproduce phone scoreboard crops and verify them against committed pixels."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/phone_scoreboards"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((FIXTURES / "manifest.json").read_text())["fields"]
    hashes = {}
    for row in rows:
        source = args.videos / row["source"]
        if source not in hashes:
            hashes[source] = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashes[source] != row["source_sha256"]:
            raise ValueError(f"Source checksum differs: {source.name}")
        capture = cv2.VideoCapture(str(source))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            ok, image = capture.read()
            if (
                not ok
                or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != row["frame"] + 1
            ):
                raise ValueError(f"Cannot decode requested frame: {row['image']}")
        finally:
            capture.release()
        if [image.shape[1], image.shape[0]] != row["source_size"]:
            raise ValueError(f"Source dimensions differ: {row['image']}")
        x, y, width, height = row["roi"]
        crop = image[y : y + height, x : x + width]
        expected = cv2.imread(str(FIXTURES / row["image"]))
        if expected is None or not np.array_equal(crop, expected):
            raise ValueError(f"Decoded pixels differ: {row['image']}")
        if not cv2.imwrite(str(args.output / row["image"]), crop):
            raise OSError(f"Cannot write {row['image']}")
        print(f"Verified {row['image']}")


if __name__ == "__main__":
    main()
