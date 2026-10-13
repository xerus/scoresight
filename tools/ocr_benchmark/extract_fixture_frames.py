"""Re-extract the committed basketball fixtures from original camera videos."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parents[2] / "tests/fixtures/basketball"
    manifest = json.loads((fixtures / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for row in manifest["frames"]:
        capture = cv2.VideoCapture(str(args.videos / row["video"]))
        try:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame"]):
                raise ValueError(f"Cannot seek to {row['video']} frame {row['frame']}")
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode {row['video']} frame {row['frame']}")
            if round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != row["frame"] + 1:
                raise ValueError(f"Unexpected frame position for {row['video']}")
        finally:
            capture.release()
        expected = cv2.imread(str(fixtures / row["image"]))
        if image.shape != (row["height"], row["width"], 3):
            raise ValueError(f"Unexpected dimensions for {row['image']}")
        if expected is None or not np.array_equal(image, expected):
            raise ValueError(f"Decoded pixels differ for {row['image']}")
        if not cv2.imwrite(
            str(args.output / row["image"]), image, [cv2.IMWRITE_PNG_COMPRESSION, 9]
        ):
            raise OSError(f"Cannot write {row['image']}")
        print(f"Verified and extracted {row['image']}")


if __name__ == "__main__":
    main()
