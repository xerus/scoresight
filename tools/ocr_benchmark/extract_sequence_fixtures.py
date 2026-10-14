"""Re-extract and verify lossless chronological basketball crops."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, help="Optional crop manifest (sequences or fields)"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "tests/fixtures/basketball/sequences"
    manifest_path = args.manifest or root / "manifest.json"
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    sequences = manifest.get("sequences")
    if sequences is None:
        sequences = [
            dict(name=row["image"], video=row["video"], roi=row["roi"], frames=[row])
            for row in manifest["fields"]
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        capture = cv2.VideoCapture(str(args.videos / sequence["video"]))
        try:
            x, y, width, height = sequence["roi"]
            for row in sequence["frames"]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
                ok, frame = capture.read()
                if (
                    not ok
                    or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != row["frame"] + 1
                ):
                    raise ValueError(
                        f"Cannot decode {sequence['video']} frame {row['frame']}"
                    )
                crop = frame[y : y + height, x : x + width]
                original = root / row["image"]
                if hashlib.sha256(original.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError(f"Fixture checksum differs: {row['image']}")
                expected = cv2.imread(str(original))
                if expected is None or not np.array_equal(crop, expected):
                    raise ValueError(f"Decoded pixels differ: {row['image']}")
                if not cv2.imwrite(
                    str(args.output / row["image"]),
                    crop,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                ):
                    raise OSError(f"Cannot write {row['image']}")
            print(f"Verified {sequence['name']}: {len(sequence['frames'])} crops")
        finally:
            capture.release()


if __name__ == "__main__":
    main()
