"""Inventory local clips and export three frames per clip for manual review."""

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.videos.glob("*.mp4"))
    if not paths:
        parser.error("No MP4 files found")
    args.output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (960, len(paths) * 205))
    draw = ImageDraw.Draw(sheet)
    rows = []
    for index, path in enumerate(paths, 1):
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open {path}")
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if frames < 1 or fps <= 0:
                raise ValueError(f"Invalid video metadata: {path}")
            row = dict(
                video=path.name,
                bytes=path.stat().st_size,
                frames=frames,
                fps=fps,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                duration_s=frames / fps,
                samples=[],
            )
            for column, fraction in enumerate((0.1, 0.5, 0.9)):
                frame = int((frames - 1) * fraction)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"Cannot read {path} frame {frame}")
                name = f"clip{index:02d}_{column}.png"
                if not cv2.imwrite(str(args.output / name), image):
                    raise ValueError(f"Cannot write {name}")
                row["samples"].append(dict(frame=frame, time_s=frame / fps, image=name))
                thumb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                thumb.thumbnail((320, 180))
                sheet.paste(thumb, (column * 320, (index - 1) * 205 + 25))
            draw.text(
                (5, (index - 1) * 205 + 5),
                f"{index:02d} {path.name} {row['duration_s']:.1f}s",
                fill="white",
            )
            rows.append(row)
        finally:
            cap.release()
    sheet.save(args.output / "contact.jpg")
    (args.output / "inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{len(rows)} clips, {sum(r['duration_s'] for r in rows):.1f} seconds")


if __name__ == "__main__":
    main()
