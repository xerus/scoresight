"""Evaluate OICWS YOLO localization on manually labeled score regions."""

import importlib
import json
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from PIL import Image
from run import canonical

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)
model = YOLO("/tmp/ocrbench-weights/best.pt")
rows = json.loads((ROOT / "tools/ocr_benchmark/manifest.json").read_text())
sys.path.insert(0, "/tmp/lcd-digit-recognition")
o = importlib.import_module("ocr_reader")

crnn = o.get_crnn()
results = []
for size in (640, 1280):
    for confidence in (0.3, 0.05):
        for row in rows:
            frame = cv2.imread(str(ROOT / "images" / row["image"]))
            prediction = model.predict(
                frame, imgsz=size, conf=confidence, device="cpu", verbose=False
            )[0]
            boxes = prediction.boxes.xyxy.cpu().numpy().tolist()
            scores = prediction.boxes.conf.cpu().numpy().tolist()
            readings = []
            for field in ("home", "away"):
                x, y, w, h = row[field]["roi"]
                candidates = [
                    (score, box)
                    for score, box in zip(scores, boxes)
                    if x <= (box[0] + box[2]) / 2 <= x + w
                    and y <= (box[1] + box[3]) / 2 <= y + h
                ]
                raw = None
                if candidates:
                    box = max(candidates, key=lambda item: item[0])[1]
                    x0, y0, x1, y1 = map(int, box)
                    crop = frame[max(0, y0 - 5) : y1 + 5, max(0, x0 - 5) : x1 + 5]
                    tensor = o.TRANSFORM(
                        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    ).unsqueeze(0)
                    with torch.no_grad():
                        logits = crnn(tensor).log_softmax(2)
                    raw = o.decode_with_confidence(logits[:, 0, :])[0]
                predicted = canonical(raw) if raw is not None else None
                readings.append(
                    dict(
                        field=field,
                        expected=row[field]["expected"],
                        raw=raw,
                        predicted=predicted,
                        correct=predicted == row[field]["expected"],
                    )
                )
            results.append(
                dict(
                    image=row["image"],
                    imgsz=size,
                    confidence=confidence,
                    boxes=boxes,
                    scores=scores,
                    readings=readings,
                )
            )
(ROOT / "results/ocr_benchmark/detector.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
print(json.dumps(results, indent=2))
