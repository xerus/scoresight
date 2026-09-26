"""Compare warmed OICWS CRNN CPU/CUDA inference using the same eight crops.

The external checkout supplies the network architecture and decoder. Timing
includes transform, host/device transfer, forward, decode and synchronization,
excludes file I/O/model loading, and uses batch size one to reflect live fields.
"""

import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
# The isolated CUDA environment uses the existing CPU environment for OpenCV
# only; torch/torchvision/numpy are already resolved from the CUDA environment.
sys.path.append("/tmp/ocrbench-venv/lib/python3.11/site-packages")
sys.path.insert(0, "/tmp/lcd-digit-recognition")
o = importlib.import_module("ocr_reader")
torch.set_num_threads(2)
rows = json.loads((ROOT / "tools/ocr_benchmark/manifest.json").read_text())
samples = []
for row in rows:
    image = Image.open(ROOT / "images" / row["image"]).convert("RGB")
    for field in ("home", "away"):
        x, y, w, h = row[field]["roi"]
        samples.append((image.crop((x, y, x + w, y + h)), row[field]["expected"]))

result = dict(
    torch=torch.__version__,
    cuda_available=torch.cuda.is_available(),
    cuda_version=torch.version.cuda,
    batch_size=1,
    cpu_threads=2,
    runs=[],
)
model = o.get_crnn()
for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
    model.to(device)

    def infer(image):
        tensor = o.TRANSFORM(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor).log_softmax(2)
        raw = o.decode_with_confidence(logits[:, 0, :])[0]
        if device == "cuda":
            torch.cuda.synchronize()
        return raw

    for image, _ in samples:
        infer(image)
    elapsed = []
    readings = []
    for _ in range(25):
        readings = []
        for image, expected in samples:
            start = time.perf_counter()
            raw = infer(image)
            elapsed.append((time.perf_counter() - start) * 1000)
            readings.append(dict(expected=expected, raw=raw))
    result["runs"].append(
        dict(
            device=device,
            count=len(elapsed),
            median_ms=statistics.median(elapsed),
            p95_ms=float(np.percentile(elapsed, 95)),
            readings=readings,
        )
    )
if torch.cuda.is_available():
    result["gpu"] = torch.cuda.get_device_name(0)
(ROOT / "results/ocr_benchmark/gpu.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
