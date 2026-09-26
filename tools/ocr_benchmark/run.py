"""Diagnostic comparison on manually labeled scoreboard fields; no app changes.

Run with --engine tesseract in the app environment, or --engine neural in the
benchmark environment. External model repositories are imported, never copied.
Model/cache paths are configurable using --cache-root (default /tmp).
"""

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def canonical(text):
    text = text.strip()
    if not text:
        return ""
    if re.fullmatch(r"\+?\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def variants(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    led = (((h < 40) | (h > 165)) & (s > 80) & (v > 160)).astype(np.uint8) * 255
    return {"rgb": crop, "otsu": otsu, "led160": led}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["tesseract", "neural"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    samples = []
    for row in manifest:
        frame = cv2.imread(str(ROOT / "images" / row["image"]))
        for field in ("home", "away"):
            x, y, w, h = row[field]["roi"]
            crop = frame[y : y + h, x : x + w]
            samples.append(
                (row["image"], field, row[field]["expected"], variants(crop))
            )
    results = []

    def record(model, setting, predict):
        start = time.perf_counter()
        readings = []
        for name, field, truth, images in samples:
            try:
                raw = predict(images)
                error = None
            except Exception as exc:
                raw, error = None, repr(exc)
            predicted = canonical(raw) if raw is not None else None
            readings.append(
                dict(
                    image=name,
                    field=field,
                    expected=truth,
                    raw=raw,
                    predicted=predicted,
                    correct=predicted == truth,
                    error=error,
                )
            )
        result = dict(
            model=model,
            setting=setting,
            seconds=time.perf_counter() - start,
            correct=sum(r["correct"] for r in readings),
            total=len(readings),
            blank_false_positives=sum(
                r["expected"] == "" and r["predicted"] not in ("", None)
                for r in readings
            ),
            readings=readings,
        )
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            model,
            setting,
            result["correct"],
            "/",
            len(readings),
            [r["raw"] for r in readings],
            flush=True,
        )

    cache = args.cache_root
    if args.engine == "tesseract":
        from tesserocr import PyTessBaseAPI

        models = [
            (name, ROOT / "tesseract/tessdata", name)
            for name in (
                "daktronics",
                "scoreboard_general",
                "scoreboard_general_large",
                "eng",
            )
        ]
        models += [
            ("tessdata_" + kind + "_eng", cache / ("ocrbench-tessdata-" + kind), "eng")
            for kind in ("best", "fast")
        ]
        models += [
            (name, cache / "scoresight-ssd-models", name)
            for name in ("7seg", "ssd", "ssd_plus", "ssd_int")
        ]
        for name, directory, language in models:
            for oem in (0, 1):
                try:
                    api = PyTessBaseAPI(
                        path=str(directory), lang=language, oem=oem, init=False
                    )
                    api.InitFull(
                        path=str(directory),
                        lang=language,
                        oem=oem,
                        configs=[],
                        variables={"load_system_dawg": "0", "load_freq_dawg": "0"},
                    )
                except Exception as exc:
                    print(name, "OEM", oem, "unavailable:", exc, flush=True)
                    continue
                with api:
                    api.SetVariable("tessedit_char_whitelist", "0123456789")
                    for psm in (7, 8, 10, 13):
                        api.SetPageSegMode(psm)
                        for kind in ("rgb", "otsu", "led160"):
                            for scale in (1, 3):

                                def predict(images, kind=kind, scale=scale):
                                    image = images[kind]
                                    if image.ndim == 3:
                                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                    image = 255 - image
                                    image = cv2.resize(
                                        image,
                                        None,
                                        fx=scale,
                                        fy=scale,
                                        interpolation=cv2.INTER_CUBIC,
                                    )
                                    image = cv2.copyMakeBorder(
                                        image,
                                        12,
                                        12,
                                        12,
                                        12,
                                        cv2.BORDER_CONSTANT,
                                        value=255,
                                    )
                                    api.ClearAdaptiveClassifier()
                                    api.SetImage(Image.fromarray(image))
                                    return api.GetUTF8Text().strip()

                                record(
                                    name,
                                    f"oem{oem}/psm{psm}/{kind}/scale{scale}",
                                    predict,
                                )
    else:
        import torch

        torch.set_num_threads(2)
        sys.path.insert(0, str(cache / "lcd-digit-recognition"))
        import ocr_reader as o

        o.MODEL_DIR = cache / "ocrbench-weights"
        model = o.get_crnn()
        for kind in ("rgb", "otsu", "led160"):

            def predict(images, kind=kind):
                image = images[kind]
                image = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB if image.ndim == 3 else cv2.COLOR_GRAY2RGB
                )
                tensor = o.TRANSFORM(Image.fromarray(image)).unsqueeze(0)
                with torch.no_grad():
                    logits = model(tensor).log_softmax(2)
                return o.decode_with_confidence(logits[:, 0, :])[0]

            record("OICWS_CRNN", kind, predict)
        import tensorflow as tf

        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        interpreter = tf.lite.Interpreter(
            model_path=str(cache / "renjith2/model_float16.tflite")
        )
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        alphabet = string.digits + string.ascii_lowercase + "."
        for kind in ("rgb", "otsu", "led160"):
            for invert in (False, True):

                def predict(images, kind=kind, invert=invert):
                    gray = images[kind]
                    if gray.ndim == 3:
                        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                    if invert:
                        gray = 255 - gray
                    tensor = (
                        cv2.resize(gray, (200, 31))[None, :, :, None].astype(np.float32)
                        / 255
                    )
                    interpreter.set_tensor(inp["index"], tensor)
                    interpreter.invoke()
                    ids = interpreter.get_tensor(out["index"])[0]
                    return "".join(
                        alphabet[int(i)] for i in ids if 0 <= i < len(alphabet)
                    )

                record("Renjith_TFLite", f"{kind}/invert{invert}", predict)
        for path in sorted((cache / "leander-sparse/models").rglob("*.keras")):
            model = tf.keras.models.load_model(path, compile=False)
            shape = model.input_shape
            for kind in ("otsu", "led160"):
                for invert in (True, False):

                    def predict(images, kind=kind, invert=invert):
                        image = images[kind]
                        if invert:
                            image = 255 - image
                        tensor = cv2.resize(image, (shape[2], shape[1]))[
                            None, :, :, None
                        ]
                        outputs = model(tensor, training=False)
                        if not isinstance(outputs, (list, tuple)):
                            outputs = [outputs]
                        # Class 13 has no label in the published dataset rules.
                        # Preserve it as unknown rather than silently deleting
                        # heads and turning an unrecognized field into blank.
                        return "".join(
                            "0123456789.-+?"[int(np.argmax(head[0]))]
                            for head in outputs
                        )

                    record(
                        "Leander/"
                        + str(path.relative_to(cache / "leander-sparse/models")),
                        f"{kind}/invert{invert}",
                        predict,
                    )
            tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
