"""Compare a targeted bias edit and CTC head fine-tuning against bundled PP-OCR.

Development uses September recordings, grouped by source video. October crops
are evaluated only after candidate selection. No application models are replaced.
Requires the app environment plus onnx; training uses NumPy, no GPU or PyTorch.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from ctc_head import log_softmax, train_head

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from defaults import FieldType  # noqa: E402
from dnn_ocr import DnnOcr  # noqa: E402
from clock_digits import format_clock_digits  # noqa: E402

TRAIN_SOURCES = {
    "Recording from 2026-09-26 17-58-51.368406.mp4",
    "Recording from 2026-09-26 18-02-24.926611.mp4",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fields():
    fields, manifests = [], {}

    def read(relative):
        path = ROOT / "tests/fixtures" / relative / "manifest.json"
        manifests[str(path.relative_to(ROOT))] = sha256(path)
        return path.parent, json.loads(path.read_text())

    def add(base, row, source, clock_input=0, roi=None):
        path = base / row["image"]
        fields.append(
            dict(
                image=str(path.relative_to(ROOT)),
                image_sha256=sha256(path),
                source=source,
                expected=row["expected"],
                clock=bool(clock_input),
                clock_input=clock_input,
                roi=roi,
                split="evaluation"
                if "2026-10" in source or "202610" in source
                else "train"
                if source in TRAIN_SOURCES
                else "validation",
            )
        )

    base, data = read("basketball")
    for frame in data["frames"]:
        for field in frame["fields"]:
            add(
                base,
                dict(field, image=frame["image"]),
                frame["video"],
                roi=field["roi"],
            )
    base, data = read("basketball/heldout")
    for row in data["fields"]:
        add(base, row, row["video"])
    base, data = read("basketball/sequences")
    for sequence in data["sequences"]:
        # Bound repeated-frame influence in training, retaining all evaluation
        # and validation frames. All frames from a source stay in the same split.
        counts = Counter()
        for row in sequence["frames"]:
            counts[row["expected"]] += 1
            if sequence["video"] in TRAIN_SOURCES and counts[row["expected"]] > 4:
                continue
            add(base, row, sequence["video"], clock_input=sequence["clock_input"])
    base, data = read("basketball/glare")
    for row in data["fields"]:
        # Semantic clock labels with glare/uncertain visible strokes are useful
        # for evaluation, but should not become forced CTC training targets.
        add(base, row, row["video"], clock_input=1)
        fields[-1]["split"] = "stress"
    base, data = read("panel_scoreboard")
    for row in data["fields"]:
        add(base, row, data["source"])
    base, data = read("phone_scoreboards")
    for row in data["fields"]:
        add(base, row, row["source"], clock_input=row["clock_input"])
    sources = {}
    for row in fields:
        previous = sources.setdefault(row["source"], row["split"])
        # Stress crops are excluded from selection/training despite their source.
        if row["split"] != "stress" and previous != row["split"]:
            raise ValueError("Source video crosses splits")
    return fields, manifests


def crop_field(row):
    crop = cv2.imread(str(ROOT / row["image"]))
    if crop is None:
        raise ValueError(f"Cannot read {row['image']}")
    if row["roi"]:
        x, y, width, height = row["roi"]
        crop = crop[y : y + height, x : x + width]
    if not crop.size:
        raise ValueError("Empty crop")
    return crop


def variants(crop):
    yield crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yield cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)
    h, w = crop.shape[:2]
    for shear in (-0.25, 0.25):
        yield cv2.warpAffine(
            crop,
            np.float32([[1, shear, -shear * h / 2], [0, 1, 0]]),
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )


def tensor(crop, minimum_width):
    h, w = crop.shape[:2]
    width = max(1, math.ceil(48 * w / h))
    image = cv2.resize(crop, (width, 48)).astype(np.float32)
    result = np.zeros((1, 3, 48, max(width, minimum_width)), dtype=np.float32)
    result[0, :, :, :width] = image.transpose(2, 0, 1) / 127.5 - 1
    return result


def decode(probabilities, characters):
    indices = probabilities.argmax(axis=1)
    selected = np.ones(len(indices), dtype=bool)
    selected[1:] = indices[1:] != indices[:-1]
    selected &= indices != 0
    text = "".join(characters[i] for i in indices[selected])
    scores = probabilities.max(axis=1)[selected]
    return text, float(np.mean(np.round(scores, 5))) if len(scores) else 0.0


def normalize(text, row):
    if row["clock"]:
        return (
            format_clock_digits(text, row["clock_input"]) if text is not None else None
        )
    if text and text.isascii() and text.isdigit():
        return text.lstrip("0") or "0"
    return text


def score_head(cache, weights, bias, characters):
    counts = Counter(correct=0, wrong=0, rejected=0)
    sources = {}
    for row, features in cache:
        text, confidence = decode(
            np.exp(log_softmax(features @ weights + bias)), characters
        )
        if not row["expected"]:
            # Empty raw CTC is correct for a labeled blank, but is never accepted
            # as a confirmed blank by the application pipeline.
            outcome = "correct" if not text else "wrong"
        elif confidence < 0.5 or not DnnOcr.valid_text(
            text, FieldType.TIME if row["clock"] else FieldType.NUMBER
        ):
            outcome = "rejected"
        else:
            formatted = normalize(text, row)
            outcome = (
                "rejected"
                if formatted is None
                else "correct"
                if formatted == row["expected"]
                else "wrong"
            )
        counts[outcome] += 1
        sources.setdefault(row["source"], Counter(correct=0, wrong=0, rejected=0))[
            outcome
        ] += 1
    return dict(
        counts,
        macro_correct=float(
            np.mean([c["correct"] / sum(c.values()) for c in sources.values()])
        ),
        macro_wrong=float(
            np.mean([c["wrong"] / sum(c.values()) for c in sources.values()])
        ),
        sources={name: dict(c) for name, c in sources.items()},
    )


def rank(counts):
    return counts["macro_correct"], -counts["macro_wrong"]


def head_constants(model):
    """Locate the actual output head, including exports using Constant nodes."""
    producers = {output: node for node in model.graph.node for output in node.output}
    softmax = producers[model.graph.output[0].name]
    if softmax.op_type != "Softmax":
        raise ValueError("Expected final Softmax")
    add = producers[softmax.input[0]]
    matmul = producers[add.input[0]]
    if add.op_type != "Add" or matmul.op_type != "MatMul":
        raise ValueError("Expected MatMul + bias head")
    names = (matmul.input[1], add.input[1])
    constants = []
    for name in names:
        node = producers[name]
        attribute = next(a for a in node.attribute if a.name == "value")
        constants.append(attribute)
    return matmul.input[0], constants


def save_head(model, weights, bias, path):
    candidate = copy.deepcopy(model)
    _, constants = head_constants(candidate)
    for attribute, value in zip(constants, (weights, bias)):
        attribute.t.CopyFrom(numpy_helper.from_array(value.astype(np.float32)))
    onnx.checker.check_model(candidate)
    onnx.save(candidate, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / "models/en_PP-OCRv4_rec_mobile.onnx"
    model = onnx.load(str(model_path))
    feature_name, constants = head_constants(model)
    weights, bias = (numpy_helper.to_array(a.t).copy() for a in constants)
    baseline = DnnOcr(str(model_path))
    characters = baseline.characters
    fields, manifests = load_fields()
    feature_model = copy.deepcopy(model)
    feature_model.graph.output.append(
        helper.make_tensor_value_info(
            feature_name, onnx.TensorProto.FLOAT, [None, None, weights.shape[0]]
        )
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        feature_model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    def features(crop, clock):
        probabilities, values = session.run(
            None, {baseline.input_name: tensor(crop, 320 if clock else 128)}
        )
        rebuilt = np.exp(log_softmax(values[0] @ weights + bias))
        if not np.allclose(probabilities[0], rebuilt, atol=2e-5):
            raise ValueError("Extracted head does not reproduce ONNX probabilities")
        return values[0].astype(np.float64)

    train, validation = [], []
    for row in fields:
        if row["split"] not in ("train", "validation"):
            continue
        crop = crop_field(row)
        if row["split"] == "validation":
            validation.append((row, features(crop, row["clock"])))
        else:
            # Compact-clock layout supplies punctuation in the app. Train the
            # observed digits, rather than forcing a colon into invisible pixels.
            observed = (
                row["expected"].replace(":", "").replace(".", "")
                if row["clock"]
                else row["expected"]
            )
            target = [characters.index(c) for c in observed]
            for variant in variants(crop):
                train.append((features(variant, row["clock"]), target))
    print(
        f"Extracted {len(train)} training variants and {len(validation)} validation crops",
        flush=True,
    )
    baseline_score = score_head(validation, weights, bias, characters)
    search = []
    # Narrow hypothesis: the pretrained head underweights 7 relative to 1.
    # This is a direct two-bias edit, not a general implementation of the article.
    selected_edit = (weights.copy(), bias.copy())
    edit_score = baseline_score
    edit_spec = dict(seven_bias=0.0)
    for amount in (0.25, 0.5, 1.0, 2.0, 4.0):
        edited = bias.copy()
        edited[characters.index("7")] += amount
        edited[characters.index("1")] -= amount
        counts = score_head(validation, weights, edited, characters)
        spec = dict(method="bias_edit", seven_bias=amount, validation=counts)
        search.append(spec)
        if rank(counts) > rank(edit_score):
            selected_edit, edit_score, edit_spec = (
                (weights.copy(), edited),
                counts,
                spec,
            )
    # Always retain a genuinely trained candidate even if no checkpoint wins.
    selected_tuned, tuned_score, tuned_spec = None, None, None
    for rate in (0.001, 0.003):
        for epoch, w, b, loss in train_head(train, weights, bias, rate, (5, 15, 40)):
            counts = score_head(validation, w, b, characters)
            spec = dict(
                method="ctc_head",
                learning_rate=rate,
                epoch=epoch,
                train_loss=loss,
                validation=counts,
            )
            search.append(spec)
            print(json.dumps(spec), flush=True)
            if tuned_score is None or rank(counts) > rank(tuned_score):
                selected_tuned, tuned_score, tuned_spec = (
                    (w.copy(), b.copy()),
                    counts,
                    spec,
                )
    models = {
        "baseline": (weights, bias),
        "bias_edit": selected_edit,
        "ctc_head": selected_tuned,
    }
    # Freeze candidates before accessing any October crop pixels or predictions.
    selected = dict(bias_edit=edit_spec, ctc_head=tuned_spec)
    frozen = args.artifacts / "selection.json"
    frozen.write_text(
        json.dumps(dict(selected=selected, search=search), indent=2) + "\n"
    )
    readers, artifacts = {}, {}
    for name, (w, b) in models.items():
        path = model_path if name == "baseline" else args.artifacts / f"{name}.onnx"
        if name != "baseline":
            save_head(model, w, b, path)
        readers[name] = DnnOcr(str(path))
        artifacts[name] = dict(
            path=str(path),
            sha256=sha256(path),
            changed_parameters=int(
                np.count_nonzero(w != weights) + np.count_nonzero(b != bias)
            ),
        )
    summary, readings = {}, []
    for row in fields:
        crop = crop_field(row)
        result = dict(row, predictions={})
        for name, reader in readers.items():
            for mode in ("native", "tta", "led_checked"):
                field_type = FieldType.TIME if row["clock"] else FieldType.NUMBER
                if mode == "native":
                    text, confidence = reader.predict(
                        crop, minimum_width=320 if row["clock"] else 128
                    )
                    passes = 1
                else:
                    text, confidence, alternatives = reader.recognize(
                        crop, field_type, check_led=mode == "led_checked"
                    )
                    passes = sum("confidence" in a for a in alternatives)
                formatted = normalize(text, row)
                accepted = (
                    bool(text)
                    and formatted is not None
                    and confidence >= 0.5
                    and reader.valid_text(text, field_type)
                )
                outcome = (
                    "rejected"
                    if not accepted
                    else "correct"
                    if formatted == row["expected"]
                    else "wrong"
                )
                key = f"{name}/{mode}"
                counts = summary.setdefault(row["split"], {}).setdefault(
                    key, dict(correct=0, wrong=0, rejected=0)
                )
                counts[outcome] += 1
                result["predictions"][key] = dict(
                    text=text,
                    formatted=formatted,
                    confidence=confidence,
                    outcome=outcome,
                    passes=passes,
                )
        readings.append(result)
    output = dict(
        scope="Exploratory final-layer experiment. Validation is same-session, grouped by source video. October evaluation uses previously reviewed development fixtures from other sessions; not a new blind benchmark. Adjacent frames are correlated. Glare stress crops never train/select candidates.",
        seed=20261110,
        platform=platform.platform(),
        python=platform.python_version(),
        packages={
            name: importlib.metadata.version(name)
            for name in ("numpy", "onnx", "onnxruntime", "opencv-python")
        },
        manifests=manifests,
        code_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                Path(__file__).resolve(),
                Path(__file__).with_name("ctc_head.py"),
                ROOT / "src/dnn_ocr.py",
                ROOT / "src/led_digit_evidence.py",
            )
        },
        training_variants=len(train),
        selection_sha256=sha256(frozen),
        selection_rule="Maximize source-video macro-average whole-field correctness, then minimize macro wrong rate; ties keep earlier/smaller candidate. Native CTC decoding with threshold 0.5 and app clock formatting. No October results used for selection.",
        baseline_validation=baseline_score,
        selected=selected,
        search=search,
        artifacts=artifacts,
        summary=summary,
        fields=readings,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
