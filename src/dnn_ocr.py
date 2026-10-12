"""PP-OCRv4 English recognition on CPU with selective test-time augmentation."""

from collections import Counter
import math
import re

import cv2
import numpy as np

from defaults import FieldType


class DnnOcr:
    def __init__(self, model_path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        characters = self.session.get_modelmeta().custom_metadata_map["character"]
        self.characters = [""] + characters.splitlines() + [" "]
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, crop):
        """Preserve aspect ratio, normalize BGR, then greedily decode CTC."""
        height, width = crop.shape[:2]
        resized_width = max(1, math.ceil(48 * width / height))
        padded_width = max(320, int(48 * width / height))
        resized_width = min(resized_width, padded_width)
        image = cv2.resize(crop, (resized_width, 48)).astype(np.float32)
        image = image.transpose(2, 0, 1) / 127.5 - 1
        tensor = np.zeros((1, 3, 48, padded_width), dtype=np.float32)
        tensor[0, :, :, :resized_width] = image
        probabilities = self.session.run(None, {self.input_name: tensor})[0][0]
        indices = probabilities.argmax(axis=1)
        selected = np.ones(len(indices), dtype=bool)
        selected[1:] = indices[1:] != indices[:-1]
        selected &= indices != 0
        text = "".join(self.characters[i] for i in indices[selected])
        scores = probabilities.max(axis=1)[selected]
        confidence = float(np.mean(np.round(scores, 5))) if len(scores) else 0.0
        return text, round(confidence, 5)

    @staticmethod
    def valid_text(text, field_type):
        if field_type == FieldType.NUMBER:
            return re.fullmatch(r"[0-9]+", text) is not None
        if field_type == FieldType.TIME:
            return re.fullmatch(r"[0-9]+(?:[:.][0-9]+)?", text) is not None
        return bool(text.strip())

    def recognize(self, crop, field_type=FieldType.NUMBER):
        text, confidence = self.predict(crop)
        readings = [{"variant": "native", "text": text, "confidence": confidence}]
        if confidence >= 0.85 and self.valid_text(text, field_type):
            return text, confidence, readings
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        alternatives = [("inverted_gray", cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR))]
        height, width = crop.shape[:2]
        for shear in (-0.25, 0.25):
            transform = np.float32([[1, shear, -shear * height / 2], [0, 1, 0]])
            image = cv2.warpAffine(
                crop, transform, (width, height), borderMode=cv2.BORDER_REPLICATE
            )
            alternatives.append((f"shear_{shear}", image))
        for name, image in alternatives:
            value, score = self.predict(image)
            readings.append({"variant": name, "text": value, "confidence": score})
        votes = Counter(
            r["text"] for r in readings[1:] if self.valid_text(r["text"], field_type)
        )
        winner, count = votes.most_common(1)[0] if votes else (None, 0)
        if count < 2:
            return None, 0.0, readings
        confidence = float(
            np.mean([r["confidence"] for r in readings[1:] if r["text"] == winner])
        )
        return winner, confidence, readings
