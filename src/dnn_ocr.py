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

    def predict(self, crop, minimum_width=320):
        """Preserve aspect ratio, normalize BGR, then greedily decode CTC."""
        height, width = crop.shape[:2]
        resized_width = max(1, math.ceil(48 * width / height))
        padded_width = max(minimum_width, resized_width)
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

    def recognize(
        self, crop, field_type=FieldType.NUMBER, check_led=False, integer_seconds=False
    ):
        if not check_led or field_type == FieldType.TEXT:
            return self._recognize(crop, field_type, integer_seconds)
        from led_digit_evidence import digit_evidence

        shape_text, bounds, details = digit_evidence(crop)
        evidence_reading = dict(details, variant="led_shapes", text=shape_text)
        if details.get("reason") == "clipped_digits":
            return None, 0.0, [evidence_reading]
        text, confidence, readings = self._recognize(crop, field_type, integer_seconds)
        readings.append(evidence_reading)

        def masked_reading():
            core_min = details.get("core_min", 160)
            masked = cv2.cvtColor(
                (crop.min(axis=2) >= core_min).astype(np.uint8) * 255,
                cv2.COLOR_GRAY2BGR,
            )
            if core_min != 240 and bounds is not None:
                left, top, right, bottom = bounds
                masked = masked[top:bottom, left:right]
            value, score, alternatives = self._recognize(
                masked, field_type, integer_seconds
            )
            readings.extend(
                dict(reading, variant="core_" + reading["variant"])
                for reading in alternatives
            )
            if core_min == 240 and field_type == FieldType.TIME:

                def matches_evidence(candidate):
                    if not self.valid_text(candidate or "", field_type):
                        return False
                    digits = candidate.replace(":", "").replace(".", "")
                    if shape_text is not None:
                        return digits == shape_text
                    return (
                        len(digits) >= details.get("min_digits", 0)
                        and digits.startswith(details.get("known_prefix", ""))
                        and digits.endswith(details.get("known_suffix", ""))
                    )

                if score >= 0.5 and matches_evidence(value):
                    return value, score
                # Two bounded native passes on less aggressive masks can retain
                # thin strokes erased by the strict green-core threshold. Both
                # must agree confidently and satisfy the original shape evidence.
                candidates = []
                core = crop.min(axis=2)
                for threshold in (220, 200):
                    variant = cv2.cvtColor(
                        (core >= threshold).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR
                    )
                    candidate, candidate_score = self.predict(
                        variant, minimum_width=320
                    )
                    readings.append(
                        dict(
                            variant=f"core_{threshold}",
                            text=candidate,
                            confidence=candidate_score,
                        )
                    )
                    candidates.append((candidate.replace(" ", ""), candidate_score))
                if (
                    candidates[0][0] == candidates[1][0]
                    and all(conf >= 0.8 for _, conf in candidates)
                    and matches_evidence(candidates[0][0])
                ):
                    return candidates[0][0], min(conf for _, conf in candidates)
            return value, score

        if shape_text is None:
            # Partial evidence can establish a missing terminal stroke even
            # when glare prevents classifying the preceding joined digits.
            minimum = details.get("min_digits", 0)
            suffix = details.get("known_suffix", "")
            prefix = details.get("known_prefix", "")
            native = next((r["text"] for r in readings if r["variant"] == "native"), "")
            anchor_source = (
                (native or "").lstrip(".: ")
                if field_type == FieldType.TIME
                else native or ""
            )
            anchor = re.match(r"[0-9]+", anchor_source)
            observed = text.replace(":", "").replace(".", "") if text else ""
            contradicts = (suffix and not observed.endswith(suffix)) or (
                prefix and not observed.startswith(prefix)
            )
            if anchor and suffix and len(anchor[0]) >= minimum:
                anchor_text = anchor[0][: -len(suffix)]
            else:
                anchor_text = anchor[0] if anchor else ""
            if (
                bounds is not None
                and (suffix or prefix)
                and minimum >= 2
                and (len(observed) < minimum or contradicts)
                and anchor
            ):
                retry, score = masked_reading()
                digits = retry.replace(":", "").replace(".", "") if retry else ""
                if (
                    self.valid_text(retry or "", field_type)
                    and len(digits) >= minimum
                    and (
                        digits.startswith(prefix)
                        if prefix
                        else digits.startswith(anchor_text)
                    )
                    and digits.endswith(suffix)
                ):
                    return retry, score, readings
                return None, 0.0, readings
            # A lone neural digit is unsafe when the bright silhouette could
            # contain joined digits. Retry the complete silhouette and require
            # an additional observed leading digit while preserving the suffix.
            boxes = details.get("boxes", [])
            observed = text.replace(":", "").replace(".", "") if text else ""
            wide_single = len(boxes) == 1 and boxes[0][2] / boxes[0][3] >= 0.7
            consistent_parts = len(boxes) >= 2 and all(
                box[3] >= 0.8 * max(part[3] for part in boxes) for box in boxes
            )
            if (
                bounds is not None
                and observed.isascii()
                and observed.isdigit()
                and len(observed) == 1
                and (wide_single or consistent_parts)
            ):
                left, top, right, bottom = bounds
                retry, retry_confidence, retry_readings = self._recognize(
                    crop[top:bottom, left:right], field_type, integer_seconds
                )
                readings.extend(
                    dict(reading, variant="tight_" + reading["variant"])
                    for reading in retry_readings
                )
                retry_digits = retry.replace(":", "").replace(".", "") if retry else ""
                if (
                    self.valid_text(retry or "", field_type)
                    and len(retry_digits) == 2
                    and retry_digits.endswith(observed)
                ):
                    return retry, retry_confidence, readings
                return None, 0.0, readings
            return text, confidence, readings

        def agrees(value):
            if value is None or not self.valid_text(value, field_type):
                return False
            observed_digits = value.replace(":", "").replace(".", "")
            if field_type == FieldType.NUMBER:
                return (observed_digits.lstrip("0") or "0") == (
                    shape_text.lstrip("0") or "0"
                )
            return observed_digits == shape_text

        if agrees(text) and confidence >= 0.5:
            return text, confidence, readings
        if details.get("core_min") == 240:
            retry_mask, mask_confidence = masked_reading()
            if agrees(retry_mask) and mask_confidence >= 0.5:
                return retry_mask, mask_confidence, readings
        left, top, right, bottom = bounds
        retry, retry_confidence, retry_readings = self._recognize(
            crop[top:bottom, left:right], field_type, integer_seconds
        )
        readings.extend(
            dict(reading, variant="tight_" + reading["variant"])
            for reading in retry_readings
        )
        # A tighter neural reading must agree with the independently observed
        # shapes. Never publish a shape-only substitution or a shortened value.
        if agrees(retry) and retry_confidence >= 0.5:
            return retry, retry_confidence, readings
        if details.get("core_min") != 240:
            retry_mask, mask_confidence = masked_reading()
            if agrees(retry_mask) and mask_confidence >= 0.5:
                return retry_mask, mask_confidence, readings
        low_confidence = [
            (value, score)
            for value, score in ((text, confidence), (retry, retry_confidence))
            if agrees(value)
        ]
        if low_confidence:
            value, score = max(low_confidence, key=lambda pair: pair[1])
            return value, score, readings
        return None, 0.0, readings

    def _recognize(self, crop, field_type, integer_seconds=False):
        # Short numeric fields need less right padding. Keep the original
        # convention for clocks/text and choose per call rather than mutating
        # shared model state while processing different fields.
        minimum_width = (
            128
            if field_type == FieldType.NUMBER
            or (field_type == FieldType.TIME and integer_seconds)
            else 320
        )
        text, confidence = self.predict(crop, minimum_width=minimum_width)
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
            value, score = self.predict(image, minimum_width=minimum_width)
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
