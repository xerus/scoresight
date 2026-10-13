"""Verify field-specific padding reaches ONNX without squeezing wide crops."""

import math
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from defaults import FieldType
from dnn_ocr import DnnOcr


class RecordingSession:
    def __init__(self, confidence=0.95):
        self.confidence = confidence
        self.inputs = []

    def run(self, outputs, inputs):
        self.inputs.append(inputs["image"].copy())
        # A blank then a seven is enough to exercise the real CTC decoder.
        return [np.array([[[1.0, 0.0], [0.0, self.confidence]]], np.float32)]


class DnnPaddingTests(unittest.TestCase):
    def reader(self, session=None):
        reader = DnnOcr.__new__(DnnOcr)
        reader.input_name = "image"
        reader.characters = ["", "7"]
        reader.session = session if session is not None else RecordingSession()
        return reader

    def test_predict_default_preserves_legacy_320_width(self):
        reader = self.reader()
        self.assertEqual(reader.predict(np.zeros((40, 80, 3), np.uint8)), ("7", 0.95))
        tensor = reader.session.inputs[0]
        self.assertEqual(tensor.shape, (1, 3, 48, 320))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(tensor[0, :, :, :96] == -1))
        self.assertTrue(np.all(tensor[0, :, :, 96:] == 0))

    def test_numeric_padding_uses_128_width_without_changing_pixels(self):
        reader = self.reader()
        crop = np.full((40, 80, 3), (0, 127, 255), np.uint8)
        reader.predict(crop)
        reader.predict(crop, minimum_width=128)
        default, numeric = reader.session.inputs
        self.assertEqual(numeric.shape, (1, 3, 48, 128))
        np.testing.assert_array_equal(numeric, default[:, :, :, :128])
        self.assertTrue(np.all(numeric[0, :, :, 96:] == 0))

    def test_wide_inputs_keep_entire_aspect_ratio_at_either_minimum_width(self):
        # Fractional widths round up both below and beyond the old 320 minimum.
        for crop_width in (80, 201):
            crop = np.zeros((29, crop_width, 3), np.uint8)
            crop[:, -10:] = (255, 127, 0)
            scaled_width = math.ceil(48 * crop.shape[1] / crop.shape[0])
            expected = (
                cv2.resize(crop, (scaled_width, 48))
                .astype(np.float32)
                .transpose(2, 0, 1)
                / 127.5
                - 1
            )
            for minimum_width in (128, 320):
                with self.subTest(crop_width=crop_width, minimum_width=minimum_width):
                    reader = self.reader()
                    reader.predict(crop, minimum_width=minimum_width)
                    tensor = reader.session.inputs[0]
                    self.assertEqual(
                        tensor.shape, (1, 3, 48, max(minimum_width, scaled_width))
                    )
                    np.testing.assert_array_equal(
                        tensor[0, :, :, :scaled_width], expected
                    )
                    self.assertTrue(np.all(tensor[0, :, :, scaled_width:] == 0))

    def test_field_types_choose_numeric_or_original_padding(self):
        reader = self.reader()
        crop = np.zeros((40, 80, 3), np.uint8)
        for field_type, width in (
            (FieldType.NUMBER, 128),
            (FieldType.TIME, 320),
            (FieldType.TEXT, 320),
            (FieldType.NUMBER, 128),
        ):
            with self.subTest(field_type=field_type):
                self.assertEqual(reader.recognize(crop, field_type)[0], "7")
                self.assertEqual(reader.session.inputs[-1].shape, (1, 3, 48, width))

    def test_all_tta_variants_use_the_same_field_padding(self):
        crop = np.zeros((40, 80, 3), np.uint8)
        for field_type, width in (
            (FieldType.NUMBER, 128),
            (FieldType.TIME, 320),
            (FieldType.TEXT, 320),
        ):
            with self.subTest(field_type=field_type):
                reader = self.reader(RecordingSession(confidence=0.6))
                text, _, readings = reader.recognize(crop, field_type)
                self.assertEqual(text, "7")
                self.assertEqual(
                    [reading["variant"] for reading in readings],
                    ["native", "inverted_gray", "shear_-0.25", "shear_0.25"],
                )
                self.assertEqual(
                    [tensor.shape[-1] for tensor in reader.session.inputs], [width] * 4
                )

    def test_reentrant_text_reading_cannot_change_numeric_tta_padding(self):
        session = RecordingSession(confidence=0.6)
        reader = self.reader(session)
        crop = np.zeros((40, 80, 3), np.uint8)
        original_run = session.run
        entered = False

        def reentrant_run(outputs, inputs):
            nonlocal entered
            prediction = original_run(outputs, inputs)
            if not entered:
                entered = True
                self.assertEqual(reader.recognize(crop, FieldType.TEXT)[0], "7")
            return prediction

        session.run = reentrant_run
        self.assertEqual(reader.recognize(crop, FieldType.NUMBER)[0], "7")
        self.assertEqual(
            [tensor.shape[-1] for tensor in session.inputs],
            [128, 320, 320, 320, 320, 128, 128, 128],
        )

    def test_concurrent_field_readings_keep_independent_tta_padding(self):
        session = RecordingSession(confidence=0.6)
        reader = self.reader(session)
        original_run = session.run
        rendezvous = threading.Barrier(2)
        local = threading.local()
        widths = {}
        widths_lock = threading.Lock()

        def concurrent_run(outputs, inputs):
            prediction = original_run(outputs, inputs)
            with widths_lock:
                widths.setdefault(threading.get_ident(), []).append(
                    inputs["image"].shape[-1]
                )
            if not getattr(local, "entered", False):
                local.entered = True
                rendezvous.wait(timeout=5)
            return prediction

        session.run = concurrent_run
        crop = np.zeros((40, 80, 3), np.uint8)
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [
                workers.submit(reader.recognize, crop, field_type)
                for field_type in (FieldType.NUMBER, FieldType.TIME)
            ]
            self.assertEqual(
                [future.result(timeout=10)[0] for future in futures], ["7", "7"]
            )
        self.assertEqual(sorted(widths.values()), [[128] * 4, [320] * 4])


if __name__ == "__main__":
    unittest.main()
