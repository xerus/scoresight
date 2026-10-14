"""Selective LED checks must verify neural text rather than substitute guesses."""

import unittest
import threading
from unittest.mock import Mock, patch

import numpy as np
import cv2

from defaults import FieldType, default_info_for_box_name, normalize_settings_dict
from dnn_ocr import DnnOcr
from led_digit_evidence import digit_evidence
from storage import TextDetectionTargetMemoryStorage
from tesseract import TextDetector
from text_detection_target import TextDetectionTarget


class LedEvidencePolicyTests(unittest.TestCase):
    def test_clipped_cores_reject_before_running_neural_inference(self):
        reader = self.reader([])
        with patch(
            "led_digit_evidence.digit_evidence",
            return_value=(None, None, {"reason": "clipped_digits"}),
        ):
            self.assertIsNone(
                reader.recognize(np.zeros((40, 80, 3), np.uint8), check_led=True)[0]
            )
        reader._recognize.assert_not_called()

    def test_partial_terminal_shape_requires_matching_prefix_count_and_suffix(self):
        evidence = (
            None,
            (3, 3, 70, 35),
            {
                "reason": "joined_or_wide_digits",
                "min_digits": 3,
                "known_suffix": "1",
                "core_min": 240,
                "boxes": [(3, 3, 60, 32)],
            },
        )
        for candidate, expected in (
            ("741", "741"),
            ("941", None),
            ("74", None),
            ("742", None),
        ):
            with self.subTest(candidate=candidate):
                reader = self.reader(
                    [
                        (
                            None,
                            0,
                            [{"variant": "native", "text": "74", "confidence": 0.4}],
                        ),
                        (candidate, 0.95, []),
                    ]
                )
                with patch("led_digit_evidence.digit_evidence", return_value=evidence):
                    result = reader.recognize(
                        np.zeros((40, 80, 3), np.uint8), FieldType.TIME, check_led=True
                    )
                self.assertEqual(result[0], expected)

    def test_merged_stroke_requires_a_complete_tighter_neural_value(self):
        evidence = (
            None,
            (3, 3, 70, 35),
            {"reason": "unknown_digit", "boxes": [(3, 3, 60, 32)]},
        )
        for retry, expected in (("14", "14"), ("4", None), ("13", None), ("144", None)):
            with self.subTest(retry=retry):
                reader = self.reader([("4", 0.9, []), (retry, 0.95, [])])
                with patch("led_digit_evidence.digit_evidence", return_value=evidence):
                    result = reader.recognize(
                        np.zeros((40, 80, 3), np.uint8), FieldType.TIME, check_led=True
                    )
                self.assertEqual(result[0], expected)

    def test_weak_bridge_separates_one_zero_without_splitting_an_eight(self):
        image = np.zeros((90, 90, 3), np.uint8)
        cv2.rectangle(image, (5, 10), (15, 80), (255, 255, 255), -1)
        cv2.rectangle(image, (27, 10), (70, 80), (255, 255, 255), 7)
        cv2.line(image, (15, 76), (27, 76), (255, 255, 255), 2)
        self.assertEqual(digit_evidence(image)[0], "10")
        image[:] = 0
        cv2.rectangle(image, (27, 10), (70, 80), (255, 255, 255), 7)
        cv2.line(image, (27, 45), (70, 45), (255, 255, 255), 7)
        self.assertEqual(digit_evidence(image)[0], "8")

    def test_option_round_trips_and_is_forwarded_to_the_neural_reader(self):
        info = default_info_for_box_name("Home Score")
        self.assertFalse(normalize_settings_dict({}, info)["check_led"])
        self.assertFalse(
            normalize_settings_dict({"check_led": "false"}, info)["check_led"]
        )
        settings = normalize_settings_dict({"check_led": True}, info)
        box = TextDetectionTarget(0, 0, 80, 40, "Home Score", settings)
        reader = TextDetector.__new__(TextDetector)
        reader.ocr_model_index = TextDetector.OcrModelIndex.PPOCR_TTA
        reader.api_lock = threading.Lock()
        reader.dnn = Mock()
        reader.dnn.recognize.return_value = ("7", 0.9, [])
        reader.detect_multi_text(
            None, None, [box], color=np.zeros((40, 80, 3), np.uint8)
        )
        self.assertTrue(reader.dnn.recognize.call_args.kwargs["check_led"])
        storage = TextDetectionTargetMemoryStorage()
        storage.clear()
        try:
            storage.add_item(box)
            self.assertTrue(storage.getBoxesForStorage()[0]["settings"]["check_led"])
        finally:
            storage.clear()

    def reader(self, responses):
        reader = DnnOcr.__new__(DnnOcr)
        reader._recognize = Mock(side_effect=responses)
        return reader

    def test_conflicting_tighter_reading_cannot_publish_shape_only_value(self):
        reader = self.reader([("1", 0.9, []), ("1", 0.95, []), ("1", 0.95, [])])
        with patch(
            "led_digit_evidence.digit_evidence",
            return_value=("7", (3, 3, 30, 35), {"reason": "clear_digits"}),
        ):
            text, confidence, _ = reader.recognize(
                np.zeros((40, 80, 3), np.uint8), check_led=True
            )
        self.assertIsNone(text)
        self.assertEqual(confidence, 0)

    def test_agreeing_tighter_neural_reading_recovers_ambiguous_seven(self):
        reader = self.reader(
            [
                ("1", 0.6, []),
                ("7", 0.9, [{"variant": "native", "text": "7", "confidence": 0.9}]),
            ]
        )
        with patch(
            "led_digit_evidence.digit_evidence",
            return_value=("7", (3, 3, 30, 35), {"reason": "clear_digits"}),
        ):
            text, confidence, readings = reader.recognize(
                np.zeros((40, 80, 3), np.uint8), check_led=True
            )
        self.assertEqual((text, confidence), ("7", 0.9))
        self.assertEqual(reader._recognize.call_args.args[0].shape, (32, 27, 3))
        self.assertEqual(readings[-1]["variant"], "tight_native")

    def test_matching_digits_preserve_clock_punctuation_without_extra_inference(self):
        reader = self.reader([("7:20", 0.9, [])])
        with patch(
            "led_digit_evidence.digit_evidence",
            return_value=("720", (3, 3, 30, 35), {"reason": "clear_digits"}),
        ):
            self.assertEqual(
                reader.recognize(
                    np.zeros((40, 80, 3), np.uint8), FieldType.TIME, check_led=True
                )[0],
                "7:20",
            )
        self.assertEqual(reader._recognize.call_count, 1)

    def test_unknown_evidence_is_recorded_and_cannot_change_neural_result(self):
        reader = self.reader([("108", 0.9, [])])
        with patch(
            "led_digit_evidence.digit_evidence",
            return_value=(None, None, {"reason": "joined_or_wide_digits"}),
        ):
            text, _, readings = reader.recognize(
                np.zeros((40, 80, 3), np.uint8), check_led=True
            )
        self.assertEqual(text, "108")
        self.assertIsNone(readings[-1]["text"])
        self.assertEqual(reader._recognize.call_count, 1)

    def test_check_is_opt_in_and_general_text_does_not_use_it(self):
        for kind, enabled in ((FieldType.NUMBER, False), (FieldType.TEXT, True)):
            reader = self.reader([("HOME", 0.9, [])])
            with patch("led_digit_evidence.digit_evidence") as evidence:
                reader.recognize(
                    np.zeros((40, 80, 3), np.uint8), kind, check_led=enabled
                )
                evidence.assert_not_called()

    def test_blank_invalid_and_merged_shapes_do_not_establish_digits(self):
        for crop in (
            None,
            np.zeros((40, 80, 3), np.uint8),
            np.full((40, 80, 3), 255, np.uint8),
        ):
            with self.subTest(crop_type=type(crop)):
                self.assertIsNone(digit_evidence(crop)[0])


if __name__ == "__main__":
    unittest.main()
