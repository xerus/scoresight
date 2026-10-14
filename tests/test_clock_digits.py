"""Explicit compact-clock input must not guess absent or invalid digits."""

import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QRectF

from clock_digits import format_clock_digits
from defaults import FieldType, default_info_for_box_name, normalize_settings_dict
from storage import TextDetectionTargetMemoryStorage
from tesseract import TextDetector
from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult
from test_led_clock import display


class CompactClockTests(unittest.TestCase):
    def test_tesseract_uses_the_same_explicit_clock_format_before_filtering(self):
        reader = TextDetector.__new__(TextDetector)
        reader.api_lock = threading.Lock()
        reader.ocr_model_index = TextDetector.OcrModelIndex.GENERAL_ENGLISH
        reader.api = Mock()
        reader.api.GetUTF8Text.return_value = "642"
        reader.api.MeanTextConf.return_value = 90
        settings = normalize_settings_dict(
            {
                "type": FieldType.TIME,
                "clock_input": 1,
                "format_regex": r"[0-9]:[0-5][0-9]",
            },
            default_info_for_box_name("Time"),
        )
        box = TextDetectionTarget(0, 0, 80, 40, "Time", settings)
        with patch("tesseract.iterate_level", return_value=[]):
            result = reader.detect_multi_text(
                None, None, [box], color=np.zeros((40, 80, 3), np.uint8)
            )[0]
        self.assertEqual(
            (result.text, result.state),
            ("6:42", TextDetectionTargetWithResult.ResultState.Success),
        )

    def test_explicit_layouts_and_already_formatted_values(self):
        for mode, raw, expected in (
            (1, "642", "6:42"),
            (2, "0642", "06:42"),
            (3, "498", "49.8"),
            (4, "49", "4.9"),
        ):
            with self.subTest(mode=mode):
                self.assertEqual(format_clock_digits(raw, mode), expected)
                self.assertEqual(format_clock_digits(expected, mode), expected)
        self.assertEqual(format_clock_digits("720", 0), "720")
        self.assertEqual(format_clock_digits("", 1), "")
        self.assertEqual(format_clock_digits("14", 5), "14")
        self.assertIsNone(format_clock_digits("60", 5))
        self.assertIsNone(format_clock_digits("4.9", 5))

    def test_missing_extra_invalid_and_wrong_separator_inputs_are_rejected(self):
        for raw, mode in (
            ("42", 1),
            ("1642", 1),
            ("642", 2),
            ("699", 1),
            ("609", 3),
            ("4.2", 1),
            ("6:42", 2),
            ("6O2", 1),
        ):
            with self.subTest(raw=raw, mode=mode):
                self.assertIsNone(format_clock_digits(raw, mode))

    def test_neural_formatting_precedes_regex_and_countdown_confirmation(self):
        reader = TextDetector.__new__(TextDetector)
        reader.api_lock = threading.Lock()
        reader.ocr_model_index = TextDetector.OcrModelIndex.PPOCR_TTA
        reader.dnn = Mock()
        box = TextDetectionTarget(
            0,
            0,
            80,
            40,
            "Time",
            {
                "type": FieldType.TIME,
                "clock_input": 1,
                "format_regex": r"[0-9]:[0-5][0-9]",
                "temporal_mode": 2,
            },
        )
        image = np.zeros((40, 80, 3), np.uint8)
        states = TextDetectionTargetWithResult.ResultState
        for tick, raw, state, expected in (
            (1, "642", states.FailedFilter, "6:42"),
            (2, "641", states.Success, "6:41"),
            (3, "641", states.Success, "6:41"),
            (4, "699", states.FailedFilter, "699"),
        ):
            reader.dnn.recognize.return_value = (raw, 0.9, [])
            result = reader.detect_multi_text(
                None, None, [box], color=image, observation_time=tick
            )[0]
            self.assertEqual((result.state, result.text), (state, expected))

    def test_led_compact_cells_are_formatted_without_separator_evidence(self):
        image, cells = display("642")
        reader = TextDetector.__new__(TextDetector)
        reader.ocr_model_index = TextDetector.OcrModelIndex.OPENCV_SEVEN_SEGMENT
        box = TextDetectionTarget(
            0,
            0,
            image.shape[1],
            120,
            "Time",
            {
                "type": FieldType.TIME,
                "clock_input": 1,
                "segment_layout": "ddd",
                "segment_color": 1,
                "composite_box": True,
                "format_regex": r"[0-9]:[0-5][0-9]",
            },
            [QRectF(*cell) for cell in cells],
        )
        result = reader.detect_multi_text(None, None, [box], color=image)[0]
        self.assertEqual(
            (result.text, result.state),
            ("6:42", TextDetectionTargetWithResult.ResultState.Success),
        )

    def test_compact_mode_is_opt_in_and_saved(self):
        info = default_info_for_box_name("Time")
        self.assertEqual(normalize_settings_dict({}, info)["clock_input"], 0)
        self.assertEqual(
            normalize_settings_dict({"clock_input": True}, info)["clock_input"], 0
        )
        storage = TextDetectionTargetMemoryStorage()
        storage.clear()
        try:
            storage.add_item(
                TextDetectionTarget(
                    0,
                    0,
                    80,
                    40,
                    "Time",
                    normalize_settings_dict({"clock_input": 1}, info),
                )
            )
            self.assertEqual(
                storage.getBoxesForStorage()[0]["settings"]["clock_input"], 1
            )
        finally:
            storage.clear()


if __name__ == "__main__":
    unittest.main()
