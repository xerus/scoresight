"""Synthetic smoke tests for the optional calibrated LED reader."""

import unittest

import numpy as np
from PySide6.QtCore import QRectF

from defaults import FieldType, default_info_for_box_name, normalize_settings_dict
from seven_segment import ZONES, read_score
from storage import TextDetectionTargetMemoryStorage
from tesseract import TextDetector
from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult

DIGIT_SEGMENTS = {
    "0": "1111110",
    "1": "0110000",
    "2": "1101101",
    "3": "1111001",
    "4": "0110011",
    "5": "1011011",
    "6": "1011111",
    "7": "1110000",
    "8": "1111111",
    "9": "1111011",
}


def scoreboard(home="4", away="2"):
    """Render seven-segment bars into the reader's geometric sample zones."""
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    for cell_number, digit in enumerate((home, away)):
        if not digit:
            continue
        x_offset = cell_number * 100
        for active, (x0, y0, x1, y1) in zip(DIGIT_SEGMENTS[digit], ZONES):
            if active != "1":
                continue
            width, height = x1 - x0, y1 - y0
            x = int((x0 + x1) * 50 - width * 22)
            right = int((x0 + x1) * 50 + width * 22)
            y = int((y0 + y1) * 60 - height * 22)
            bottom = int((y0 + y1) * 60 + height * 22)
            frame[y:bottom, x_offset + x : x_offset + right] = (0, 0, 255)
    return frame


class SevenSegmentTests(unittest.TestCase):
    @staticmethod
    def cells(digits, color=(0, 0, 255), intensity=255):
        image = np.concatenate(
            [scoreboard(digit, "")[:, :100] for digit in digits], axis=1
        )
        image[np.any(image > 0, axis=2)] = tuple(
            channel * intensity // 255 for channel in color
        )
        return image

    def test_single_digits_and_three_digit_basketball_scores(self):
        for digit in DIGIT_SEGMENTS:
            with self.subTest(digit=digit):
                self.assertEqual(
                    read_score(self.cells([digit]), digit_count=1)[0], digit
                )
        for digits, expected in (
            (["1", "0", "8"], "108"),
            (["", "9", "9"], "99"),
            (["", "", "7"], "7"),
            (["0", "0", "0"], "000"),
            (["", "", ""], ""),
        ):
            with self.subTest(digits=digits):
                self.assertEqual(
                    read_score(self.cells(digits), digit_count=3)[0], expected
                )

    def test_missing_middle_or_final_digit_is_unknown(self):
        for digits in (["1", "", "8"], ["1", "0", ""], ["7", ""]):
            with self.subTest(digits=digits):
                self.assertIsNone(
                    read_score(self.cells(digits), digit_count=len(digits))[0]
                )

    def test_green_other_colors_and_dim_leds(self):
        green = self.cells(["2", "4"], color=(0, 255, 0))
        self.assertEqual(read_score(green, led_color=1)[0], "24")
        self.assertEqual(read_score(green, led_color=2)[0], "24")
        self.assertEqual(read_score(green)[0], "")
        blue = self.cells(["1", "7"], color=(255, 0, 0))
        self.assertEqual(read_score(blue, led_color=2)[0], "17")
        dim = self.cells(["7", "1"], intensity=100)
        self.assertEqual(read_score(dim, brightness=80)[0], "71")
        self.assertEqual(read_score(dim)[0], "")
        white = self.cells(["1", "7"], color=(255, 255, 255))
        self.assertEqual(read_score(white, led_color=2)[0], "")

    def test_invalid_digit_count_and_color_are_rejected(self):
        for count in (0, 4, "3", 2.0, True):
            with self.subTest(count=count):
                self.assertIsNone(read_score(scoreboard(), digit_count=count)[0])
        self.assertIsNone(read_score(scoreboard(), led_color=99)[0])

    def test_calibration_defaults_validation_and_storage(self):
        info = default_info_for_box_name("Home Score")
        defaults = normalize_settings_dict({}, info)
        self.assertEqual(defaults["segment_digits"], 2)
        self.assertEqual(defaults["segment_color"], 0)
        self.assertEqual(defaults["segment_brightness"], 140)
        invalid = normalize_settings_dict(
            {"segment_digits": 4, "segment_color": "green", "segment_brightness": 0},
            info,
        )
        for key in ("segment_digits", "segment_color", "segment_brightness"):
            self.assertEqual(invalid[key], defaults[key])
        calibration = {
            "segment_digits": 3,
            "segment_color": 1,
            "segment_brightness": 80,
        }
        settings = normalize_settings_dict(calibration, info)
        target = TextDetectionTarget(0, 0, 300, 120, "Home Score", settings)
        storage = TextDetectionTargetMemoryStorage()
        storage.clear()
        try:
            storage.add_item(target)
            saved = storage.getBoxesForStorage()[0]["settings"]
            restored = normalize_settings_dict(saved, info)
            for key, value in calibration.items():
                self.assertEqual(restored[key], value)
        finally:
            storage.clear()

    def test_detector_uses_calibration_and_normalizes_numeric_zero(self):
        # Exercise real dispatch without loading unrelated Tesseract models.
        detector = TextDetector.__new__(TextDetector)
        detector.ocr_model_index = TextDetector.OcrModelIndex.OPENCV_SEVEN_SEGMENT
        target = TextDetectionTarget(
            0,
            0,
            300,
            120,
            "Home Score",
            {
                "type": FieldType.NUMBER,
                "segment_digits": 3,
                "segment_color": 1,
                "segment_brightness": 80,
                "remove_leading_zeros": True,
                "format_regex": r"[0-9]{1,3}",
            },
        )
        for digits, expected in ((["1", "0", "8"], "108"), (["0", "0", "0"], "0")):
            image = self.cells(digits, color=(0, 255, 0), intensity=100)
            binary = np.zeros(image.shape[:2], dtype=np.uint8)
            result = detector.detect_multi_text(binary, binary, [target], color=image)[
                0
            ]
            self.assertEqual(
                result.state, TextDetectionTargetWithResult.ResultState.Success
            )
            self.assertEqual(result.text, expected)
            self.assertTrue(np.any(binary))

    def test_two_digits_and_unlit_slots(self):
        self.assertEqual(read_score(scoreboard("1", "0"))[0], "10")
        self.assertEqual(read_score(scoreboard("", "4"))[0], "4")
        self.assertEqual(read_score(scoreboard("", ""))[0], "")

    def test_unknown_pattern_is_rejected(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        x0, y0, x1, y1 = ZONES[6]
        image[int(y0 * 120) : int(y1 * 120), int(x0 * 100) : int(x1 * 100)] = (
            0,
            0,
            255,
        )
        self.assertIsNone(read_score(image)[0])

    def test_detector_preserves_empty_and_rejects_non_numeric(self):
        detector = TextDetector()
        detector.setOcrModel(TextDetector.OcrModelIndex.OPENCV_SEVEN_SEGMENT)
        image = scoreboard("", "0")
        binary = np.zeros(image.shape[:2], dtype=np.uint8)
        target = QRectF(0, 0, 200, 120)
        target.settings = {
            "type": FieldType.NUMBER,
            "format_regex": r"^\d{1,2}$",
        }
        target.ocrResultPerCharacterSmoother = type(
            "Smoother", (), {"clear": lambda self: None}
        )()
        result = detector.detect_multi_text(binary, binary, [target], color=image)[0]
        self.assertEqual(
            result.state, TextDetectionTargetWithResult.ResultState.Success
        )
        self.assertEqual(result.text, "0")
        self.assertTrue(np.any(binary))

        image = scoreboard("", "")
        result = detector.detect_multi_text(binary, binary, [target], color=image)[0]
        self.assertEqual(result.state, TextDetectionTargetWithResult.ResultState.Empty)
        self.assertEqual(result.text, "")

        target.settings["type"] = FieldType.TIME
        result = detector.detect_multi_text(binary, binary, [target], color=image)[0]
        self.assertEqual(
            result.state, TextDetectionTargetWithResult.ResultState.FailedFilter
        )


if __name__ == "__main__":
    unittest.main()
