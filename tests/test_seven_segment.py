"""Synthetic smoke tests for the optional calibrated LED reader."""

import unittest

import numpy as np
from PySide6.QtCore import QRectF

from defaults import FieldType
from seven_segment import ZONES, read_score
from tesseract import TextDetector
from text_detection_target import TextDetectionTargetWithResult

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
