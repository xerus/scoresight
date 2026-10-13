"""Real ONNX inference on lossless camera fixtures, without source videos."""

import importlib.util
import json
from pathlib import Path
import unittest

import cv2

from dnn_ocr import DnnOcr

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/basketball"
MODEL = ROOT / "models/en_PP-OCRv4_rec_mobile.onnx"


@unittest.skipUnless(
    importlib.util.find_spec("onnxruntime") is not None and MODEL.is_file(),
    "Requires ONNX Runtime and the bundled PP-OCRv4 model",
)
class BasketballCameraOcrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DnnOcr(str(MODEL))
        cls.frames = json.loads((FIXTURES / "manifest.json").read_text())["frames"]

    def readings(self, names):
        for frame in self.frames:
            if frame["image"] not in names:
                continue
            image = cv2.imread(str(FIXTURES / frame["image"]))
            self.assertIsNotNone(image)
            for field in frame["fields"]:
                x, y, width, height = field["roi"]
                crop = image[y : y + height, x : x + width]
                yield frame["image"], field, self.reader.recognize(crop)

    def test_zero_seven_and_three_digit_scores(self):
        for image, field, (text, confidence, _) in self.readings(
            {"zero.png", "seven.png", "three_digit.png"}
        ):
            with self.subTest(image=image, field=field["name"]):
                self.assertEqual(text, field["expected"])
                self.assertGreaterEqual(confidence, 0.5)

    def test_thin_digits_are_correct_or_unknown(self):
        for image, field, (text, confidence, _) in self.readings(
            {"eleven.png", "foul_one.png"}
        ):
            with self.subTest(image=image, field=field["name"]):
                self.assertIn(text, (None, field["expected"]))
                if text is not None:
                    self.assertGreaterEqual(confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
