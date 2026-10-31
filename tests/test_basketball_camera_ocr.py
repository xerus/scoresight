"""Real ONNX inference on lossless camera fixtures, without source videos."""

import importlib.util
import json
from pathlib import Path
import unittest

import cv2

from dnn_ocr import DnnOcr
from defaults import FieldType
from led_digit_evidence import digit_evidence

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

    def test_green_glare_masks_recover_complete_clock_digits(self):
        from clock_digits import format_clock_digits

        root = FIXTURES / "glare"
        for row in json.loads((root / "manifest.json").read_text())["fields"]:
            with self.subTest(frame=row["frame"]):
                crop = cv2.imread(str(root / row["image"]))
                text, confidence, _ = self.reader.recognize(
                    crop, FieldType.TIME, check_led=True
                )
                self.assertEqual(format_clock_digits(text, 1), row["expected"])
                self.assertGreaterEqual(confidence, 0.5)

    def test_panel_scoreboard_seventeen_is_recovered_or_rejected_without_shortening(
        self,
    ):
        root = ROOT / "tests/fixtures/panel_scoreboard"
        for row in json.loads((root / "manifest.json").read_text())["fields"]:
            with self.subTest(image=row["image"]):
                crop = cv2.imread(str(root / row["image"]))
                text, confidence, readings = self.reader.recognize(crop, check_led=True)
                if row["recover"]:
                    self.assertEqual(text, row["expected"])
                    self.assertGreaterEqual(confidence, 0.5)
                else:
                    self.assertIn(text, (None, row["expected"]))
                json.dumps(readings)

    def test_panel_scoreboard_acquires_seventeen_with_shifted_and_wider_boxes(self):
        root = ROOT / "tests/fixtures/panel_scoreboard"
        boxes = [
            (21, 22, 54, 58),
            (15, 15, 65, 70),
            (8, 8, 80, 84),
            (10, 15, 65, 70),
            (20, 15, 65, 70),
            (15, 10, 65, 70),
            (15, 20, 65, 70),
        ]
        for frame in (0, 32, 80, 164, 240, 295, 327):
            image = cv2.imread(str(root / f"context_{frame:04}.png"))
            for box in boxes:
                x, y, width, height = box
                for check in (False, True):
                    with self.subTest(frame=frame, box=box, check_led=check):
                        text, _, _ = self.reader.recognize(
                            image[y : y + height, x : x + width], check_led=check
                        )
                        self.assertIn(text, (None, "17"))
                        # These two frames must acquire the score for every
                        # tested box, with the optional LED check on or off.
                        if frame in (32, 164):
                            self.assertEqual(text, "17")

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

    def test_led_check_recovers_ambiguous_seven_using_actual_neural_inference(self):
        row = next(
            frame for frame in self.frames if frame["image"] == "seven_ambiguous.png"
        )
        x, y, width, height = row["fields"][0]["roi"]
        image = cv2.imread(str(FIXTURES / row["image"]))
        crop = image[y : y + height, x : x + width]
        text, confidence, readings = self.reader.recognize(crop, check_led=True)
        self.assertEqual(text, "7")
        self.assertGreaterEqual(confidence, 0.5)
        self.assertTrue(
            any(reading["variant"].startswith("tight_") for reading in readings)
        )
        json.dumps(readings)  # Full HTTP JSON must be able to expose diagnostics.

    def test_real_one_shape_is_distinct_from_seven(self):
        row = next(frame for frame in self.frames if frame["image"] == "foul_one.png")
        field = next(field for field in row["fields"] if field["name"] == "away_fouls")
        x, y, width, height = field["roi"]
        image = cv2.imread(str(FIXTURES / row["image"]))
        self.assertEqual(digit_evidence(image[y : y + height, x : x + width])[0], "1")

    def test_core_mask_recovers_green_clock_and_previously_rejected_thin_digits(self):
        fields = json.loads((FIXTURES / "clock_manifest.json").read_text())["fields"]
        row = next(f for f in fields if f["expected_digits"] == "741")
        x, y, w, h = row["roi"]
        image = cv2.imread(str(FIXTURES / row["image"]))
        text, confidence, readings = self.reader.recognize(
            image[y : y + h, x : x + w], FieldType.TIME, check_led=True
        )
        self.assertEqual(text, "741")
        self.assertGreaterEqual(confidence, 0.5)
        self.assertTrue(any(r["variant"].startswith("core_") for r in readings))
        json.dumps(readings)
        for frame in self.frames:
            for field in frame["fields"]:
                if field["expected"] not in ("11", "1"):
                    continue
                x, y, w, h = field["roi"]
                image = cv2.imread(str(FIXTURES / frame["image"]))
                text, confidence, _ = self.reader.recognize(
                    image[y : y + h, x : x + w], check_led=True
                )
                self.assertEqual(text, field["expected"])
                self.assertGreaterEqual(confidence, 0.5)

    def test_led_check_preserves_leading_one_in_real_shot_clock(self):
        fields = json.loads((FIXTURES / "clock_manifest.json").read_text())["fields"]
        row = next(
            field
            for field in fields
            if field["name"] == "shot_clock" and field["expected_digits"] == "14"
        )
        x, y, width, height = row["roi"]
        image = cv2.imread(str(FIXTURES / row["image"]))
        text, confidence, _ = self.reader.recognize(
            image[y : y + height, x : x + width], FieldType.TIME, check_led=True
        )
        self.assertEqual(text, "14")
        self.assertGreaterEqual(confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
