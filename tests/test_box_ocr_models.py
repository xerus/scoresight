"""Mixed reader dispatch, caching, failure isolation, and basketball formats."""

import re
import unittest
from unittest.mock import Mock, patch

from defaults import (
    BOX_OCR_MODELS,
    FieldType,
    basketball_settings_for_box,
    default_info_for_box_name,
    normalize_settings_dict,
)
from storage import TextDetectionTargetMemoryStorage
from tesseract import TextDetector
from text_detection_target import (
    TextDetectionResult,
    TextDetectionTarget,
    TextDetectionTargetWithResult,
)

SUCCESS = TextDetectionTargetWithResult.ResultState.Success
FAILED = TextDetectionTargetWithResult.ResultState.FailedFilter


def target(name, model=-1, mode=0):
    return TextDetectionTarget(
        0, 0, 100, 80, name, {"box_ocr_model": model, "temporal_mode": mode}
    )


class BoxReaderTests(unittest.TestCase):
    def detector(self):
        reader = TextDetector.__new__(TextDetector)
        reader.ocr_model_index = 1
        reader._detect_selected_model = Mock(
            side_effect=lambda binary, gray, rects, color=None: [
                TextDetectionResult("12", SUCCESS) for _ in rects
            ]
        )
        return reader

    def test_mixed_models_keep_order_and_load_each_override_once(self):
        reader = self.detector()
        neural = Mock()
        neural._detect_selected_model.side_effect = (
            lambda binary, gray, rects, color=None: [
                TextDetectionResult("24", SUCCESS) for _ in rects
            ]
        )
        boxes = [target("Home", 6), target("Clock"), target("Away", 6)]
        with patch("tesseract.TextDetector", return_value=neural) as factory:
            for _ in range(2):
                results = reader.detect_multi_text(None, None, boxes)
                self.assertEqual([r.text for r in results], ["24", "12", "24"])
            factory.assert_called_once_with(6)
        self.assertEqual(
            neural._detect_selected_model.call_args.args[2], [boxes[0], boxes[2]]
        )
        self.assertEqual(reader._detect_selected_model.call_args.args[2], [boxes[1]])

    def test_failed_optional_model_rejects_only_its_fields_and_throttles_reload(self):
        reader = self.detector()
        boxes = [target("Home", 6), target("Clock")]
        with patch(
            "tesseract.TextDetector", side_effect=RuntimeError("missing model")
        ) as factory:
            with patch("tesseract.time.monotonic", return_value=10):
                first = reader.detect_multi_text(None, None, boxes, observation_time=1)
                second = reader.detect_multi_text(None, None, boxes, observation_time=2)
            self.assertEqual([r.state for r in first], [FAILED, SUCCESS])
            self.assertEqual([r.state for r in second], [FAILED, SUCCESS])
            factory.assert_called_once_with(6)
            with patch("tesseract.time.monotonic", return_value=16):
                reader.detect_multi_text(None, None, boxes, observation_time=3)
            self.assertEqual(factory.call_count, 2)

    def test_changing_reader_resets_confirmation_even_when_value_matches(self):
        reader = self.detector()
        box = target("Home", mode=1)
        self.assertEqual(
            reader.detect_multi_text(None, None, [box], observation_time=1)[0].state,
            FAILED,
        )
        self.assertEqual(
            reader.detect_multi_text(None, None, [box], observation_time=2)[0].state,
            SUCCESS,
        )
        neural = Mock()
        neural._detect_selected_model.return_value = [
            TextDetectionResult("12", SUCCESS)
        ]
        box.settings["box_ocr_model"] = 6
        with patch("tesseract.TextDetector", return_value=neural):
            self.assertEqual(
                reader.detect_multi_text(None, None, [box], observation_time=3)[
                    0
                ].state,
                FAILED,
            )
        neural._detect_selected_model.return_value = [
            TextDetectionResult("12", SUCCESS)
        ]
        self.assertEqual(
            reader.detect_multi_text(None, None, [box], observation_time=4)[0].state,
            SUCCESS,
        )

    def test_inherited_boxes_follow_global_changes_and_overrides_stay_selected(self):
        reader = self.detector()
        override = Mock()
        override._detect_selected_model.return_value = [
            TextDetectionResult("24", SUCCESS)
        ]
        boxes = [target("Home", 6), target("Clock")]
        with patch("tesseract.TextDetector", return_value=override) as factory:
            reader.detect_multi_text(None, None, boxes)
            reader.ocr_model_index = 2
            reader.detect_multi_text(None, None, boxes)
            factory.assert_called_once_with(6)
        self.assertEqual(reader._detect_selected_model.call_count, 2)

    def test_invalid_model_inherits_and_valid_selection_round_trips(self):
        info = default_info_for_box_name("Home Score")
        for value in (4, 99, "6", True):
            self.assertEqual(
                normalize_settings_dict({"box_ocr_model": value}, info)[
                    "box_ocr_model"
                ],
                -1,
            )
        reader = self.detector()
        self.assertEqual(
            reader.detect_multi_text(None, None, [target("Home", 99)])[0].text, "12"
        )
        storage = TextDetectionTargetMemoryStorage()
        storage.clear()
        try:
            for index, model in enumerate(BOX_OCR_MODELS):
                storage.add_item(target(str(index), model))
            self.assertEqual(
                [
                    box["settings"]["box_ocr_model"]
                    for box in storage.getBoxesForStorage()
                ],
                list(BOX_OCR_MODELS),
            )
        finally:
            storage.clear()


class BasketballPresetTests(unittest.TestCase):
    def test_formats_cover_three_digit_scores_fractional_clocks_and_overtime(self):
        cases = {
            "Home Score": (["0", "108", "199"], ["1O8", "1000"], FieldType.NUMBER, 1),
            "Time": (
                ["10:00", "0:00", "59.9", "0.0"],
                ["2:99", "60.0"],
                FieldType.TIME,
                2,
            ),
            "Shot Clock": (
                ["24", "14", "9", "4.9", "00"],
                ["99", "4:9"],
                FieldType.TIME,
                2,
            ),
            "Period": (["1", "5", "6"], ["OT", "100"], FieldType.NUMBER, 1),
            "Away Fouls": (["0", "5", "10"], ["A", "100"], FieldType.NUMBER, 1),
        }
        for name, (valid, invalid, field_type, mode) in cases.items():
            profile = basketball_settings_for_box(name)
            with self.subTest(name=name):
                self.assertEqual(profile["type"], field_type)
                self.assertEqual(profile["temporal_mode"], mode)
                for text in valid:
                    self.assertIsNotNone(
                        re.fullmatch(profile["format_regex"], text), text
                    )
                for text in invalid:
                    self.assertIsNone(re.fullmatch(profile["format_regex"], text), text)
                self.assertNotIn("box_ocr_model", profile)
                self.assertNotIn("segment_digits", profile)
        self.assertIsNone(basketball_settings_for_box("Custom"))


if __name__ == "__main__":
    unittest.main()
