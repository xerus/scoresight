"""Neural OCR agreement, confidence, and output-state regressions."""

import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from defaults import FieldType
from dnn_ocr import DnnOcr
from tesseract import TextDetector
from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult


class DnnOcrTests(unittest.TestCase):
    def recognizer(self, predictions):
        reader = DnnOcr.__new__(DnnOcr)
        reader.predict = Mock(side_effect=predictions)
        return reader

    def test_confident_native_reading_needs_only_one_call(self):
        reader = self.recognizer([("17", 0.95)])
        text, confidence, readings = reader.recognize(np.zeros((40, 80, 3), np.uint8))
        self.assertEqual((text, confidence, len(readings)), ("17", 0.95, 1))

    def test_two_shears_can_correct_native_one_to_seven(self):
        reader = self.recognizer([("1", 0.62), ("1", 0.52), ("7", 0.83), ("7", 0.60)])
        text, _, readings = reader.recognize(np.zeros((40, 80, 3), np.uint8))
        self.assertEqual(text, "7")
        self.assertEqual(len(readings), 4)

    def test_invalid_confident_native_reading_still_retries(self):
        reader = self.recognizer(
            [("5O", 0.99), ("50", 0.77), ("5O", 0.68), ("50", 0.80)]
        )
        self.assertEqual(reader.recognize(np.zeros((40, 80, 3), np.uint8))[0], "50")

    def test_disagreement_and_empty_decoding_are_unknown(self):
        for predictions in (
            [("1", 0.6), ("1", 0.7), ("7", 0.8), ("", 0.0)],
            [("", 0.0)] * 4,
        ):
            with self.subTest(predictions=predictions):
                reader = self.recognizer(predictions)
                self.assertIsNone(reader.recognize(np.zeros((40, 80, 3), np.uint8))[0])

    def test_clock_votes_preserve_separator(self):
        reader = self.recognizer(
            [("2:58", 0.7), ("2:58", 0.8), ("2:58", 0.8), ("2.58", 0.8)]
        )
        self.assertEqual(
            reader.recognize(np.zeros((40, 80, 3), np.uint8), FieldType.TIME)[0], "2:58"
        )

    def test_ctc_preserves_repeated_digits_separated_by_blank(self):
        reader = DnnOcr.__new__(DnnOcr)
        reader.characters = ["", "1", "7"]
        reader.input_name = "image"
        probabilities = np.zeros((1, 7, 3), np.float32)
        for position, index in enumerate([0, 1, 1, 0, 1, 2, 2]):
            probabilities[0, position, index] = 1
        reader.session = Mock()
        reader.session.run.return_value = [probabilities]
        self.assertEqual(reader.predict(np.zeros((40, 80, 3), np.uint8)), ("117", 1.0))
        tensor = reader.session.run.call_args[0][1]["image"]
        self.assertEqual(tensor.shape, (1, 3, 48, 320))
        self.assertTrue(np.all(tensor[0, :, :, :96] == -1))
        self.assertTrue(np.all(tensor[0, :, :, 96:] == 0))


class DnnIntegrationTests(unittest.TestCase):
    def detector(self, prediction):
        detector = TextDetector.__new__(TextDetector)
        detector.api_lock = threading.Lock()
        detector.api = Mock()
        detector.dnn = Mock()
        detector.dnn.recognize.return_value = prediction
        detector.ocr_model_index = TextDetector.OcrModelIndex.PPOCR_TTA
        return detector

    def test_zero_is_valid_but_unknown_and_empty_cannot_clear_output(self):
        states = TextDetectionTargetWithResult.ResultState
        target = TextDetectionTarget(
            5, 5, 10, 10, "Score", {"remove_leading_zeros": True}
        )
        for text, expected_state, expected_text in (
            ("00", states.Success, "0"),
            (None, states.FailedFilter, ""),
            ("", states.FailedFilter, ""),
        ):
            with self.subTest(text=text):
                detector = self.detector((text, 0.9, []))
                result = detector.detect_multi_text(
                    None, None, [target], color=np.zeros((20, 20, 3), np.uint8)
                )[0]
                self.assertEqual(
                    (result.state, result.text), (expected_state, expected_text)
                )

    def test_confidence_regex_and_out_of_frame_crops_are_rejected(self):
        states = TextDetectionTargetWithResult.ResultState
        for settings in (
            {"conf_thresh": 0.95},
            {"format_regex": "^[0-5]$"},
            {"format_regex": "["},
        ):
            target = TextDetectionTarget(5, 5, 10, 10, "Score", settings)
            result = self.detector(("7", 0.9, [])).detect_multi_text(
                None, None, [target], color=np.zeros((20, 20, 3), np.uint8)
            )[0]
            self.assertEqual(result.state, states.FailedFilter)
        detector = self.detector(("7", 0.9, []))
        target = TextDetectionTarget(15, 5, 10, 10, "Score")
        self.assertEqual(
            detector.detect_multi_text(
                None, None, [target], color=np.zeros((20, 20, 3), np.uint8)
            )[0].state,
            states.FailedFilter,
        )
        detector.dnn.recognize.assert_not_called()

    def test_model_is_cached_and_failed_load_preserves_previous_engine(self):
        detector = self.detector(("7", 0.9, []))
        cached = detector.dnn
        detector.setOcrModel(TextDetector.OcrModelIndex.PPOCR_TTA)
        self.assertIs(detector.dnn, cached)
        detector.api = Mock()
        detector.dnn = None
        detector.ocr_model_index = TextDetector.OcrModelIndex.GENERAL_ENGLISH
        with patch("dnn_ocr.DnnOcr", side_effect=RuntimeError("broken model")):
            with self.assertRaises(RuntimeError):
                detector.setOcrModel(TextDetector.OcrModelIndex.PPOCR_TTA)
        self.assertEqual(
            detector.ocr_model_index, TextDetector.OcrModelIndex.GENERAL_ENGLISH
        )
        detector.api.End.assert_not_called()


if __name__ == "__main__":
    unittest.main()
