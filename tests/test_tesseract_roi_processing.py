"""ROI-scoped OCR preprocessing and input scaling tests."""

import threading
import unittest

import cv2
import numpy as np
from PySide6.QtCore import QRectF

from defaults import FieldType, default_info_for_box_name, normalize_settings_dict
from storage import TextDetectionTargetMemoryStorage
from tesseract import TextDetector
from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult


class FakeTesseract:
    def __init__(self):
        self.image_size = None

    def SetImage(self, image):
        self.image_size = image.size

    def SetVariable(self, *_args):
        pass

    def GetUTF8Text(self):
        return ""

    def MeanTextConf(self):
        return 100


class RoiProcessingTests(unittest.TestCase):
    def test_horizontal_scale_widens_ocr_patch_and_is_saved_per_box(self):
        detector = TextDetector.__new__(TextDetector)
        detector.api_lock = threading.Lock()
        detector.api = FakeTesseract()
        detector.ocr_model_index = TextDetector.OcrModelIndex.SCOREBOARD_GENERAL

        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        target = TextDetectionTarget(10, 10, 40, 30, "Home Score")
        target.settings = normalize_settings_dict(
            {"hscale": 15, "rescale_patch": False},
            default_info_for_box_name(target.name),
        )
        detector.detect_multi_text(None, None, [target], color=frame)
        self.assertEqual(detector.api.image_size, (60, 30))

        target.settings["hscale"] = 10
        detector.detect_multi_text(None, None, [target], color=frame)
        self.assertEqual(detector.api.image_size, (40, 30))

        target.settings["hscale"] = 15
        storage = TextDetectionTargetMemoryStorage()
        storage.clear()
        try:
            storage.add_item(target)
            self.assertEqual(storage.getBoxesForStorage()[0]["settings"]["hscale"], 15)
        finally:
            storage.clear()

    def test_color_input_processes_only_selected_roi_and_scales_patch(self):
        detector = TextDetector.__new__(TextDetector)
        detector.api_lock = threading.Lock()
        detector.api = FakeTesseract()
        detector.ocr_model_index = TextDetector.OcrModelIndex.SCOREBOARD_GENERAL

        frame = np.full((80, 160, 3), (40, 60, 80), dtype=np.uint8)
        frame[20:50, 20:60] = (30, 150, 230)
        preview = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preview_before = preview.copy()

        target = QRectF(20, 20, 40, 30)
        target.settings = {
            "type": FieldType.NUMBER,
            "binarization_method": TextDetector.BinarizationMethod.GLOBAL,
            "rescale_patch": True,
            "format_regex": r"^\d+$",
        }
        target.ocrResultPerCharacterSmoother = type(
            "Smoother", (), {"clear": lambda self: None}
        )()

        # Normal color-view operation can omit full-frame grayscale and binary
        # arrays; preprocessing uses only the selected color crop.
        result = detector.detect_multi_text(None, None, [target], color=frame)[0]
        self.assertEqual(result.state, TextDetectionTargetWithResult.ResultState.Empty)
        self.assertEqual(detector.api.image_size[1], 35)

        # Binary View replaces the selected region and preserves the rest.
        binary_preview = preview.copy()
        detector.detect_multi_text(binary_preview, preview, [target], color=frame)
        self.assertTrue(np.array_equal(binary_preview[:20], preview_before[:20]))
        self.assertTrue(np.array_equal(binary_preview[50:], preview_before[50:]))
        self.assertTrue(np.array_equal(binary_preview[:, :20], preview_before[:, :20]))
        self.assertTrue(np.array_equal(binary_preview[:, 60:], preview_before[:, 60:]))
        self.assertTrue(
            np.isin(np.unique(binary_preview[20:50, 20:60]), [0, 255]).all()
        )


if __name__ == "__main__":
    unittest.main()
