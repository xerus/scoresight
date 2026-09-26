"""The preview can hide box names without hiding OCR results."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from resizable_rect import ResizableRectWithNameTypeAndResult
from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult


class BoxDisplayStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_results_without_names_even_when_selected(self):
        target = TextDetectionTarget(10, 20, 100, 50, "Home Score")
        rect = ResizableRectWithNameTypeAndResult(
            target, image_size=1000, result="2", boxDisplayStyle=4
        )

        self.assertTrue(rect.isVisible())
        self.assertFalse(rect.posItem.isVisible())
        self.assertFalse(rect.bgItem.isVisible())
        self.assertTrue(rect.resultItem.isVisible())

        rect.setSelected(True)
        self.assertFalse(rect.posItem.isVisible())
        self.assertFalse(rect.bgItem.isVisible())

        result = TextDetectionTargetWithResult(
            target,
            "3",
            TextDetectionTargetWithResult.ResultState.Success,
            effectiveRect=QRectF(5, 5, 90, 40),
        )
        rect.updateResult(result)
        self.assertEqual(rect.resultItem.text(), "3")
        self.assertIsNotNone(rect.effectiveRect)

        rect.setSelected(False)
        rect.setBoxDisplayStyle(3)
        self.assertTrue(rect.posItem.isVisible())
        self.assertTrue(rect.bgItem.isVisible())


if __name__ == "__main__":
    unittest.main()
