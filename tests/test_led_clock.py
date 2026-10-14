"""Calibrated digits and observed clock punctuation, including confirmation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
from PySide6.QtCore import QRectF

from defaults import FieldType, default_info_for_box_name, normalize_settings_dict
from seven_segment import read_display
from storage import TextDetectionTargetMemoryStorage
from source_view import ImageViewer
from tesseract import TextDetector
from test_seven_segment import scoreboard
from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult


def display(text, color=(0, 255, 0), hide_separator=False):
    """Use separated cells, leaving room for narrow punctuation and margins."""
    parts, cells, left = [], [], 0
    for char in text:
        width = 20 if char in ":." else 100
        cell = np.zeros((120, width, 3), np.uint8)
        if char in ":." and not hide_separator:
            for y in (36, 82) if char == ":" else (106,):
                cv2.circle(cell, (10, y), 4, color, -1)
        elif char.isdigit():
            cell = scoreboard(char, "")[:, :100]
            cell[np.any(cell > 0, axis=2)] = color
        cells.append((left, 0, width, 120))
        parts.extend([cell, np.zeros((120, 6, 3), np.uint8)])
        left += width + 6
    return np.concatenate(parts, axis=1), cells


class LedClockTests(unittest.TestCase):
    def test_cell_edits_are_visible_when_storage_emits_change(self):
        box = TextDetectionTarget(0, 0, 100, 80, "Time")
        cells = [QRectF(10, 0, 30, 80)]
        observed = []
        storage = SimpleNamespace(
            get_data=lambda: [box],
            edit_item=Mock(
                side_effect=lambda name, target: observed.append(target.mini_rects)
            ),
        )
        ImageViewer.boxChanged(
            SimpleNamespace(detectionTargetsStorage=storage),
            "Time",
            QRectF(0, 0, 100, 80),
            cells,
        )
        self.assertEqual(observed, [cells])

    def test_explicit_clock_layouts_and_nonuniform_numeric_cells(self):
        for text, layout in (
            ("12:34", "dd:dd"),
            ("6:42", "d:dd"),
            ("09.8", "dd.d"),
            ("0.0", "d.d"),
            ("108", "number"),
        ):
            with self.subTest(text=text):
                image, cells = display(text)
                result, mask, details = read_display(
                    image,
                    layout=layout,
                    cells=list(reversed(cells)),
                    digit_count=3,
                    led_color=1,
                )
                self.assertEqual(result, text)
                self.assertEqual(mask.shape, image.shape[:2])
                self.assertEqual(len(details), len(text))

    def test_unlit_leading_minutes_allowed_but_missing_seconds_rejected(self):
        image, cells = display(" 6:42")
        self.assertEqual(
            read_display(image, layout="dd:dd", cells=cells, led_color=1)[0], "6:42"
        )
        for text in ("12: 4", "12:3 ", "  :34", "12:99"):
            with self.subTest(text=text):
                image, cells = display(text)
                self.assertIsNone(
                    read_display(image, layout="dd:dd", cells=cells, led_color=1)[0]
                )

    def test_separators_must_be_observed_and_correctly_shaped(self):
        for text, layout in (("12:34", "dd:dd"), ("4.9", "d.d")):
            image, cells = display(text, hide_separator=True)
            self.assertIsNone(
                read_display(image, layout=layout, cells=cells, led_color=1)[0]
            )
        image, cells = display("12:34")
        x, _, width, height = cells[2]
        image[:60, x : x + width] = 0  # One colon dot cannot establish punctuation.
        self.assertIsNone(
            read_display(image, layout="dd:dd", cells=cells, led_color=1)[0]
        )
        image[:, x : x + width] = 0
        cv2.line(image, (x + 10, 20), (x + 10, 100), (0, 255, 0), 5)
        self.assertIsNone(
            read_display(image, layout="dd:dd", cells=cells, led_color=1)[0]
        )
        image, cells = display("4.9", hide_separator=True)
        cv2.circle(image, (cells[1][0] + 10, 30), 4, (0, 255, 0), -1)
        self.assertIsNone(
            read_display(image, layout="d.d", cells=cells, led_color=1)[0]
        )

    def test_physical_blank_and_zero_are_distinct(self):
        image, cells = display("00:00")
        self.assertEqual(
            read_display(image, layout="dd:dd", cells=cells, led_color=1)[0], "00:00"
        )
        image[:] = 0
        self.assertEqual(
            read_display(image, layout="dd:dd", cells=cells, led_color=1)[0], ""
        )

    def test_invalid_and_missing_calibration_is_rejected(self):
        image, cells = display("12:34")
        for calibration in (
            None,
            cells[:-1],
            [(0, 0, 0, 120)] * 5,
            [(0, 0, 100, 120)] * 5,
            [(0, 0, 9999, 120)] * 5,
            [(float("nan"), 0, 100, 120)] * 5,
        ):
            with self.subTest(cells=calibration):
                self.assertIsNone(
                    read_display(image, layout="dd:dd", cells=calibration, led_color=1)[
                        0
                    ]
                )

    def test_clock_dispatch_confirms_ticks_and_restarts_after_cell_edit(self):
        reader = TextDetector.__new__(TextDetector)
        reader.ocr_model_index = TextDetector.OcrModelIndex.OPENCV_SEVEN_SEGMENT
        image, cells = display("1:02")
        box = TextDetectionTarget(
            0,
            0,
            image.shape[1],
            image.shape[0],
            "Time",
            {
                "type": FieldType.TIME,
                "segment_layout": "d:dd",
                "segment_color": 1,
                "composite_box": True,
                "temporal_mode": 2,
                "ordinal_indicator": True,
                "remove_leading_zeros": True,
            },
            [QRectF(*cell) for cell in cells],
        )
        state = TextDetectionTargetWithResult.ResultState
        first = reader.detect_multi_text(
            None, None, [box], color=image, observation_time=1
        )[0]
        self.assertEqual(first.state, state.FailedFilter)
        image, _ = display("1:01")
        second = reader.detect_multi_text(
            None, None, [box], color=image, observation_time=2
        )[0]
        self.assertEqual((second.text, second.state), ("1:01", state.Success))
        # A calibration edit resets history even when the next text is identical.
        box.mini_rects[0].setWidth(99)
        third = reader.detect_multi_text(
            None, None, [box], color=image, observation_time=3
        )[0]
        self.assertEqual(third.state, state.FailedFilter)

    def test_layout_and_cells_round_trip_through_box_storage(self):
        settings = normalize_settings_dict(
            {"type": FieldType.TIME, "segment_layout": "dd.d", "composite_box": True},
            default_info_for_box_name("Time"),
        )
        self.assertEqual(
            normalize_settings_dict({"segment_layout": "bad"}, None)["segment_layout"],
            "number",
        )
        image, cells = display("04.9")
        box = TextDetectionTarget(
            0,
            0,
            image.shape[1],
            120,
            "Time",
            settings,
            [QRectF(*cell) for cell in cells],
        )
        storage = TextDetectionTargetMemoryStorage()
        storage.clear()
        try:
            storage.add_item(box)
            saved = storage.getBoxesForStorage()[0]
            self.assertEqual(saved["settings"]["segment_layout"], "dd.d")
            self.assertEqual(len(saved["mini_rects"]), 4)
        finally:
            storage.clear()


if __name__ == "__main__":
    unittest.main()
