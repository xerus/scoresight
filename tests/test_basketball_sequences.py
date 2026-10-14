"""Replay reviewed camera crops through OCR and publication confirmation."""

import importlib.util
import json
from pathlib import Path
import unittest

from dnn_ocr import DnnOcr

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/en_PP-OCRv4_rec_mobile.onnx"
FIXTURES = ROOT / "tests/fixtures/basketball/sequences"


@unittest.skipUnless(
    importlib.util.find_spec("onnxruntime") is not None and MODEL.is_file(),
    "Requires ONNX Runtime and the bundled PP-OCRv4 model",
)
class BasketballSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "sequence_readers", ROOT / "tools/ocr_benchmark/sequence_readers.py"
        )
        cls.benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.benchmark)
        cls.reader = DnnOcr(str(MODEL))
        cls.sequences = json.loads((FIXTURES / "manifest.json").read_text())[
            "sequences"
        ]

    def replay(self, name, clock_input=None):
        sequence = next(s for s in self.sequences if s["name"] == name)
        result = self.benchmark.evaluate_sequence(
            sequence,
            FIXTURES,
            self.reader,
            True,
            sequence["clock_input"] if clock_input is None else clock_input,
        )
        self.assertEqual(result["summary"]["accepted_wrong"], 0)
        self.assertGreater(result["summary"]["accepted_correct"], 0)
        return result

    def test_score_change_acquires_seven_without_publishing_one(self):
        result = self.replay("score_5_to_7")
        self.assertEqual(result["events"][-1]["expected"], "7")
        self.assertIsNotNone(result["events"][-1]["delay_seconds"])
        self.assertLessEqual(result["events"][-1]["delay_seconds"], 0.6)

    def test_shot_clock_retains_leading_one(self):
        result = self.replay("shot_clock_fourteen", 5)
        self.assertEqual(result["summary"]["wrong_published_samples"], 0)

    def test_stopped_clock_is_acquired_and_clipped_frames_cannot_change_it(self):
        result = self.replay("game_clock_stopped")
        self.assertEqual(result["summary"]["wrong_published_samples"], 0)

    def test_blank_shot_clock_does_not_invent_zero_before_reset(self):
        result = self.replay("shot_clock_reset", 5)
        self.assertEqual(result["summary"]["wrong_published_samples"], 0)
        self.assertEqual(result["events"][-1]["expected"], "24")
        self.assertIsNotNone(result["events"][-1]["delay_seconds"])
