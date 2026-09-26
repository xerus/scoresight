"""A confirmed blank clears output only when the user allows empty values."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from file_output import save_text_files
from text_detection_target import TextDetectionTargetWithResult


class BlankOutputTests(unittest.TestCase):
    def test_blank_zero_and_uncertain_are_distinct(self):
        state = TextDetectionTargetWithResult.ResultState
        cases = [
            (state.Empty, "", False, ""),
            (state.Empty, "", True, "previous"),
            (state.Success, "0", False, "0"),
            (state.FailedFilter, "", False, "previous"),
            (state.FailedFilter, None, False, "previous"),
        ]
        for result_state, value, skip_empty, expected in cases:
            with self.subTest(state=result_state, value=value, skip_empty=skip_empty):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "Away Score.txt"
                    output.write_text("previous")
                    target = SimpleNamespace(
                        name="Away Score",
                        result=value,
                        result_state=result_state,
                        settings={"skip_empty": skip_empty},
                    )
                    save_text_files([target], directory, 0)
                    self.assertEqual(output.read_text(), expected)


if __name__ == "__main__":
    unittest.main()
