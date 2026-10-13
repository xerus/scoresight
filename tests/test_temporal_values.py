"""Observed-value confirmation with reversible score and clock continuity."""

import unittest
from unittest.mock import Mock, patch
import threading

import cv2
import numpy as np

from defaults import default_info_for_box_name, normalize_settings_dict
from storage import TextDetectionTargetMemoryStorage
from temporal_values import TemporalValueFilter as Filter
from tesseract import TextDetector
from text_detection_target import (
    TextDetectionResult,
    TextDetectionTarget,
    TextDetectionTargetWithResult,
)


class TemporalValueTests(unittest.TestCase):
    def setUp(self):
        self.reader = Filter()
        self.now = 0

    def observe(self, value, mode=Filter.NUMBER, step=0.2, context="field"):
        self.now += step
        return self.reader.observe(value, self.now, mode, context)[0]

    def bootstrap(self, value, mode=Filter.NUMBER):
        self.assertFalse(self.observe(value, mode))
        self.assertTrue(self.observe(value, mode))

    def test_transient_seven_to_one_does_not_replace_known_score(self):
        self.bootstrap("7")
        self.assertFalse(self.observe("1"))
        self.assertFalse(self.observe("1"))
        self.assertTrue(self.observe("7"))
        self.assertEqual(self.reader.accepted[1], 7)

    def test_one_two_and_three_point_scores_are_confirmed(self):
        for increment in (1, 2, 3):
            with self.subTest(increment=increment):
                self.reader.clear()
                self.bootstrap("24")
                self.assertFalse(self.observe(str(24 + increment)))
                self.assertTrue(self.observe(str(24 + increment)))

    def test_corrections_large_jumps_and_resets_are_eventually_accepted(self):
        for value in ("1", "0", "108"):
            with self.subTest(value=value):
                self.reader.clear()
                self.bootstrap("7")
                self.assertFalse(self.observe(value))
                self.assertFalse(self.observe(value))
                self.assertTrue(self.observe(value))

    def test_invalid_frames_break_confirmation_without_inventing_zero(self):
        self.bootstrap("7")
        self.assertFalse(self.observe("1"))
        self.assertFalse(self.observe(None))
        self.assertFalse(self.observe("1"))
        self.assertFalse(self.observe("1"))
        self.assertTrue(self.observe("7"))
        self.assertFalse(self.observe(""))
        self.assertTrue(self.observe(""))
        self.assertEqual(self.reader.accepted[0], "blank")

    def test_alternating_values_never_synthesize_a_string(self):
        self.bootstrap("17")
        for value in ("71", "11") * 5:
            self.assertFalse(self.observe(value))
        self.assertEqual(self.reader.accepted[1], 17)

    def test_countdown_accepts_progress_and_stops_without_extra_tick_delay(self):
        self.bootstrap("2:58", Filter.COUNTDOWN)
        self.assertTrue(self.observe("2:57", Filter.COUNTDOWN, step=1))
        self.assertTrue(self.observe("2:57", Filter.COUNTDOWN, step=3))
        self.assertTrue(self.observe("2:56", Filter.COUNTDOWN, step=1))

    def test_one_hz_clock_bootstrap_uses_progressing_values(self):
        self.assertFalse(self.observe("2:58", Filter.COUNTDOWN, step=1))
        self.assertTrue(self.observe("2:57", Filter.COUNTDOWN, step=1))

    def test_clock_reset_can_progress_while_waiting_for_confirmation(self):
        self.bootstrap("0:07", Filter.COUNTDOWN)
        self.assertFalse(self.observe("10:00", Filter.COUNTDOWN, step=1))
        self.assertFalse(self.observe("9:59", Filter.COUNTDOWN, step=1))
        self.assertTrue(self.observe("9:58", Filter.COUNTDOWN, step=1))

    def test_clock_seven_one_error_is_held_and_recovery_is_immediate(self):
        self.bootstrap("0:07", Filter.COUNTDOWN)
        self.assertFalse(self.observe("0:01", Filter.COUNTDOWN))
        self.assertFalse(self.observe("0:01", Filter.COUNTDOWN))
        self.assertTrue(self.observe("0:06", Filter.COUNTDOWN))

    def test_fractional_count_up_clock_and_minute_rollover(self):
        self.bootstrap("59.8", Filter.COUNTUP)
        self.assertTrue(self.observe("59.9", Filter.COUNTUP))
        self.assertTrue(self.observe("1:00.0", Filter.COUNTUP))
        self.assertFalse(self.observe("0:58.0", Filter.COUNTUP))

    def test_clock_unknown_requires_reconfirmation(self):
        self.bootstrap("0:17", Filter.COUNTDOWN)
        self.assertFalse(self.observe(None, Filter.COUNTDOWN))
        self.assertFalse(self.observe("0:16", Filter.COUNTDOWN))
        self.assertTrue(self.observe("0:16", Filter.COUNTDOWN))

    def test_duplicate_video_frames_do_not_count_and_backward_seek_resets(self):
        self.assertFalse(self.reader.observe("7", 1, Filter.NUMBER, "video")[0])
        self.assertFalse(self.reader.observe("7", 1, Filter.NUMBER, "video")[0])
        self.assertTrue(self.reader.observe("7", 2, Filter.NUMBER, "video")[0])
        self.assertFalse(self.reader.observe("1", 0, Filter.NUMBER, "video")[0])
        self.assertTrue(self.reader.observe("1", 0.2, Filter.NUMBER, "video")[0])

    def test_new_geometry_source_or_settings_context_starts_fresh(self):
        self.bootstrap("7")
        self.assertFalse(self.observe("1", context="new_field"))
        self.assertTrue(self.observe("1", context="new_field"))

    def test_period_ordinals_preserve_numeric_continuity(self):
        self.bootstrap("1st")
        self.assertFalse(self.observe("2nd"))
        self.assertTrue(self.observe("2nd"))


class TemporalIntegrationTests(unittest.TestCase):
    def test_temporal_confirmation_requires_fresh_ocr_and_bypasses_character_votes(
        self,
    ):
        state = TextDetectionTargetWithResult.ResultState
        detector = TextDetector.__new__(TextDetector)
        detector.api_lock = threading.Lock()
        detector.api = Mock()
        detector.api.GetUTF8Text.return_value = "7"
        detector.api.MeanTextConf.return_value = 100
        detector.ocr_model_index = 1
        target = TextDetectionTarget(
            0,
            0,
            40,
            40,
            "Home Score",
            normalize_settings_dict(
                {"temporal_mode": 1, "skip_similar_image": True, "smoothing": True},
                default_info_for_box_name("Home Score"),
            ),
        )
        target.ocrResultPerCharacterSmoother.get_smoothed_result = Mock(
            side_effect=AssertionError("must use whole readings")
        )
        frame = np.zeros((40, 40, 3), np.uint8)
        with patch("tesseract.iterate_level", return_value=[]):
            self.assertEqual(
                detector.detect_multi_text(
                    None, None, [target], color=frame, observation_time=0
                )[0].state,
                state.FailedFilter,
            )
            self.assertEqual(
                detector.detect_multi_text(
                    None, None, [target], color=frame, observation_time=0.2
                )[0].state,
                state.Success,
            )
        self.assertEqual(detector.api.GetUTF8Text.call_count, 2)

    def test_leading_zero_removal_never_changes_blank_into_zero(self):
        detector = TextDetector.__new__(TextDetector)
        detector.api_lock = threading.Lock()
        detector.api = Mock()
        detector.api.GetUTF8Text.return_value = ""
        detector.api.MeanTextConf.return_value = 100
        detector.ocr_model_index = 1
        target = TextDetectionTarget(
            0,
            0,
            40,
            40,
            "Home Score",
            normalize_settings_dict(
                {"remove_leading_zeros": True}, default_info_for_box_name("Home Score")
            ),
        )
        result = detector.detect_multi_text(
            None, None, [target], color=np.zeros((40, 40, 3), np.uint8)
        )[0]
        self.assertEqual(
            (result.text, result.state),
            ("", TextDetectionTargetWithResult.ResultState.Empty),
        )

    def test_file_time_uses_media_position_with_frame_rate_fallback(self):
        from camera_thread import TimerThread
        from types import SimpleNamespace

        for properties, expected in (
            ({cv2.CAP_PROP_POS_MSEC: 12500}, 12.5),
            (
                {
                    cv2.CAP_PROP_POS_MSEC: 0,
                    cv2.CAP_PROP_POS_FRAMES: 300,
                    cv2.CAP_PROP_FPS: 30,
                },
                10,
            ),
            (
                {
                    cv2.CAP_PROP_POS_MSEC: float("nan"),
                    cv2.CAP_PROP_POS_FRAMES: 300,
                    cv2.CAP_PROP_FPS: 0,
                },
                None,
            ),
        ):
            with self.subTest(properties=properties):
                capture = SimpleNamespace(get=lambda prop: properties.get(prop, 0))
                self.assertEqual(
                    TimerThread.observation_time_for_file(
                        SimpleNamespace(video_capture=capture)
                    ),
                    expected,
                )

    def test_pending_readings_are_rejected_and_history_survives_local_copies(self):
        state = TextDetectionTargetWithResult.ResultState
        detector = TextDetector.__new__(TextDetector)
        detector.ocr_model_index = 6
        detector._detect_multi_text = Mock()
        stored = TextDetectionTarget(0, 0, 40, 40, "Home Score", {"temporal_mode": 1})
        observed = []
        for tick, value in enumerate(("7", "7", "1", "7", "9", "9")):
            local = TextDetectionTarget(0, 0, 40, 40, stored.name, stored.settings)
            local.temporal_filter = stored.temporal_filter
            detector._detect_multi_text.return_value = [
                TextDetectionResult(value, state.Success)
            ]
            result = detector.detect_multi_text(
                None, None, [local], observation_time=tick / 5
            )[0]
            observed.append((result.text, result.state))
        self.assertEqual(
            observed,
            [
                ("7", state.FailedFilter),
                ("7", state.Success),
                ("1", state.FailedFilter),
                ("7", state.Success),
                ("9", state.FailedFilter),
                ("9", state.Success),
            ],
        )
        detector.reset_temporal()
        detector._detect_multi_text.return_value = [
            TextDetectionResult("9", state.Success)
        ]
        self.assertEqual(
            detector.detect_multi_text(None, None, [local], observation_time=2)[
                0
            ].state,
            state.FailedFilter,
        )

    def test_mode_is_opt_in_and_saved_with_box_settings(self):
        info = default_info_for_box_name("Home Score")
        self.assertEqual(normalize_settings_dict({}, info)["temporal_mode"], 0)
        self.assertEqual(
            normalize_settings_dict({"temporal_mode": 99}, info)["temporal_mode"], 0
        )
        target = TextDetectionTarget(
            0,
            0,
            40,
            40,
            "Home Score",
            normalize_settings_dict({"temporal_mode": 2}, info),
        )
        storage = TextDetectionTargetMemoryStorage()
        storage.clear()
        try:
            storage.add_item(target)
            self.assertEqual(
                storage.getBoxesForStorage()[0]["settings"]["temporal_mode"], 2
            )
        finally:
            storage.clear()


if __name__ == "__main__":
    unittest.main()
