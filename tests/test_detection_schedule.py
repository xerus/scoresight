"""OCR deadlines belong to the input timeline, not processing completion."""

import unittest

from detection_schedule import DetectionSchedule


class DetectionScheduleTests(unittest.TestCase):
    def test_late_frames_do_not_shift_the_fixed_grid(self):
        schedule = DetectionSchedule()
        for timestamp in (0, 0.21, 0.4, 0.61, 0.8):
            self.assertTrue(schedule.due(timestamp, 0.2), timestamp)
        self.assertAlmostEqual(schedule.next_at, 1.0)

    def test_multiple_frames_in_one_slot_only_trigger_once(self):
        schedule = DetectionSchedule()
        self.assertTrue(schedule.due(0, 0.2))
        for timestamp in (0, 0.03, 0.1, 0.199):
            self.assertFalse(schedule.due(timestamp, 0.2))
        self.assertTrue(schedule.due(0.2, 0.2))

    def test_skipped_slots_are_not_queued(self):
        schedule = DetectionSchedule()
        schedule.due(0, 0.2)
        self.assertTrue(schedule.due(2.03, 0.2))
        self.assertFalse(schedule.due(2.03, 0.2))
        self.assertFalse(schedule.due(2.1, 0.2))
        self.assertTrue(schedule.due(2.2, 0.2))

    def test_backward_seek_and_reset_start_a_new_schedule(self):
        schedule = DetectionSchedule()
        self.assertTrue(schedule.due(75.23, 0.2))
        self.assertTrue(schedule.due(0, 0.2))
        schedule.reset()
        self.assertTrue(schedule.due(0, 0.2))

    def test_cadence_change_uses_new_grid(self):
        schedule = DetectionSchedule()
        schedule.due(0.21, 0.2)
        self.assertTrue(schedule.due(0.22, 0.1))
        self.assertFalse(schedule.due(0.29, 0.1))
        self.assertTrue(schedule.due(0.3, 0.1))

    def test_invalid_timestamp_or_interval_does_not_poison_history(self):
        schedule = DetectionSchedule()
        for timestamp, interval in ((float("nan"), 0.2), (0, 0), (0, float("inf"))):
            self.assertFalse(schedule.due(timestamp, interval))
        self.assertTrue(schedule.due(0, 0.2))
