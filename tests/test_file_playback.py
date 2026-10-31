"""Playback follows source time even when OCR takes longer than a video frame."""

import unittest

import cv2

from file_playback import RealtimeFilePlayback


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class Capture:
    def __init__(self, fps=30, timestamps=True, length=10000):
        self.fps, self.timestamps, self.length = fps, timestamps, length
        self.index = -1
        self.decoded = []

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC:
            return max(0, self.index) / self.fps * 1000 if self.timestamps else 0
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.index + 1
        raise AssertionError(prop)

    def grab(self):
        self.index += 1
        return self.index < self.length

    def retrieve(self):
        self.decoded.append(self.index)
        return True, self.index

    def read(self):
        return self.retrieve() if self.grab() else (False, None)


class FilePlaybackTests(unittest.TestCase):
    def setUp(self):
        self.time = FakeTime()
        self.playback = RealtimeFilePlayback(self.time, self.time.sleep)
        self.capture = Capture()

    def test_source_fps_paces_fast_processing(self):
        for _ in range(31):
            self.playback.read(self.capture)
        self.assertAlmostEqual(self.time.now, 1)

    def test_slow_ocr_does_not_accumulate_playback_delay(self):
        self.playback.read(self.capture)
        for _ in range(20):
            self.time.now += 0.18  # expensive OCR instead of a 33 ms frame
            _, frame = self.playback.read(self.capture)
            self.assertLessEqual(abs(frame / 30 - self.time.now), 1 / 30 + 1e-8)
        self.assertGreater(self.capture.index, 100)
        self.assertLess(len(self.capture.decoded), self.capture.index)

    def test_pause_resume_does_not_skip_the_paused_interval(self):
        self.playback.read(self.capture)
        self.time.now += 60
        self.playback.reset()
        _, frame = self.playback.read(self.capture)
        self.assertEqual(frame, 1)

    def test_playback_speed_uses_elapsed_time(self):
        for speed in (2, 8):
            with self.subTest(speed=speed):
                self.playback.reset()
                self.capture = Capture()
                self.playback.read(self.capture, speed)
                self.time.now += 1
                _, frame = self.playback.read(self.capture, speed)
                self.assertAlmostEqual(frame / 30, speed, delta=1 / 30 + 1e-8)

    def test_missing_timestamps_fall_back_to_frame_position(self):
        self.capture.timestamps = False
        for _ in range(31):
            self.playback.read(self.capture)
        self.assertAlmostEqual(self.time.now, 1)

    def test_end_of_file_during_catchup_returns_failure(self):
        self.capture.length = 2
        self.playback.read(self.capture)
        self.time.now += 1
        self.assertEqual(self.playback.read(self.capture), (False, None))
