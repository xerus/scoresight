"""Pace file capture against source timestamps without accumulating OCR delay."""

import math
import time

import cv2


class RealtimeFilePlayback:
    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock = clock
        self.sleep = sleep
        self.reset()

    def reset(self):
        self.started_at = None
        self.started_position = None

    @staticmethod
    def position(capture, fps):
        seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        frames = capture.get(cv2.CAP_PROP_POS_FRAMES)
        if math.isfinite(seconds) and (seconds > 0 or (seconds == 0 and frames <= 1)):
            return seconds
        return max(0, frames - 1) / fps

    def read(self, capture, speed=1):
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30
        ok, frame = capture.read()
        if not ok:
            return ok, frame
        position = self.position(capture, fps)
        now = self.clock()
        if self.started_at is None:
            self.started_at = now
            self.started_position = position
        target = self.started_position + (now - self.started_at) * speed
        # Discard overdue frames without converting each one to BGR. A fixed
        # target bounds this catch-up pass even on a slow decoder.
        while position + 1 / fps < target:
            if not capture.grab():
                return False, None
            position = self.position(capture, fps)
            ok, frame = None, None
        if ok is None:
            ok, frame = capture.retrieve()
            if not ok:
                return ok, frame
        deadline = self.started_at + (position - self.started_position) / speed
        delay = deadline - self.clock()
        if delay > 0:
            self.sleep(delay)
        return True, frame
