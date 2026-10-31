"""Sample fixed intervals of input time without queuing missed detections."""

import math


class DetectionSchedule:
    def __init__(self):
        self.reset()

    def reset(self):
        self.next_at = None
        self.last_timestamp = None
        self.interval = None

    def due(self, timestamp, interval):
        if not math.isfinite(timestamp) or not math.isfinite(interval) or interval <= 0:
            return False
        if (
            self.interval != interval
            or self.last_timestamp is not None
            and timestamp < self.last_timestamp
        ):
            self.reset()
        self.interval = interval
        self.last_timestamp = timestamp
        if self.next_at is not None and timestamp + 1e-9 < self.next_at:
            return False
        # Advance directly to the next grid boundary, even after a long stall.
        # Never issue a burst of jobs for deadlines that have already passed.
        self.next_at = (math.floor((timestamp + 1e-9) / interval) + 1) * interval
        return True
