"""Confirm observed whole values using soft score/clock continuity checks."""

import math
import re


class TemporalValueFilter:
    OFF = 0
    NUMBER = 1
    COUNTDOWN = 2
    COUNTUP = 3

    def __init__(self):
        self.clear()

    def clear(self):
        self.context = None
        self.accepted = None
        self.accepted_at = None
        self.pending = None
        self.pending_at = None
        self.pending_count = 0
        self.last_at = None
        self.interrupted = False

    @staticmethod
    def parse(text, mode):
        if text == "":
            return ("blank", None, 1.0)
        if mode == TemporalValueFilter.NUMBER:
            match = re.fullmatch(r"([0-9]+)(?:st|nd|rd|th)?", text)
            return ("number", int(match[1]), 1.0) if match else None
        match = re.fullmatch(r"(?:(\d+):)?(\d+)(?:\.(\d+))?", text)
        if not match or (match[1] is not None and int(match[2]) >= 60):
            return None
        fraction = match[3]
        value = int(match[2]) + (int(match[1]) * 60 if match[1] else 0)
        if fraction:
            value += int(fraction) / (10 ** len(fraction))
        return ("clock", value, 10 ** -len(fraction) if fraction else 1.0)

    @staticmethod
    def clock_step(previous, current, elapsed, mode):
        if previous[0] != "clock" or current[0] != "clock":
            return False
        advance = previous[1] - current[1]
        if mode == TemporalValueFilter.COUNTUP:
            advance = -advance
        # Displays quantize time, and clocks may stop. A reversal or reset is
        # allowed through confirmation rather than forbidden by this check.
        return -1e-8 <= advance <= elapsed + max(previous[2], current[2]) + 0.25

    def observe(self, text, timestamp, mode, context):
        """Return acceptance and diagnostics; None means rejected OCR input.

        Pending readings never replace the accepted value. Two matching number
        readings confirm normal +1/+2/+3 changes; other changes need three.
        Clock candidates may progress between observations instead of needing
        identical strings. No extrapolated or recombined value is returned.
        """
        if not math.isfinite(timestamp):
            self.clear()
            return False, {"reason": "invalid_timestamp"}
        if context != self.context or (
            self.last_at is not None and timestamp < self.last_at
        ):
            self.clear()
            self.context = context
        if self.last_at is not None and timestamp == self.last_at:
            return False, {"reason": "duplicate_frame"}
        self.last_at = timestamp
        current = self.parse(text, mode) if text is not None else None
        if current is None:
            self.pending = None
            self.pending_count = 0
            self.interrupted = True
            return False, {"reason": "invalid_reading"}

        clock_mode = mode in (self.COUNTDOWN, self.COUNTUP)
        same = self.accepted is not None and current[:2] == self.accepted[:2]
        plausible = (
            clock_mode
            and self.accepted is not None
            and self.clock_step(
                self.accepted, current, timestamp - self.accepted_at, mode
            )
        )
        if same and (not clock_mode or not self.interrupted):
            required = 1
        elif plausible and not self.interrupted:
            required = 1
        elif self.accepted is None or current[0] == "blank":
            required = 2
        elif clock_mode:
            required = 2 if self.interrupted and plausible else 3
        elif (
            current[0] == "number"
            and self.accepted[0] == "number"
            and 1 <= current[1] - self.accepted[1] <= 3
        ):
            required = 2
        else:
            required = 3

        matching = self.pending is not None and current[:2] == self.pending[:2]
        if clock_mode and self.pending is not None:
            matching = matching or self.clock_step(
                self.pending, current, timestamp - self.pending_at, mode
            )
        self.pending_count = self.pending_count + 1 if matching else 1
        self.pending, self.pending_at = current, timestamp
        if self.pending_count < required:
            return False, {
                "reason": "confirming_change",
                "observations": self.pending_count,
                "required": required,
            }
        self.accepted, self.accepted_at = current, timestamp
        self.pending = None
        self.pending_count = 0
        self.interrupted = False
        return True, {"reason": "confirmed", "required": required}
