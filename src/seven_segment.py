"""Original ScoreSight OpenCV reader for two calibrated warm-LED digit cells.

Developed against the local four-image scoreboard dataset. No external code or
weights. BGR uint8 input; returns (text, mask, segment details). Empty string means
unlit, None means uncertain. Fixed color/geometry thresholds require calibration.
"""

import cv2
import numpy as np

# Seven rectangular occupancy regions: top, upper-right, lower-right,
# bottom, lower-left, upper-left, middle. Coordinates relative to each slot.
ZONES = [
    (0.35, 0.04, 0.65, 0.18),
    (0.68, 0.16, 0.98, 0.43),
    (0.60, 0.56, 0.94, 0.86),
    (0.35, 0.81, 0.65, 0.97),
    (0.01, 0.56, 0.34, 0.86),
    (0.07, 0.16, 0.40, 0.43),
    (0.35, 0.40, 0.65, 0.60),
]
PATTERNS = {
    "1111110": "0",
    "0110000": "1",
    "1101101": "2",
    "1111001": "3",
    "0110011": "4",
    "1011011": "5",
    "1011111": "6",
    "1110000": "7",
    "1111111": "8",
    "1111011": "9",
}


def threshold_led_colors(crop, brightness=140, saturation=90):
    """Return a binary mask for bright, saturated red/yellow display LEDs."""
    if (
        not isinstance(crop, np.ndarray)
        or crop.ndim != 3
        or crop.shape[2] != 3
        or crop.dtype != np.uint8
        or crop.shape[0] < 1
        or crop.shape[1] < 1
    ):
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return (
        ((hsv[:, :, 0] <= 35) | (hsv[:, :, 0] >= 165))
        & (hsv[:, :, 1] >= saturation)
        & (hsv[:, :, 2] >= brightness)
    ).astype(np.uint8) * 255


def read_score(crop, brightness=140, occupancy=0.09):
    if (
        not isinstance(crop, np.ndarray)
        or crop.ndim != 3
        or crop.shape[2] != 3
        or crop.shape[0] < 12
        or crop.shape[1] < 16
        or crop.dtype != np.uint8
    ):
        return None, None, []
    mask = threshold_led_colors(crop, brightness=brightness)
    digits, details = [], []
    for cell in np.array_split(mask, 2, axis=1):
        h, w = cell.shape
        densities = [
            float(
                np.mean(cell[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] > 0)
            )
            for x0, y0, x1, y1 in ZONES
        ]
        pattern = "".join("1" if p >= occupancy else "0" for p in densities)
        # Require no bright evidence for blank; isolated spots become rejection.
        value = "" if np.count_nonzero(cell) < 3 else PATTERNS.get(pattern)
        digits.append(value)
        details.append({"pattern": pattern, "densities": densities, "value": value})
    # A lit tens digit followed by a blank units cell is ambiguous, not a number.
    result = (
        None if None in digits or (digits[0] and not digits[1]) else "".join(digits)
    )
    return result, mask, details
