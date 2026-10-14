"""ScoreSight OpenCV reader for calibrated LED digits and clock punctuation.

Developed against the local four-image scoreboard dataset. No external code or
weights. BGR uint8 input; returns (text, mask, segment details). Empty string means
unlit, None means uncertain. Fixed color/geometry thresholds require calibration.
"""

import cv2
import numpy as np

from defaults import SEGMENT_LAYOUTS

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


def decode_digit(cell, occupancy=0.09):
    """Read one complete digit cell; tiny or unknown patterns are rejected."""
    h, w = cell.shape
    if h < 12 or w < 8:
        return None, {"value": None}
    densities = [
        float(np.mean(cell[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] > 0))
        for x0, y0, x1, y1 in ZONES
    ]
    pattern = "".join("1" if p >= occupancy else "0" for p in densities)
    value = "" if np.count_nonzero(cell) < 3 else PATTERNS.get(pattern)
    return value, {"pattern": pattern, "densities": densities, "value": value}


def threshold_led_colors(crop, brightness=140, saturation=90, led_color=0):
    """Mask warm (0), green (1), or all saturated (2) bright LEDs."""
    if (
        not isinstance(crop, np.ndarray)
        or crop.ndim != 3
        or crop.shape[2] != 3
        or crop.dtype != np.uint8
        or crop.shape[0] < 1
        or crop.shape[1] < 1
    ):
        return None
    if led_color not in (0, 1, 2):
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    if led_color == 0:
        color_mask = (hue <= 35) | (hue >= 165)
    elif led_color == 1:
        color_mask = (hue > 35) & (hue <= 90)
    else:
        color_mask = np.ones(hue.shape, dtype=bool)
    return (
        color_mask & (hsv[:, :, 1] >= saturation) & (hsv[:, :, 2] >= brightness)
    ).astype(np.uint8) * 255


def read_score(crop, brightness=140, occupancy=0.09, digit_count=2, led_color=0):
    if (
        not isinstance(crop, np.ndarray)
        or crop.ndim != 3
        or crop.shape[2] != 3
        or crop.shape[0] < 12
        or crop.shape[1] < 16
        or crop.dtype != np.uint8
        or type(digit_count) is not int
        or digit_count not in (1, 2, 3)
        or crop.shape[1] < 8 * digit_count
    ):
        return None, None, []
    mask = threshold_led_colors(crop, brightness=brightness, led_color=led_color)
    if mask is None:
        return None, None, []
    digits, details = [], []
    for cell in np.array_split(mask, digit_count, axis=1):
        value, detail = decode_digit(cell, occupancy)
        digits.append(value)
        details.append(detail)
    # Leading unlit cells are allowed. Once digits begin, every cell must be lit.
    result = "".join(value or "" for value in digits)
    seen_digit = False
    for value in digits:
        if value is None or (seen_digit and value == ""):
            result = None
            break
        seen_digit = seen_digit or bool(value)
    return result, mask, details


def read_separator(cell, symbol):
    """Require observed, separate colon dots or one low decimal dot."""
    if np.count_nonzero(cell) < 3:
        return ""
    count, _, stats, centers = cv2.connectedComponentsWithStats(cell)
    blobs = [
        centers[index]
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 3
    ]
    height, width = cell.shape
    if symbol == ".":
        return "." if len(blobs) == 1 and blobs[0][1] >= 0.65 * height else None
    if len(blobs) != 2:
        return None
    upper, lower = sorted(blobs, key=lambda center: center[1])
    return (
        ":"
        if upper[1] < 0.5 * height <= lower[1]
        and lower[1] - upper[1] >= 0.15 * height
        and abs(upper[0] - lower[0]) <= 0.3 * width
        else None
    )


def read_display(
    crop, digit_count=2, led_color=0, brightness=140, layout="number", cells=None
):
    """Read explicit local XYWH cells, including observed clock separators.

    Cells are ordered left-to-right, must not overlap, and use field-relative
    pixel coordinates. Clock separators should use the full digit height.
    Without explicit cells, only the numeric equal-width layout is supported.
    """
    if layout not in SEGMENT_LAYOUTS:
        return None, None, []
    if layout == "number" and (
        type(digit_count) is not int or digit_count not in (1, 2, 3)
    ):
        return None, None, []
    if layout == "number" and cells is None:
        return read_score(
            crop, digit_count=digit_count, led_color=led_color, brightness=brightness
        )
    mask = threshold_led_colors(crop, brightness=brightness, led_color=led_color)
    if mask is None or not cells:
        return None, mask, []
    symbols = "d" * digit_count if layout == "number" else layout
    if len(cells) != len(symbols):
        return None, mask, []
    try:
        bounds = sorted(cells, key=lambda cell: cell[0])
        if any(len(cell) != 4 or not np.all(np.isfinite(cell)) for cell in bounds):
            return None, mask, []
        bounds = [tuple(int(value) for value in cell) for cell in bounds]
    except (IndexError, TypeError, ValueError, OverflowError):
        return None, mask, []
    height, width = mask.shape
    last_right = 0
    values, details = [], []
    for symbol, (x, y, w, h) in zip(symbols, bounds):
        if (
            x < last_right
            or y < 0
            or w < 1
            or h < 12
            or x + w > width
            or y + h > height
        ):
            return None, mask, details
        last_right = x + w
        cell = mask[y : y + h, x : x + w]
        if symbol == "d":
            value, detail = decode_digit(cell, 0.09)
        else:
            value = read_separator(cell, symbol)
            detail = {"value": value, "separator": symbol}
        values.append(value)
        details.append(dict(detail, bounds=[x, y, w, h]))
    if None in values:
        return None, mask, details
    if not any(values):
        return "", mask, details
    # Only leading digit cells may be unlit. Missing punctuation remains unknown.
    started = False
    for symbol, value in zip(symbols, values):
        if value:
            started = True
        elif started or symbol != "d":
            return None, mask, details
    text = "".join(values)
    if layout != "number":
        parts = text.split(":" if ":" in layout else ".")
        if not parts[0] or (":" in layout and int(parts[1]) >= 60):
            return None, mask, details
    return text, mask, details
