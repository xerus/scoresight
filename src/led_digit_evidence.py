"""Conservative bright-core digit evidence for LED OCR ambiguity checks.

The fixed policy was developed on the committed camera fixtures. It uses white
LED cores rather than colored glow, rejects joined/wide shapes, and is not an
independent accuracy guarantee. Unknown evidence must not be treated as agreement.
"""

import math

import cv2
import numpy as np

from seven_segment import decode_digit


def split_leading_stroke(mask, box):
    """Separate a tall thin leading stroke across a weak connecting bridge.

    Ring digits have substantial top/bottom connections across their interior;
    those do not pass the projection limit. Both resulting parts must still be
    decoded independently by the caller.
    """
    x, y, w, h = box
    glyph = mask[y : y + h, x : x + w]
    if w / h < 0.7:
        return None
    candidates = []
    for cut in range(8, min(w - 8, int(h * 0.32)) + 1):
        bridge = np.count_nonzero(glyph[:, cut - 1 : cut + 1]) / (2 * h)
        if bridge > 0.12:
            continue
        left = glyph[:, :cut]
        _, _, pieces, _ = cv2.connectedComponentsWithStats(left)
        tall = [p for p in pieces[1:] if p[cv2.CC_STAT_HEIGHT] >= 0.8 * h]
        if len(tall) != 1:
            continue
        lx, ly, lw, lh, _ = map(int, tall[0])
        if not 0.08 <= lw / lh < 0.32:
            continue
        rx, ry, rw, rh = cv2.boundingRect(glyph[:, cut:])
        if rh < 0.8 * h or rw < 8:
            continue
        candidates.append(
            (bridge, (x + lx, y + ly, lw, lh), (x + cut + rx, y + ry, rw, rh))
        )
    if not candidates:
        return None
    _, left, right = min(candidates, key=lambda item: item[0])
    return [left, right]


def digit_evidence(crop):
    """Return a complete numeric shape reading and tight crop bounds, or unknown."""
    if (
        not isinstance(crop, np.ndarray)
        or crop.dtype != np.uint8
        or crop.ndim != 3
        or crop.shape[2] != 3
        or min(crop.shape[:2]) < 8
    ):
        return None, None, {"reason": "invalid_image"}
    height, width = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    colored = (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 140)
    green = colored & (hsv[:, :, 0] > 35) & (hsv[:, :, 0] <= 90)
    green_display = np.count_nonzero(green) > 0.7 * max(1, np.count_nonzero(colored))
    core_min = 240 if green_display else 160
    mask = (crop.min(axis=2) >= core_min).astype(np.uint8) * 255
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    boxes = sorted(
        (
            tuple(int(value) for value in row[:4])
            for row in components[1:]
            if row[cv2.CC_STAT_HEIGHT] >= max(12, height * 0.4)
        ),
        key=lambda box: box[0],
    )
    if not boxes or len(boxes) > 6:
        return None, None, {"reason": "no_clear_digits"}
    if any(x == 0 or x + w == width for x, y, w, h in boxes):
        return None, None, {"reason": "clipped_digits", "core_min": core_min}
    separated = []
    for box in boxes:
        separated.extend(split_leading_stroke(mask, box) or [box])
    boxes = separated
    left = max(0, min(x for x, _, _, _ in boxes) - 3)
    top = max(0, min(y for _, y, _, _ in boxes) - 3)
    right = min(width, max(x + w for x, _, w, _ in boxes) + 3)
    bottom = min(height, max(y + h for _, y, _, h in boxes) + 3)
    bounds = (left, top, right, bottom)
    digits, details, reasons = [], [], []
    last_right = 0
    previous_width = 0
    for x, y, w, h in boxes:
        aspect = w / h
        overlap_tolerance = max(1, min(previous_width, w) * 0.1)
        reason = None
        value, detail = None, {}
        if x < last_right - overlap_tolerance or not 0.08 <= aspect <= 0.85:
            reason = "joined_or_wide_digits"
        elif aspect < 0.32:
            value, detail = "1", {"pattern": "thin_stroke"}
        else:
            value, detail = decode_digit(mask[y : y + h, x : x + w], occupancy=0.25)
            if not value:
                reason = "unknown_digit"
        if value:
            glyph = mask[y : y + h, x : x + w]
            contours, hierarchy = cv2.findContours(
                glyph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            holes = (
                0
                if hierarchy is None
                else int(
                    sum(
                        parent >= 0 and cv2.contourArea(contour) >= max(3, w * h * 0.01)
                        for contour, (_, _, _, parent) in zip(contours, hierarchy[0])
                    )
                )
            )
            expected_holes = {"0": 1, "6": 1, "8": 2, "9": 1}.get(value, 0)
            if holes != expected_holes:
                value, reason = None, "unclear_topology"
            detail["holes"] = holes
        last_right = x + w
        previous_width = w
        digits.append(value)
        if reason:
            reasons.append(reason)
        details.append(dict(detail, value=value, bounds=[x, y, w, h]))
    prefix = []
    for digit in digits:
        if digit is None:
            break
        prefix.append(digit)
    suffix = []
    for digit in reversed(digits):
        if digit is None:
            break
        suffix.append(digit)
    return (
        "".join(digits) if all(digits) else None,
        bounds,
        {
            "reason": reasons[0] if reasons else "clear_digits",
            "boxes": boxes,
            "digits": details,
            "core_min": core_min,
            "min_digits": sum(max(1, math.ceil(w / h / 0.85)) for x, y, w, h in boxes),
            "known_prefix": "".join(prefix),
            "known_suffix": "".join(reversed(suffix)),
        },
    )
