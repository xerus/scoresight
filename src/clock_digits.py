"""Format an explicitly configured compact clock without inventing digits."""

import re


FORMATS = {
    1: (3, 1, ":", r"[0-9]:[0-5][0-9]"),
    2: (4, 2, ":", r"[0-9]{2}:[0-5][0-9]"),
    3: (3, 2, ".", r"[0-5][0-9]\.[0-9]"),
    4: (2, 1, ".", r"[0-9]\.[0-9]"),
}


def format_clock_digits(text, mode=0):
    """Return observed text, declared-layout time, or None for invalid input.

    0 preserves observed punctuation; 1=MSS, 2=MMSS, 3=SSt, 4=St, 5=seconds. Compact
    modes require every configured digit and valid seconds. Formatting is an
    explicit display-layout mapping, not an inferred tick or guessed separator.
    """
    if mode == 0 or text == "":
        return text
    if mode == 5:
        return (
            text
            if isinstance(text, str) and re.fullmatch(r"[0-5]?[0-9]", text)
            else None
        )
    specification = FORMATS.get(mode)
    if specification is None or not isinstance(text, str):
        return None
    length, split, separator, pattern = specification
    if re.fullmatch(pattern, text):
        return text
    if re.fullmatch(rf"[0-9]{{{length}}}", text) is None:
        return None
    formatted = text[:split] + separator + text[split:]
    return formatted if re.fullmatch(pattern, formatted) else None
