"""Readers for official timing documents (wige / ADAC, ITS Chrono / FFSA, Al Kamel / FIA WEC)."""

import math


def parse_laptime(value: object) -> float:
    """Convert ``"1:02:03.456"``, ``"8:32.481"`` or ``"34.112"`` to seconds (NaN if not a time)."""
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text == "-":
        return math.nan
    seconds = 0.0
    try:
        for part in text.split(":"):
            seconds = seconds * 60 + float(part)
    except ValueError:
        return math.nan
    return seconds
