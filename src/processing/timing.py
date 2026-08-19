"""
Timing extraction for decoded Morse.

`MorseDetector` answers "what did the operator send". This module answers
"when". It walks the same thresholded envelope and reports, in seconds:

  * `element_segments()`  - every dit and dah, with a start and an end
  * `character_events()`  - every decoded character, with the Morse pattern
                            that produced it

Both were previously written inline inside the GUIs, once per application.
They live here so the file dashboard, the live stream decoder and the tests
all measure timing the same way.

All times are in seconds. `offset_seconds` shifts them onto an absolute
timeline, which is what the streaming decoder needs: its buffer holds only
the last few seconds of audio, but characters must be reported against the
time since the stream started.
"""

from dataclasses import dataclass

import numpy as np

from src.processing.morse_table import INVERSE_MORSE_TABLE


@dataclass
class ElementSegment:
    """A single dit or dah, located in time."""
    start_seconds: float
    end_seconds: float
    kind: str  # "dit" or "dah"

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class CharacterEvent:
    """One decoded character, located in time."""
    start_seconds: float
    end_seconds: float
    character: str
    morse: str  # "" for a word space


def _binary_envelope(envelope: np.ndarray, detector) -> np.ndarray:
    """Threshold the envelope exactly the way MorseDetector does."""
    threshold = detector.threshold_ratio * np.max(envelope)
    return (envelope > threshold).astype(np.int8)


def element_segments(envelope: np.ndarray, detector, sample_rate: int,
                     offset_seconds: float = 0.0) -> list:
    """
    Return every on-period in the envelope, classified as a dit or a dah.

    This is the raw detector output: elements that never resolve into a
    character are still reported, which is what the Morse timeline draws.
    """
    dit = getattr(detector, "dit_samples", None)
    if not dit or len(envelope) == 0 or np.max(envelope) <= 0:
        return []

    binary = _binary_envelope(envelope, detector)
    dah_threshold = getattr(detector, "dah_threshold", 2.0)

    segments = []
    total = len(binary)
    index = 0
    while index < total:
        if binary[index] == 1:
            start = index
            while index < total and binary[index] == 1:
                index += 1
            kind = "dit" if (index - start) < dah_threshold * dit else "dah"
            segments.append(ElementSegment(
                start_seconds=offset_seconds + start / sample_rate,
                end_seconds=offset_seconds + index / sample_rate,
                kind=kind))
        else:
            index += 1
    return segments


def character_events(envelope: np.ndarray, detector, sample_rate: int,
                     offset_seconds: float = 0.0) -> list:
    """
    Return every decoded character with the time its last element ended.

    A word space is reported as a `CharacterEvent` whose character is " "
    and whose morse is "". Unrecognised patterns decode to "?", matching
    `MorseDecoder`.
    """
    dit = getattr(detector, "dit_samples", None)
    if not dit or len(envelope) == 0 or np.max(envelope) <= 0:
        return []

    binary = _binary_envelope(envelope, detector)
    dah_threshold = getattr(detector, "dah_threshold", 2.0)
    letter_gap = getattr(detector, "letter_gap_threshold", 2.0) * dit
    word_gap = getattr(detector, "word_gap_threshold", 5.0) * dit

    events = []
    pattern = []
    pattern_start = None
    total = len(binary)

    index = 0
    while index < total and binary[index] == 0:
        index += 1

    def flush(end_sample):
        """Turn the collected dits and dahs into one character event."""
        nonlocal pattern_start
        if not pattern:
            return
        text = "".join(pattern)
        events.append(CharacterEvent(
            start_seconds=offset_seconds + pattern_start / sample_rate,
            end_seconds=offset_seconds + end_sample / sample_rate,
            character=INVERSE_MORSE_TABLE.get(text, "?"),
            morse=text))
        pattern.clear()
        pattern_start = None

    while index < total:
        start = index
        while index < total and binary[index] == 1:
            index += 1
        if pattern_start is None:
            pattern_start = start
        on_length = index - start
        pattern.append("." if on_length < dah_threshold * dit else "-")
        letter_end = index

        start = index
        while index < total and binary[index] == 0:
            index += 1
        gap = index - start

        if index >= total:
            # Trailing silence: the character is complete.
            flush(letter_end)
            break
        if gap >= word_gap:
            flush(letter_end)
            # The space is timestamped at the END of the gap, where the next
            # character begins. Event times therefore increase strictly, which
            # is what the streaming decoder relies on to know what it has
            # already emitted.
            events.append(CharacterEvent(
                start_seconds=offset_seconds + letter_end / sample_rate,
                end_seconds=offset_seconds + index / sample_rate,
                character=" ", morse=""))
        elif gap >= letter_gap:
            flush(letter_end)

    return events
