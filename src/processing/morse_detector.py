"""
Morse element detector.

Takes a smoothed envelope (from SignalProcessor) and converts it to a
Morse code string of dits ('.'), dahs ('-'), and gaps (' ', '   ').
"""

import numpy as np


class MorseDetector:
    """Detect Morse elements from a smoothed envelope signal."""

    def __init__(
        self,
        sample_rate: int = 44100,
        threshold_ratio: float = 0.3,
        dah_threshold: float = 2.0,
        letter_gap_threshold: float = 2.0,
        word_gap_threshold: float = 5.0,
    ):
        """
        Parameters
        ----------
        sample_rate : int
            Audio sample rate in Hz.
        threshold_ratio : float
            Fraction of envelope peak above which the signal is considered "on".
        dah_threshold : float
            On-period multiple of dit length above which it's classified as a dah.
        letter_gap_threshold : float
            Gap-period multiple of dit length above which it's a letter boundary.
        word_gap_threshold : float
            Gap-period multiple of dit length above which it's a word boundary.
        """
        self.sample_rate = sample_rate
        self.threshold_ratio = threshold_ratio
        self.dah_threshold = dah_threshold
        self.letter_gap_threshold = letter_gap_threshold
        self.word_gap_threshold = word_gap_threshold
        self.dit_samples = None
        self.estimated_wpm = None

    def detect(self, envelope: np.ndarray) -> str:
        """
        Convert smoothed envelope into Morse string.

        Letter boundary = single space. Word boundary = three spaces.

        Example output: ".... . .-.. .-.. ---   .-- --- .-. .-.. -.."

        Returns empty string if signal is silent.
        """
        if len(envelope) == 0 or np.max(envelope) < 1e-6:
            return ""

        # 1. Threshold to binary on/off
        threshold = self.threshold_ratio * np.max(envelope)
        binary = (envelope > threshold).astype(np.int8)

        # 2. Extract run lengths
        on_durations, gap_durations = self._extract_runs(binary)
        if len(on_durations) == 0:
            return ""

        # 3. Estimate dit length and WPM
        self.dit_samples = self._estimate_dit_length(on_durations)
        self.estimated_wpm = self._dit_samples_to_wpm(self.dit_samples)

        # 4. Classify each element and gap
        result = []
        for i, on_dur in enumerate(on_durations):
            if on_dur < self.dah_threshold * self.dit_samples:
                result.append(".")
            else:
                result.append("-")

            if i < len(gap_durations):
                gap = gap_durations[i]
                if gap < self.letter_gap_threshold * self.dit_samples:
                    pass  # intra-letter gap, no separator
                elif gap < self.word_gap_threshold * self.dit_samples:
                    result.append(" ")
                else:
                    result.append("   ")

        return "".join(result)

    def _extract_runs(self, binary: np.ndarray) -> tuple:
        """Extract durations of consecutive 1s (on) and 0s between them (gap)."""
        on_durations = []
        gap_durations = []
        n = len(binary)
        i = 0

        # Skip leading zeros
        while i < n and binary[i] == 0:
            i += 1

        while i < n:
            # Measure an "on" run
            start = i
            while i < n and binary[i] == 1:
                i += 1
            on_durations.append(i - start)

            # Measure the gap that follows (only if there is another "on" after)
            start = i
            while i < n and binary[i] == 0:
                i += 1
            if i < n:
                gap_durations.append(i - start)

        return on_durations, gap_durations

    def _estimate_dit_length(self, on_durations: list) -> int:
        """
        Estimate dit length by sorting on-durations and finding the largest jump.

        Dits cluster around one length, dahs around 3x that length.
        The biggest gap in sorted values separates them.
        """
        durations = np.array(on_durations)
        if len(durations) == 1:
            return int(durations[0])

        sorted_durs = np.sort(durations)
        gaps = np.diff(sorted_durs)
        split_idx = int(np.argmax(gaps))

        dit_group = sorted_durs[:split_idx + 1]
        return int(np.mean(dit_group))

    def _dit_samples_to_wpm(self, dit_samples: int) -> float:
        """Convert dit length in samples to WPM."""
        if dit_samples == 0:
            return 0.0
        dit_seconds = dit_samples / self.sample_rate
        return 60.0 / (50.0 * dit_seconds)
