"""
Carrier frequency detection for Morse signals.

Before bandpass filtering, we need to know WHERE the Morse tone is in the
frequency spectrum. A hardcoded assumption (e.g. 700 Hz) fails on real-world
audio that may use any tone. This class finds the dominant frequency using
an FFT, so the bandpass filter can be tuned to the actual signal.
"""

import numpy as np


class FrequencyDetector:
    """Detect the dominant (carrier) frequency of an audio signal via FFT."""

    def __init__(
        self,
        sample_rate: int = 44100,
        min_freq: float = 100.0,
        max_freq: float = 3000.0,
    ):
        """
        Parameters
        ----------
        sample_rate : int
            Audio sample rate in Hz.
        min_freq : float
            Lowest frequency to consider as a possible carrier (Hz).
            Rejects low-frequency hum and DC offset.
        max_freq : float
            Highest frequency to consider (Hz). Rejects high-frequency hiss.
        """
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq

    def detect(self, signal: np.ndarray) -> float:
        """
        Find the dominant frequency in the signal.

        Computes the magnitude spectrum via FFT, restricts to the plausible
        Morse frequency band [min_freq, max_freq], and returns the frequency
        with the most energy.

        Returns
        -------
        float
            The dominant frequency in Hz. Returns 0.0 for a silent signal.
        """
        if len(signal) == 0 or np.max(np.abs(signal)) < 1e-6:
            return 0.0

        # Real FFT: magnitude spectrum of the signal
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sample_rate)

        # Restrict to the plausible Morse carrier band
        band_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        if not np.any(band_mask):
            return 0.0

        band_spectrum = spectrum[band_mask]
        band_freqs = freqs[band_mask]

        # The frequency with the most energy is the carrier
        peak_index = int(np.argmax(band_spectrum))
        return float(band_freqs[peak_index])

    def get_spectrum(self, signal: np.ndarray) -> tuple:
        """
        Return the full magnitude spectrum for plotting/visualization.

        Returns
        -------
        (freqs, magnitudes) : tuple of np.ndarray
            Frequencies in Hz and their corresponding magnitudes.
        """
        if len(signal) == 0:
            return np.array([]), np.array([])
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sample_rate)
        return freqs, spectrum
