"""
Signal processing for the Morse decoder.

Applies a bandpass filter to isolate the CW (continuous wave) tone
from background noise, then extracts the envelope (amplitude over time)
which reveals the on/off pattern of the Morse code.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


class SignalProcessor:
    """Bandpass filter + envelope detection for CW Morse signals."""

    def __init__(
        self,
        sample_rate: int = 44100,
        center_frequency: float = 700.0,
        bandwidth: float = 200.0,
        filter_order: int = 4,
        smoothing_ms: float = 20.0,
    ):
        """
        Parameters
        ----------
        sample_rate : int
            Audio sample rate in Hz.
        center_frequency : float
            Center of the bandpass filter, in Hz. Default 700.
        bandwidth : float
            Width of the passband in Hz. Default 200 (so 600-800 Hz).
        filter_order : int
            Butterworth filter order. Higher = sharper but more compute.
        smoothing_ms : float
            Moving-average window length for the envelope, in milliseconds.
        """
        self.sample_rate = sample_rate
        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
        self.filter_order = filter_order
        self.smoothing_ms = smoothing_ms
        self._sos = self._design_bandpass()

    def _design_bandpass(self):
        """Design the Butterworth bandpass filter as second-order sections."""
        nyquist = self.sample_rate / 2.0
        low = (self.center_frequency - self.bandwidth / 2.0) / nyquist
        high = (self.center_frequency + self.bandwidth / 2.0) / nyquist
        return butter(self.filter_order, [low, high], btype="band", output="sos")
    
    def set_center_frequency(self, center_frequency: float) -> None:
        """Retune the bandpass filter to a new center frequency."""
        self.center_frequency = center_frequency
        self._sos = self._design_bandpass()

    def bandpass(self, signal: np.ndarray) -> np.ndarray:
        """Apply zero-phase bandpass filter around the CW frequency."""
        return sosfiltfilt(self._sos, signal)

    def envelope(self, signal: np.ndarray) -> np.ndarray:
        """Extract envelope via Hilbert transform (amplitude over time)."""
        analytic = hilbert(signal)
        return np.abs(analytic)

    def smooth(self, envelope: np.ndarray) -> np.ndarray:
        """Smooth the envelope with a moving-average window."""
        window_samples = max(1, int(self.smoothing_ms * self.sample_rate / 1000.0))
        kernel = np.ones(window_samples) / window_samples
        return np.convolve(envelope, kernel, mode="same")

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Full chain: bandpass -> envelope -> smooth. Returns the smoothed envelope."""
        filtered = self.bandpass(signal)
        env = self.envelope(filtered)
        return self.smooth(env)
