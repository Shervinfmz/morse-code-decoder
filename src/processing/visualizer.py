"""
Signal visualization for the Morse decoder.

Provides an abstract Visualizer base class and three concrete views:
  - OscilloscopeView : time-domain waveform
  - FFTView          : frequency spectrum (single FFT of the whole signal)
  - WaterfallView    : spectrogram (frequency content over time)

Each view follows the same interface so they can be used interchangeably,
mirroring the AudioSource design.
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


class Visualizer(ABC):
    """Abstract base for signal visualizations."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    @abstractmethod
    def render(self, signal: np.ndarray, ax=None):
        """
        Draw the visualization for the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The audio samples to visualize.
        ax : matplotlib axes, optional
            Axes to draw on. If None, a new figure and axes are created.

        Returns
        -------
        matplotlib axes
            The axes the visualization was drawn on.
        """
        raise NotImplementedError

    def _get_axes(self, ax):
        """Return the given axes, or create a new figure+axes if None."""
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))
        return ax


class OscilloscopeView(Visualizer):
    """Time-domain waveform view."""

    def render(self, signal: np.ndarray, ax=None):
        ax = self._get_axes(ax)
        time = np.arange(len(signal)) / self.sample_rate
        ax.plot(time, signal, color="#1f77b4", linewidth=0.5)
        ax.set_title("Oscilloscope (time domain)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        return ax


class FFTView(Visualizer):
    """Frequency-spectrum view (single FFT of the entire signal)."""

    def __init__(self, sample_rate: int = 44100, max_display_freq: float = 2000.0):
        super().__init__(sample_rate)
        self.max_display_freq = max_display_freq

    def render(self, signal: np.ndarray, ax=None):
        ax = self._get_axes(ax)
        if len(signal) == 0:
            ax.set_title("FFT spectrum (no signal)")
            return ax

        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sample_rate)

        mask = freqs <= self.max_display_freq
        ax.plot(freqs[mask], spectrum[mask], color="#9467bd", linewidth=0.8)

        # Mark the dominant frequency
        if np.any(mask) and np.max(spectrum[mask]) > 0:
            peak_idx = int(np.argmax(spectrum[mask]))
            peak_freq = freqs[mask][peak_idx]
            ax.axvline(
                peak_freq, color="#d62728", linestyle="--", linewidth=1.0,
                label=f"Peak: {peak_freq:.0f} Hz",
            )
            ax.legend(loc="upper right")

        ax.set_title("FFT spectrum (frequency domain)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)
        return ax


class WaterfallView(Visualizer):
    """Spectrogram view: frequency content over time (the 'waterfall')."""

    def __init__(self, sample_rate: int = 44100, max_display_freq: float = 2000.0):
        super().__init__(sample_rate)
        self.max_display_freq = max_display_freq

    def render(self, signal: np.ndarray, ax=None):
        ax = self._get_axes(ax)
        if len(signal) == 0:
            ax.set_title("Waterfall (no signal)")
            return ax

        freqs, times, sxx = spectrogram(
            signal,
            fs=self.sample_rate,
            nperseg=1024,
            noverlap=512,
        )

        # Limit the frequency range we display
        mask = freqs <= self.max_display_freq
        freqs = freqs[mask]
        sxx = sxx[mask, :]

        # For long files the spectrogram has tens of thousands of time
        # columns; rendering all of them freezes the GUI. Subsample the
        # columns down to a manageable number for display.
        max_columns = 800
        if sxx.shape[1] > max_columns:
            step = sxx.shape[1] // max_columns
            sxx = sxx[:, ::step]
            times = times[::step]

        # Convert power to decibels for better visual contrast.
        sxx_db = 10 * np.log10(sxx + 1e-12)

        # shading="auto" is faster than "gouraud" and avoids the freeze.
        mesh = ax.pcolormesh(times, freqs, sxx_db, shading="auto", cmap="viridis")
        ax.set_title("Waterfall (spectrogram)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(mesh, ax=ax, label="Power (dB)")
        return ax
