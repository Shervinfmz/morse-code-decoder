"""
Abstract base class for audio input sources.

All audio inputs (microphone, file) must implement this interface
so the rest of the pipeline can consume them uniformly.
"""

from abc import ABC, abstractmethod
import numpy as np


class AudioSource(ABC):
    """Abstract base for audio sources."""

    def __init__(self, sample_rate: int = 44100, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._is_running = False

    @abstractmethod
    def start(self) -> None:
        """Open and begin reading from the source."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Close the source and release resources."""
        raise NotImplementedError

    @abstractmethod
    def read_chunk(self) -> np.ndarray:
        """
        Read the next chunk of audio samples.

        Returns
        -------
        np.ndarray
            1D array of float32 samples in range [-1.0, 1.0].
        """
        raise NotImplementedError

    @property
    def is_running(self) -> bool:
        return self._is_running