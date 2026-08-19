"""
In-memory audio source.

`FileSource` serves samples that live on disk. `ArraySource` serves samples
that are already in memory - a generated test signal, a slice of a longer
recording, or the output of another processing step.

Both are `AudioSource` subclasses, so nothing downstream can tell them apart.
That is the point of the abstract base class: the decoder is written against
the interface, not against a file reader.

The test suite uses this to feed a generated signal straight into the pipeline
without writing a temporary WAV file first.
"""

import numpy as np

from src.audio.audio_source import AudioSource


class ArraySource(AudioSource):
    """Serve an existing numpy array through the AudioSource interface."""

    def __init__(self, samples: np.ndarray, sample_rate: int = 44100,
                 chunk_size: int = 1024):
        """
        Parameters
        ----------
        samples : np.ndarray
            1D array of float samples, nominally in the range [-1.0, 1.0].
        sample_rate : int
            Sample rate of `samples`, in Hz.
        chunk_size : int
            Number of samples returned per `read_chunk()` call.
        """
        super().__init__(sample_rate=sample_rate, chunk_size=chunk_size)
        self._samples = np.asarray(samples, dtype=np.float32).ravel()
        self._position = 0

    def start(self) -> None:
        """Rewind to the beginning and mark the source as running."""
        self._position = 0
        self._is_running = True

    def stop(self) -> None:
        """Mark the source as stopped. There is nothing to release."""
        self._is_running = False

    def read_chunk(self) -> np.ndarray:
        """
        Return the next `chunk_size` samples.

        Returns a shorter array on the final chunk, and an empty array once
        the data is exhausted - the same contract as FileSource.
        """
        if not self._is_running:
            raise RuntimeError("ArraySource not started. Call start() first.")
        chunk = self._samples[self._position:self._position + self.chunk_size]
        self._position += len(chunk)
        return chunk

    @property
    def duration_seconds(self) -> float:
        """Length of the buffer in seconds."""
        return len(self._samples) / self.sample_rate

    @property
    def total_samples(self) -> int:
        """Number of samples in the buffer."""
        return len(self._samples)

    @property
    def is_finished(self) -> bool:
        """True once every sample has been handed out."""
        return self._position >= len(self._samples)
