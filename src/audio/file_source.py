"""
File audio source. Streams audio from .wav or .mp3 files in chunks.
"""

from pathlib import Path
import numpy as np
from scipy.io import wavfile
from src.audio.audio_source import AudioSource


class FileSource(AudioSource):
    """
    Streams audio from a file in fixed-size chunks.

    .wav files are read via scipy.
    .mp3 (and other compressed formats) are read via soundfile, which uses
    libsndfile and does NOT require ffmpeg.
    """

    def __init__(
        self,
        filepath: str,
        sample_rate: int = 44100,
        chunk_size: int = 1024,
    ):
        super().__init__(sample_rate, chunk_size)
        self.filepath = filepath
        self._position = 0
        self._samples = None

    def start(self) -> None:
        """Load the audio file into memory and prepare for reading."""
        ext = Path(self.filepath).suffix.lower()

        if ext == ".wav":
            rate, data = wavfile.read(self.filepath)
        else:
            # soundfile handles mp3, flac, ogg, etc. via libsndfile (no ffmpeg).
            import soundfile as sf
            data, rate = sf.read(self.filepath)

        data = np.asarray(data)

        # Stereo -> mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Normalize to float32 in [-1, 1] based on source dtype
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            # soundfile already returns float in [-1, 1]; just ensure float32
            data = data.astype(np.float32)

        self._samples = data
        self.sample_rate = rate
        self._position = 0
        self._is_running = True

    def stop(self) -> None:
        """Release the audio data and reset position."""
        self._is_running = False
        self._position = 0
        self._samples = None

    def read_chunk(self) -> np.ndarray:
        """Return the next chunk of samples. Empty array at EOF."""
        if not self._is_running:
            raise RuntimeError("FileSource not started. Call start() first.")
        if self._samples is None or self._position >= len(self._samples):
            return np.array([], dtype=np.float32)

        end = self._position + self.chunk_size
        chunk = self._samples[self._position:end]
        self._position = end
        return chunk.astype(np.float32)

    @property
    def duration_seconds(self) -> float:
        """Total duration of loaded audio in seconds."""
        if self._samples is None:
            return 0.0
        return len(self._samples) / self.sample_rate

    @property
    def total_samples(self) -> int:
        """Total number of samples in the loaded file."""
        if self._samples is None:
            return 0
        return len(self._samples)

    @property
    def is_finished(self) -> bool:
        """True when all samples have been read."""
        if self._samples is None:
            return True
        return self._position >= len(self._samples)
