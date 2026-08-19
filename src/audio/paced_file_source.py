"""
Real-time file source.

`FileSource` hands out its samples as fast as they are asked for, which is
right for decoding a recording but wrong for testing a live decoder: the whole
file arrives in milliseconds and nothing is ever "streaming".

`PacedFileSource` releases the same samples at the speed the audio actually
plays. Ask it for a chunk before the audio has caught up and it returns an
empty array, exactly as a microphone does when no new block has been captured
yet.

That makes it a stand-in for a live input:

    source = PacedFileSource("recording.wav")   # behaves like a microphone
    source = MicrophoneSource(device_id=3)      # is a microphone

Both are `AudioSource` subclasses, so `StreamDecoder` and the live window
cannot tell them apart. It is used to test the streaming decoder on a machine
with no sound card, and as a rehearsal path for the live window when no radio
is available.

It does NOT make a file decode "live" in any meaningful sense - the audio is
already known. It exists so the live code path can be exercised.
"""

import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from src.audio.audio_source import AudioSource


class PacedFileSource(AudioSource):
    """Serve a file through the AudioSource interface at real-time speed."""

    def __init__(self, filepath: str, sample_rate: int = 44100,
                 chunk_size: int = 1024, speed: float = 1.0):
        """
        Parameters
        ----------
        filepath : str
            A .wav file, or any format soundfile can read.
        chunk_size : int
            Samples per chunk, matching the live capture block size.
        speed : float
            Playback rate. 1.0 is real time; 2.0 releases audio twice as fast,
            which is useful in tests that should not take a full minute.
        """
        super().__init__(sample_rate, chunk_size)
        self.filepath = filepath
        self.speed = speed
        self._samples = None
        self._position = 0
        self._started_at = None

    def start(self) -> None:
        """Load the file and start the clock."""
        extension = Path(self.filepath).suffix.lower()
        if extension == ".wav":
            rate, data = wavfile.read(self.filepath)
        else:
            import soundfile as sf
            data, rate = sf.read(self.filepath)

        data = np.asarray(data)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)

        self._samples = data
        self.sample_rate = rate
        self._position = 0
        self._started_at = time.time()
        self._is_running = True

    def stop(self) -> None:
        """Stop serving samples."""
        self._is_running = False
        self._position = 0
        self._samples = None
        self._started_at = None

    def read_chunk(self) -> np.ndarray:
        """
        Return the next chunk if the clock has reached it, else an empty array.

        Never blocks, so a GUI timer can call it freely.
        """
        if not self._is_running:
            raise RuntimeError("PacedFileSource not started. Call start() first.")
        if self._samples is None or self._position >= len(self._samples):
            return np.array([], dtype=np.float32)

        elapsed = (time.time() - self._started_at) * self.speed
        released = int(elapsed * self.sample_rate)
        if released <= self._position:
            return np.array([], dtype=np.float32)

        end = min(self._position + self.chunk_size, released, len(self._samples))
        chunk = self._samples[self._position:end]
        self._position = end
        return chunk.astype(np.float32)

    @property
    def duration_seconds(self) -> float:
        if self._samples is None:
            return 0.0
        return len(self._samples) / self.sample_rate

    @property
    def total_samples(self) -> int:
        return 0 if self._samples is None else len(self._samples)

    @property
    def is_finished(self) -> bool:
        if self._samples is None:
            return True
        return self._position >= len(self._samples)
