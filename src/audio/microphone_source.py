"""
Microphone audio source. Captures live audio via sounddevice.
"""

import queue

import numpy as np

from src.audio.audio_source import AudioSource

# sounddevice needs the PortAudio system library, which is not present on
# every machine. This class is one of four AudioSource implementations, so a
# missing sound card must not stop a program that only reads files. The
# failure is deferred to the moment capture is actually attempted.
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the host machine
    sd = None
    SOUNDDEVICE_AVAILABLE = False
    _IMPORT_ERROR = str(exc)


def _require_sounddevice() -> None:
    """Raise a clear error if audio capture is not available on this machine."""
    if not SOUNDDEVICE_AVAILABLE:
        raise RuntimeError(
            "Live audio capture is unavailable: " + _IMPORT_ERROR +
            ". Install the sounddevice package and the PortAudio library.")


class MicrophoneSource(AudioSource):
    """
    Captures audio from the system microphone or a virtual audio cable.

    Uses the sounddevice callback API. Each captured block is placed on
    a thread-safe queue and consumed by read_chunk() on the main thread.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_size: int = 1024,
        device_id: int = None,
        queue_max_size: int = 100,
        read_timeout: float = 0.0,
    ):
        super().__init__(sample_rate, chunk_size)
        self.device_id = device_id
        # How long read_chunk() waits for data. 0.0 makes it non-blocking,
        # which is what a GUI timer needs; a script can raise it.
        self.read_timeout = read_timeout
        self._queue = queue.Queue(maxsize=queue_max_size)
        self._stream = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Called by sounddevice on every audio block (runs on its own thread)."""
        if status:
            print(f"Audio callback status: {status}")
        mono = indata[:, 0] if indata.ndim > 1 else indata
        try:
            self._queue.put_nowait(mono.copy())
        except queue.Full:
            pass

    def start(self) -> None:
        """Open the input stream and begin capturing."""
        _require_sounddevice()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            device=self.device_id,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._is_running = True

    def stop(self) -> None:
        """Close the input stream and drain the queue."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._is_running = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def read_chunk(self, timeout: float = None) -> np.ndarray:
        """
        Return the next captured chunk, or an empty array if none is ready.

        Takes no required argument, so this matches the AudioSource interface
        and the class can be swapped for any other source. `self.read_timeout`
        controls how long it waits; the default of 0.0 never blocks.
        """
        if not self._is_running:
            raise RuntimeError("MicrophoneSource not started. Call start() first.")
        wait = self.read_timeout if timeout is None else timeout
        try:
            if wait <= 0:
                return self._queue.get_nowait()
            return self._queue.get(timeout=wait)
        except queue.Empty:
            return np.array([], dtype=np.float32)

    @staticmethod
    def list_devices() -> None:
        """Print all available audio devices on this system."""
        _require_sounddevice()
        print(sd.query_devices())

    @staticmethod
    def input_devices() -> list:
        """
        Return [(device_id, name), ...] for every device that can record.

        A virtual audio cable (VB-CABLE, VoiceMeeter) appears here exactly
        like a physical microphone, which is how audio from a WebSDR in the
        browser reaches this program without going through the air.
        """
        _require_sounddevice()
        devices = []
        for index, info in enumerate(sd.query_devices()):
            if info.get("max_input_channels", 0) > 0:
                devices.append((index, info.get("name", f"device {index}")))
        return devices