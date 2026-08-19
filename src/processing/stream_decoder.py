"""
Real-time streaming Morse decoder.

The file decoder waits for a complete recording. A live stream never ends, so
this class decodes continuously from a rolling buffer of the most recent
audio.

Why a rolling buffer is necessary
---------------------------------
Two parts of the decoder need context, not just the current instant:

  * the on/off threshold is a fraction of the loudest point in the envelope
  * the dit length is estimated by clustering all the element durations

A single 1024-sample block (about 23 ms) contains neither. Keeping the last
few seconds gives both, and lets the existing `decode_pipeline` run unchanged
on that window.

How new characters are identified
---------------------------------
Every decode re-reads the whole buffer, so most of what it returns has already
been reported. Two rules decide what is genuinely new:

  1. **Settled.** A character is only emitted once at least `settle_seconds`
     of audio follows it. A character is not finished until the gap after it
     proves it is finished, so emitting immediately would report a `T` that is
     about to become a `TE`. This is the source of the decoder's latency, and
     it is real: text appears roughly half a second after it is sent.

  2. **Newer than the last one.** Character times are absolute - measured from
     the moment the stream started, not from the start of the buffer - so a
     character already emitted is recognised even after the buffer has slid
     past it.

Threading
---------
`feed()` is cheap and only appends to the buffer. `poll()` does the decoding
and is expensive. They are separated so a caller can feed from the audio
thread and poll from a worker thread without the GUI stalling. A lock guards
the buffer.

Known limitation
----------------
The dit-length estimate needs both dits and dahs inside the window to find the
boundary between them. A window containing only dits (a long run of `E`s)
cannot be scaled reliably. In normal traffic the mix arrives well within a
few seconds.
"""

import threading

import numpy as np

from src.processing.frequency_detector import FrequencyDetector
from src.processing.decode_pipeline import decode_pipeline, get_language_model
from src.processing.timing import character_events, element_segments


# Enough audio to estimate a dit length before the first decode is attempted.
MIN_AUDIO_SECONDS = 2.0

# Below this peak amplitude the window is treated as silence and skipped.
SILENCE_PEAK = 1e-4

# A CW signal is a single steady tone, so nearly all of its energy sits within
# a narrow band around one frequency. Noise spreads its energy everywhere.
# `carrier_quality` measures the fraction of 100-3000 Hz energy lying within
# CARRIER_BAND_HZ of the strongest bin. Measured values:
#
#     empty band, receiver noise         0.12
#     Morse buried in heavy noise        0.65
#     Morse in moderate noise            0.83
#     clean Morse                        0.99
#
# Below MIN_CARRIER_QUALITY the window is not decoded at all. Without this the
# decoder locks onto whatever bin is loudest in the noise, watches it wander,
# and reports the characters that wandering produces - text that looks like a
# decode but means nothing.
CARRIER_BAND_HZ = 60.0
MIN_CARRIER_QUALITY = 0.35

# Settle time is scaled to the measured speed: a character is emitted once
# this many dit lengths of audio have followed it. A word gap is 7 dits, so 8
# guarantees the following gap has been seen in full.
SETTLE_DITS = 8.0
MIN_SETTLE_SECONDS = 0.20
MAX_SETTLE_SECONDS = 1.50

# Elements older than this are dropped from the display history.
ELEMENT_HISTORY_SECONDS = 120.0


class StreamDecoder:
    """
    Decode Morse from a continuous audio stream.

    Usage:
        decoder = StreamDecoder(sample_rate=44100)
        while streaming:
            decoder.feed(source.read_chunk())
            new_text = decoder.poll()
            if new_text:
                print(new_text, end="")
    """

    def __init__(self, sample_rate: int = 44100,
                 window_seconds: float = 8.0,
                 decode_interval_seconds: float = 0.6,
                 threshold_ratio: float = 0.30,
                 apply_correction: bool = True):
        """
        Parameters
        ----------
        sample_rate : int
            Sample rate of the incoming audio, in Hz.
        window_seconds : float
            How much recent audio to keep and decode. Longer gives a better
            speed estimate; shorter costs less per decode.
        decode_interval_seconds : float
            Minimum audio between decodes. The decode itself takes roughly
            30 ms per second of buffer, so this sets the CPU load.
        threshold_ratio : float
            Passed straight to MorseDetector.
        apply_correction : bool
            Correct completed words with the language model as they finish.
        """
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.decode_interval_seconds = decode_interval_seconds
        self.threshold_ratio = threshold_ratio
        self.apply_correction = apply_correction

        self._lock = threading.Lock()
        self._detector = FrequencyDetector(sample_rate=sample_rate)
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Clear the buffer and all decoded output."""
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)
            self._buffer_start_sample = 0   # absolute index of _buffer[0]
            self._total_samples = 0         # samples ever received
            self._samples_since_decode = 0

        self.text = ""
        self.events = []                    # CharacterEvent actually emitted
        self.carrier_hz = 0.0
        self.estimated_wpm = 0.0
        self.elements = []                  # ElementSegment, absolute times
        self.envelope = None                # envelope of the last decode
        self.envelope_start_seconds = 0.0
        self.character_latency_seconds = 0.0  # how late the newest character was
        self.carrier_quality = 0.0          # 0 = noise, 1 = a single pure tone
        self.peak_hz = 0.0                  # strongest frequency, decoded or not
        self.signal_present = False         # is there a CW signal worth decoding?
        self._last_emitted_start = -1e9
        self._word_cache = {}
        self.decode_count = 0

    @property
    def stream_seconds(self) -> float:
        """Seconds of audio received since the stream started."""
        return self._total_samples / self.sample_rate

    @property
    def corrected_text(self) -> str:
        """Decoded text with completed words passed through the language model."""
        if not self.apply_correction or not self.text:
            return self.text
        parts = self.text.split(" ")
        finished, current = parts[:-1], parts[-1]
        model = get_language_model()
        out = []
        for word in finished:
            if not word:
                out.append(word)
                continue
            if word not in self._word_cache:
                self._word_cache[word] = model.correct_word(word)
            out.append(self._word_cache[word])
        out.append(current)
        return " ".join(out)

    def recent_audio(self, seconds: float) -> tuple:
        """
        Return (samples, start_seconds) for the most recent `seconds` of audio.

        Used by the display, which draws the same window the decoder sees.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return np.zeros(0, dtype=np.float32), 0.0
            wanted = int(seconds * self.sample_rate)
            samples = self._buffer[-wanted:] if wanted < len(self._buffer) else self._buffer
            start_sample = self._buffer_start_sample + (len(self._buffer) - len(samples))
            return samples.copy(), start_sample / self.sample_rate

    # ------------------------------------------------------------------ input

    def feed(self, chunk: np.ndarray) -> None:
        """
        Add a block of samples. Cheap: appends and trims, nothing else.

        Safe to call from an audio thread while another thread polls.
        """
        if chunk is None or len(chunk) == 0:
            return
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        with self._lock:
            self._buffer = np.concatenate([self._buffer, chunk])
            self._total_samples += len(chunk)
            self._samples_since_decode += len(chunk)

            # Drop anything older than the window, remembering how far the
            # buffer has slid so absolute times stay correct.
            max_samples = int(self.window_seconds * self.sample_rate)
            excess = len(self._buffer) - max_samples
            if excess > 0:
                self._buffer = self._buffer[excess:]
                self._buffer_start_sample += excess

    # ---------------------------------------------------------------- decoding

    def poll(self, force: bool = False) -> str:
        """
        Decode if enough new audio has arrived, and return newly settled text.

        Returns an empty string when there is nothing new. This is the
        expensive call - run it from a worker thread in a GUI.
        """
        with self._lock:
            enough_new = (self._samples_since_decode
                          >= self.decode_interval_seconds * self.sample_rate)
            if not force and not enough_new:
                return ""
            if len(self._buffer) < MIN_AUDIO_SECONDS * self.sample_rate:
                return ""
            buffer = self._buffer.copy()
            buffer_start_seconds = self._buffer_start_sample / self.sample_rate
            self._samples_since_decode = 0

        peak = float(np.max(np.abs(buffer))) if len(buffer) else 0.0
        if peak < SILENCE_PEAK:
            self.carrier_quality = 0.0
            self.signal_present = False
            return ""

        # Refuse to decode noise. See MIN_CARRIER_QUALITY above.
        self.carrier_quality = self._carrier_quality(buffer)
        self.signal_present = self.carrier_quality >= MIN_CARRIER_QUALITY
        if not self.signal_present:
            return ""

        # Skip the window if no plausible carrier is present. Without this the
        # bandpass filter would be asked for a negative cutoff on silence.
        carrier = self._detector.detect(buffer)
        if carrier <= 0.0:
            return ""

        try:
            result = decode_pipeline(buffer, self.sample_rate,
                                     center_frequency=carrier,
                                     threshold_ratio=self.threshold_ratio,
                                     apply_correction=False)
        except Exception:
            # A malformed window must never kill the stream.
            return ""

        self.decode_count += 1
        self.carrier_hz = result.carrier_hz
        self.estimated_wpm = result.estimated_wpm
        self.envelope = result.envelope
        self.envelope_start_seconds = buffer_start_seconds

        buffer_end_seconds = buffer_start_seconds + len(buffer) / self.sample_rate
        settle = self._settle_seconds(result.detector)
        cutoff = buffer_end_seconds - settle

        self._update_elements(result, buffer_start_seconds)

        events = character_events(result.envelope, result.detector,
                                  self.sample_rate,
                                  offset_seconds=buffer_start_seconds)

        # Successive windows measure the same character a few milliseconds
        # apart, because the threshold moves with the window's peak. Comparing
        # start times with half a dit of tolerance is far wider than that
        # jitter and far narrower than the gap between two real characters.
        tolerance = 0.5 * self._dit_seconds(result.detector)

        new_text = []
        for event in events:
            if event.end_seconds > cutoff:
                break                      # not settled yet, and neither is anything after it
            if event.start_seconds <= self._last_emitted_start + tolerance:
                continue                   # already reported on an earlier decode
            self._last_emitted_start = event.start_seconds
            if event.character == " " and not self.text and not new_text:
                continue                   # never start the output with a space
            new_text.append(event.character)
            self.events.append(event)
            # How far behind the live edge this character was reported. This
            # is the real latency: the wait needed for the following gap to
            # prove the character had ended.
            self.character_latency_seconds = buffer_end_seconds - event.end_seconds

        emitted = "".join(new_text)
        self.text += emitted
        return emitted

    def _dit_seconds(self, detector) -> float:
        """Measured dit length in seconds, or a safe default."""
        dit_samples = getattr(detector, "dit_samples", None)
        if not dit_samples:
            return MIN_SETTLE_SECONDS / SETTLE_DITS
        return dit_samples / self.sample_rate

    def _carrier_quality(self, buffer) -> float:
        """
        How concentrated the spectrum is around its strongest frequency.

        Returns roughly 1.0 for a pure tone and roughly 0.1 for band noise.
        Also records the strongest frequency in `peak_hz`, so the display can
        show where the energy is even when the window is too noisy to decode.
        That reading is what makes the panel usable as a tuning aid.
        """
        window = np.hanning(len(buffer))
        magnitude = np.abs(np.fft.rfft(buffer * window))
        freqs = np.fft.rfftfreq(len(buffer), 1.0 / self.sample_rate)
        band = (freqs >= 100.0) & (freqs <= 3000.0)
        magnitude, freqs = magnitude[band], freqs[band]
        total = float(np.sum(magnitude ** 2))
        if total <= 0:
            return 0.0
        peak_freq = freqs[int(np.argmax(magnitude))]
        self.peak_hz = float(peak_freq)
        near = np.abs(freqs - peak_freq) <= CARRIER_BAND_HZ
        return float(np.sum(magnitude[near] ** 2) / total)

    def _settle_seconds(self, detector) -> float:
        """How long to wait after a character before trusting it is complete."""
        dit_seconds = self._dit_seconds(detector)
        return float(np.clip(SETTLE_DITS * dit_seconds,
                             MIN_SETTLE_SECONDS, MAX_SETTLE_SECONDS))

    def _update_elements(self, result, buffer_start_seconds: float) -> None:
        """Merge this window's dits and dahs into the display history."""
        segments = element_segments(result.envelope, result.detector,
                                    self.sample_rate,
                                    offset_seconds=buffer_start_seconds)
        if not segments:
            return
        known_end = self.elements[-1].end_seconds if self.elements else -1.0
        # Half a dit of tolerance stops the same element being added twice when
        # consecutive windows disagree slightly about where it ended.
        tolerance = 0.5 * (segments[0].duration_seconds or 0.01)
        for segment in segments:
            if segment.end_seconds > known_end + tolerance:
                self.elements.append(segment)

        oldest = self.stream_seconds - ELEMENT_HISTORY_SECONDS
        if oldest > 0:
            self.elements = [s for s in self.elements if s.end_seconds >= oldest]
