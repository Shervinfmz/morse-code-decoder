"""
Unified decode pipeline.

Single entry point every part of the application uses to turn audio into text.
It runs the full chain in order:

    frequency detection -> adaptive filtering/envelope -> Morse detection
    -> Morse-to-text -> language-model correction

and returns BOTH the raw decoded text and the corrected text, plus the
detector and threshold so a GUI can compute per-character timing for a
synchronised playback reveal.

Because every GUI, script, and notebook calls this one function, the adaptive
smoothing and the language model are applied everywhere automatically.

The LanguageModel is built once at module load and reused.
"""

from dataclasses import dataclass

import numpy as np

from src.processing.frequency_detector import FrequencyDetector
from src.processing.signal_processor import SignalProcessor
from src.processing.morse_detector import MorseDetector
from src.processing.morse_decoder import MorseDecoder
from src.processing.adaptive_decoder import (
    BOOTSTRAP_SMOOTHING_MS, adaptive_smoothing_ms,
)
from src.processing.language_model import LanguageModel


_LANGUAGE_MODEL = None


def get_language_model() -> LanguageModel:
    """Return the shared LanguageModel, building it on first use."""
    global _LANGUAGE_MODEL
    if _LANGUAGE_MODEL is None:
        _LANGUAGE_MODEL = LanguageModel()
    return _LANGUAGE_MODEL


@dataclass
class PipelineResult:
    """The complete output of the decode pipeline."""
    carrier_hz: float
    smoothing_ms: float
    estimated_wpm: float
    envelope: np.ndarray
    morse: str
    raw_text: str            # decoder output, before language correction
    corrected_text: str      # after language-model correction
    was_corrected: bool
    detector: MorseDetector  # the detector used (has dit_samples etc.)
    threshold_ratio: float


def decode_pipeline(audio: np.ndarray, sample_rate: int,
                    center_frequency: float = None,
                    threshold_ratio: float = 0.30,
                    apply_correction: bool = True) -> PipelineResult:
    """
    Run the full decode pipeline (adaptive smoothing + language correction).

    Returns a PipelineResult with raw text, corrected text, the envelope used,
    and the detector (so callers can compute per-character timing).
    """
    if center_frequency is None:
        center_frequency = FrequencyDetector(sample_rate=sample_rate).detect(audio)

    # --- Pass 1: rough dit estimate with a moderate window ---
    bootstrap = SignalProcessor(sample_rate=sample_rate,
                                center_frequency=center_frequency,
                                smoothing_ms=BOOTSTRAP_SMOOTHING_MS)
    env1 = bootstrap.process(audio)
    det1 = MorseDetector(sample_rate=sample_rate, threshold_ratio=threshold_ratio)
    det1.detect(env1)
    dit1 = getattr(det1, "dit_samples", None)

    smoothing = adaptive_smoothing_ms(dit1, sample_rate)

    # --- Pass 2: decode with the tuned window ---
    processor = SignalProcessor(sample_rate=sample_rate,
                                center_frequency=center_frequency,
                                smoothing_ms=smoothing)
    envelope = processor.process(audio)
    detector = MorseDetector(sample_rate=sample_rate, threshold_ratio=threshold_ratio)
    morse = detector.detect(envelope)
    raw_text = MorseDecoder().decode(morse)
    wpm = detector.estimated_wpm if detector.estimated_wpm else 0.0

    # --- Language-model correction ---
    if apply_correction and raw_text:
        corrected = get_language_model().correct(raw_text)
    else:
        corrected = raw_text

    return PipelineResult(
        carrier_hz=float(center_frequency),
        smoothing_ms=float(smoothing),
        estimated_wpm=float(wpm),
        envelope=envelope,
        morse=morse,
        raw_text=raw_text,
        corrected_text=corrected,
        was_corrected=(corrected != raw_text),
        detector=detector,
        threshold_ratio=threshold_ratio,
    )
