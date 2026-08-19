"""
Verification script for MorseDetector.

Regenerates HELLO WORLD as a known-good Morse audio file, runs it through
the full pipeline (FileSource -> SignalProcessor -> MorseDetector), and
compares the detected Morse string against the expected one.
"""


import os
import sys

# Run this file directly from anywhere: python tests/test_x.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.generate_morse_audio import generate_morse_audio, text_to_morse
from src.audio.file_source import FileSource
from src.processing.signal_processor import SignalProcessor
from src.processing.morse_detector import MorseDetector


def load_full_audio(path: str):
    """Load entire audio file into a single numpy array."""
    source = FileSource(filepath=path, chunk_size=1024)
    source.start()
    sample_rate = source.sample_rate
    chunks = []
    while not source.is_finished:
        chunk = source.read_chunk()
        if len(chunk) > 0:
            chunks.append(chunk)
    source.stop()
    return np.concatenate(chunks), sample_rate


def main() -> None:
    test_text = "HELLO WORLD"
    audio_path = "audio_samples/morse_detector_test.wav"
    test_wpm = 20

    print(f"Step 1: Generating known-good '{test_text}' at {test_wpm} WPM")
    generate_morse_audio(test_text, audio_path, wpm=test_wpm)
    print()

    expected_morse = text_to_morse(test_text).replace(" / ", "   ")
    print(f"Expected Morse string:")
    print(f"  {expected_morse}")
    print()

    print(f"Step 2: Loading audio")
    audio, sample_rate = load_full_audio(audio_path)
    print(f"  Duration: {len(audio) / sample_rate:.2f} s")
    print()

    print(f"Step 3: SignalProcessor")
    processor = SignalProcessor(sample_rate=sample_rate)
    envelope = processor.process(audio)
    print(f"  Envelope peak: {np.max(envelope):.4f}")
    print()

    print(f"Step 4: MorseDetector")
    detector = MorseDetector(sample_rate=sample_rate)
    detected = detector.detect(envelope)
    print(f"  Estimated dit length: {detector.dit_samples} samples ({detector.dit_samples / sample_rate * 1000:.1f} ms)")
    print(f"  Estimated WPM:        {detector.estimated_wpm:.1f} (input was {test_wpm})")
    print()

    print(f"Detected Morse string:")
    print(f"  {detected}")
    print()

    if detected == expected_morse:
        print("MorseDetector verification: PASS")
    else:
        print("MorseDetector verification: FAIL")
        print()
        print(f"  Expected length: {len(expected_morse)}")
        print(f"  Detected length: {len(detected)}")
        print()
        # Show first 10 differences
        diff_count = 0
        for i in range(min(len(detected), len(expected_morse))):
            if detected[i] != expected_morse[i]:
                e_char = expected_morse[i].replace(" ", "_")
                d_char = detected[i].replace(" ", "_")
                print(f"  Position {i}: expected='{e_char}' detected='{d_char}'")
                diff_count += 1
                if diff_count >= 10:
                    print("  ... (more)")
                    break


if __name__ == "__main__":
    main()
