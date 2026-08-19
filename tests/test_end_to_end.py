"""
End-to-end pipeline verification.

For each test phrase:
  1. Generate Morse audio from the phrase
  2. Load the audio via FileSource
  3. Process: SignalProcessor -> MorseDetector -> MorseDecoder
  4. Compare decoded text to the original phrase

PASS only if all phrases round-trip correctly.
"""


import os
import sys

# Run this file directly from anywhere: python tests/test_x.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.generate_morse_audio import generate_morse_audio
from src.audio.file_source import FileSource
from src.processing.signal_processor import SignalProcessor
from src.processing.morse_detector import MorseDetector
from src.processing.morse_decoder import MorseDecoder


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


def run_pipeline(text: str, wpm: int) -> tuple:
    """Run text through the full pipeline. Returns (decoded_text, estimated_wpm)."""
    audio_path = "audio_samples/pipeline_test.wav"
    generate_morse_audio(text, audio_path, wpm=wpm)

    audio, sample_rate = load_full_audio(audio_path)

    processor = SignalProcessor(sample_rate=sample_rate)
    envelope = processor.process(audio)

    detector = MorseDetector(sample_rate=sample_rate)
    morse_string = detector.detect(envelope)

    decoder = MorseDecoder()
    decoded = decoder.decode(morse_string)

    return decoded, detector.estimated_wpm


def main() -> None:
    test_cases = [
        ("HELLO WORLD", 20),
        ("CQ DE TEST", 20),
        ("THE QUICK BROWN FOX", 15),
        ("ABC 123", 25),
    ]

    print("=" * 60)
    print("End-to-end pipeline verification")
    print("=" * 60)
    print()

    results = []
    for text, wpm in test_cases:
        print(f"Input:    '{text}' @ {wpm} WPM")
        decoded, estimated_wpm = run_pipeline(text, wpm)
        passed = decoded == text
        status = "PASS" if passed else "FAIL"
        print(f"Decoded:  '{decoded}'")
        print(f"Est. WPM: {estimated_wpm:.1f}")
        print(f"Result:   {status}")
        print()
        results.append((text, decoded, passed))

    print("=" * 60)
    passed_count = sum(1 for _, _, p in results if p)
    total = len(results)
    print(f"Summary: {passed_count}/{total} test cases passed")
    print("=" * 60)
    if passed_count == total:
        print()
        print("End-to-end pipeline: PASS")
        print("The core decoder is functional.")
    else:
        print()
        print("End-to-end pipeline: FAIL")
        print("Failed cases:")
        for text, decoded, passed in results:
            if not passed:
                print(f"  Expected: '{text}'")
                print(f"  Got:      '{decoded}'")


if __name__ == "__main__":
    main()
