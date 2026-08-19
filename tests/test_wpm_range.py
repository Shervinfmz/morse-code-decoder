"""
WPM range verification: 10 to 40 words per minute.

Generates the same message at every speed in the range, decodes it with the
adaptive two-pass pipeline, and reports:
  - whether the decoded text matches exactly
  - the smoothing window the adaptive step chose
  - how close the estimated WPM is to the true WPM

Adaptive smoothing matters here: a fixed window either blurs fast Morse or
lets noise through on slow Morse. Choosing the window from the measured dit
length keeps the estimate accurate across the whole range.

Run from the project root:
    python -m tests.test_wpm_range
"""


import os
import sys

# Run this file directly from anywhere: python tests/test_x.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.generate_morse_audio import generate_morse_audio
from src.audio.file_source import FileSource
from src.processing.adaptive_decoder import decode_adaptive


def load_full_audio(path):
    source = FileSource(filepath=path, chunk_size=1024)
    source.start()
    sample_rate = source.sample_rate
    chunks = []
    while not source.is_finished:
        chunk = source.read_chunk()
        if len(chunk) > 0:
            chunks.append(chunk)
    source.stop()
    if not chunks:
        return np.array([], dtype=np.float32), sample_rate
    return np.concatenate(chunks), sample_rate


def main():
    text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 1234567890"
    speeds = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40]
    path = "audio_samples/wpm_range_test.wav"

    print("WPM range verification (target: 10-40 WPM, adaptive smoothing)")
    print("=" * 72)
    print(f"Message: {text}")
    print("=" * 72)
    print(f"{'WPM':>5}{'Smoothing':>12}{'Est WPM':>10}{'Error%':>9}   {'Result'}")
    print("-" * 72)

    passed = 0
    errors = []
    for wpm in speeds:
        generate_morse_audio(text, path, wpm=wpm, frequency=600)
        audio, sample_rate = load_full_audio(path)
        result = decode_adaptive(audio, sample_rate)

        ok = result.text == text
        passed += ok
        err = abs(result.estimated_wpm - wpm) / wpm * 100 if wpm else 0.0
        errors.append(err)
        print(f"{wpm:>5}{result.smoothing_ms:>11.1f}ms"
              f"{result.estimated_wpm:>10.1f}{err:>8.1f}%   "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"      decoded: {result.text!r}")

    print("-" * 72)
    print(f"{passed}/{len(speeds)} speeds decoded exactly")
    print(f"WPM estimate error: mean {np.mean(errors):.1f}%, max {np.max(errors):.1f}%")
    if passed == len(speeds):
        print("RESULT: full 10-40 WPM range works.")
    else:
        print("RESULT: some speeds failed - see above.")


if __name__ == "__main__":
    main()
