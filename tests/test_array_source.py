"""
Verification script for ArraySource.

Two checks:

  1. ArraySource hands out every sample it was given, in order, in chunks.
  2. The decoder produces the same text whether the audio arrives from an
     ArraySource or from a FileSource. This is the concrete demonstration
     that the AudioSource abstraction works: the pipeline is written against
     the interface and cannot tell the two apart.
"""

import os
import sys

# Run this file directly: python tests/test_array_source.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from src.audio.array_source import ArraySource
from src.audio.audio_source import AudioSource
from src.audio.file_source import FileSource
from src.processing.decode_pipeline import decode_pipeline
from scripts.generate_morse_audio import generate_morse_audio


def drain(source: AudioSource) -> np.ndarray:
    """Read a source to exhaustion through the abstract interface."""
    source.start()
    chunks = []
    while not source.is_finished:
        chunk = source.read_chunk()
        if len(chunk) > 0:
            chunks.append(chunk)
    source.stop()
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def main() -> None:
    phrase = "HELLO WORLD"
    path = os.path.join(ROOT, "audio_samples", "array_source_demo.wav")

    if not os.path.exists(path):
        print(f"sample missing - generating {phrase}")
        generate_morse_audio(phrase, output_path=path, wpm=20)
        print()

    # --- check 1: chunking is lossless ---------------------------------
    original = np.linspace(-1.0, 1.0, 5000, dtype=np.float32)
    source = ArraySource(original, sample_rate=44100, chunk_size=1024)
    returned = drain(source)

    lossless = (len(returned) == len(original)
                and np.allclose(returned, original))
    print(f"samples in: {len(original)}   samples out: {len(returned)}")
    print(f"chunking lossless: {lossless}")
    print()

    # --- check 2: both sources decode identically ----------------------
    file_source = FileSource(filepath=path, chunk_size=1024)
    from_file = drain(file_source)
    rate = file_source.sample_rate

    from_array = drain(ArraySource(from_file, sample_rate=rate, chunk_size=4096))

    text_file = decode_pipeline(from_file, rate).corrected_text.strip()
    text_array = decode_pipeline(from_array, rate).corrected_text.strip()

    print(f"decoded via FileSource : {text_file}")
    print(f"decoded via ArraySource: {text_array}")
    print(f"expected               : {phrase}")
    print()

    identical = text_file == text_array
    correct = text_array == phrase

    if lossless and identical and correct:
        print("ArraySource verification: PASS")
    else:
        print("ArraySource verification: FAIL")


if __name__ == "__main__":
    main()
