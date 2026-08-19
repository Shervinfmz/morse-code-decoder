"""
Verification script for FileSource.

Loads the test tone, reads it chunk by chunk, and reports basic stats.
Confirms the audio pipeline can read a file end-to-end.
"""


import os
import sys

# Run this file directly from anywhere: python tests/test_x.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.file_source import FileSource


def main() -> None:
    path = "audio_samples/test_tone.wav"
    source = FileSource(filepath=path, chunk_size=1024)

    print(f"Opening: {path}")
    source.start()

    print(f"  Sample rate:     {source.sample_rate} Hz")
    print(f"  Total samples:   {source.total_samples}")
    print(f"  Duration:        {source.duration_seconds:.2f} s")
    print(f"  Chunk size:      {source.chunk_size} samples")
    print()

    chunk_count = 0
    total_samples_read = 0
    first_chunk_preview = None

    while not source.is_finished:
        chunk = source.read_chunk()
        if len(chunk) == 0:
            break
        if chunk_count == 0:
            first_chunk_preview = chunk[:5]
        chunk_count += 1
        total_samples_read += len(chunk)

    expected_samples = source.total_samples
    source.stop()

    print(f"First chunk preview: {first_chunk_preview}")
    print(f"Chunks read:         {chunk_count}")
    print(f"Samples read:        {total_samples_read} / {expected_samples}")
    print()

    if total_samples_read == expected_samples:
        print("FileSource verification: PASS")
    else:
        print("FileSource verification: FAIL (sample count mismatch)")


if __name__ == "__main__":
    main()