"""
Verification script for PacedFileSource.

PacedFileSource is the third implementation of AudioSource. It serves a file
at the speed the audio actually plays, so the streaming decoder can be
exercised without a sound card or a radio.

PASS requires:
  1. it releases only what the audio clock has reached, never the whole file
  2. every sample arrives exactly once, in order
  3. the total time taken matches the audio duration divided by `speed`
"""

import os
import sys
import time

# Run this file directly: python tests/test_x.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from scripts.generate_test_tone import generate_test_tone
from src.audio.paced_file_source import PacedFileSource

SPEED = 8.0  # 8x real time, so the test finishes quickly


def main() -> None:
    path = os.path.join(ROOT, "audio_samples", "test_tone.wav")
    if not os.path.exists(path):
        print("test tone missing - generating it")
        generate_test_tone(output_path=path)
        print()

    source = PacedFileSource(path, chunk_size=1024, speed=SPEED)
    source.start()

    duration = source.duration_seconds
    expected_samples = source.total_samples
    print(f"File duration:     {duration:.2f} s")
    print(f"Playback speed:    {SPEED}x  (expect ~{duration / SPEED:.2f} s to drain)")
    print()

    # Immediately after start almost no audio is due yet. A few samples may
    # slip through - the clock advances between start() and this call - but a
    # paced source must never hand over the whole file at once.
    first = source.read_chunk()
    instant_release = len(first)
    print(f"Samples available instantly: {instant_release} "
          f"of {expected_samples} ({instant_release / expected_samples * 100:.2f}%)")

    collected = [first] if len(first) else []
    empties = 0
    started = time.time()
    while not source.is_finished:
        chunk = source.read_chunk()
        if len(chunk) == 0:
            empties += 1
            time.sleep(0.002)
            continue
        collected.append(chunk)
    elapsed = time.time() - started
    source.stop()

    received = np.concatenate(collected) if collected else np.zeros(0)
    print(f"Chunks received:   {len(collected)}")
    print(f"Empty reads:       {empties}   (proves it waited for the clock)")
    print(f"Samples received:  {len(received)} / {expected_samples}")
    print(f"Time taken:        {elapsed:.2f} s")
    print()

    paced_ok = instant_release < 0.01 * expected_samples and empties > 0
    complete_ok = len(received) == expected_samples
    expected_time = duration / SPEED
    timing_ok = abs(elapsed - expected_time) < max(0.5, expected_time * 0.5)

    print(f"  Waits for the audio clock:        {'PASS' if paced_ok else 'FAIL'} "
          f"({empties} empty reads while waiting)")
    print(f"  Delivers every sample once:       {'PASS' if complete_ok else 'FAIL'}")
    print(f"  Takes about the right time:       {'PASS' if timing_ok else 'FAIL'} "
          f"({elapsed:.2f} s vs {expected_time:.2f} s expected)")
    print()

    if paced_ok and complete_ok and timing_ok:
        print("PacedFileSource verification: PASS")
    else:
        print("PacedFileSource verification: FAIL")


if __name__ == "__main__":
    main()
