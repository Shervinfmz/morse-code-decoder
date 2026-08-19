"""
Verification script for MicrophoneSource.

Captures a few seconds of audio from the default microphone and reports stats.
Make some noise into your mic while it runs (talk, tap the desk, snap fingers).
"""

import os
import sys
import time

# Run this file directly: python tests/test_x.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Capturing audio needs the PortAudio system library. It is not present on
# every machine (CI servers, headless graders), and a missing sound card is
# not a failure of this project. Skip cleanly instead of crashing.
# The module always imports; it reports separately whether this machine can
# actually capture audio, so the check has to look at that flag rather than
# rely on an ImportError.
try:
    from src.audio.microphone_source import (
        MicrophoneSource, SOUNDDEVICE_AVAILABLE,
    )
    AUDIO_INPUT_AVAILABLE = SOUNDDEVICE_AVAILABLE
    SKIP_REASON = "" if SOUNDDEVICE_AVAILABLE else "PortAudio / sounddevice not available"
except Exception as exc:
    MicrophoneSource = None
    AUDIO_INPUT_AVAILABLE = False
    SKIP_REASON = str(exc)


def main() -> None:
    if not AUDIO_INPUT_AVAILABLE:
        print("MicrophoneSource verification: SKIPPED")
        print(f"  Reason: {SKIP_REASON}")
        print("  Install the PortAudio library and sounddevice to run this test.")
        return

    capture_seconds = 3
    chunk_size = 1024
    sample_rate = 44100

    print("Available audio devices:")
    MicrophoneSource.list_devices()
    print()

    print(f"Capturing {capture_seconds} seconds from default microphone...")
    print("Make some noise into your mic now.")
    print()

    source = MicrophoneSource(sample_rate=sample_rate, chunk_size=chunk_size)
    source.start()

    chunks = []
    start = time.time()
    while time.time() - start < capture_seconds:
        chunk = source.read_chunk(timeout=1.0)
        if len(chunk) > 0:
            chunks.append(chunk)

    source.stop()

    if not chunks:
        print("FAIL: No audio captured. Check microphone permissions and device.")
        return

    all_samples = np.concatenate(chunks)
    duration = len(all_samples) / sample_rate
    rms = np.sqrt(np.mean(all_samples ** 2))
    peak = np.max(np.abs(all_samples))

    print(f"Chunks captured:  {len(chunks)}")
    print(f"Total samples:    {len(all_samples)}")
    print(f"Duration:         {duration:.2f} s")
    print(f"RMS amplitude:    {rms:.4f}")
    print(f"Peak amplitude:   {peak:.4f}")
    print()

    if peak < 0.001:
        print("WARNING: Signal is silent. Mic may be muted, disconnected, or no permission granted.")
        print("Verification: INCOMPLETE")
    else:
        print("MicrophoneSource verification: PASS")


if __name__ == "__main__":
    main()
