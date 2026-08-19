"""
Generate a simple 700 Hz test tone as a WAV file.

This simulates a continuous CW carrier — equivalent to a long "key down"
in Morse. Useful as a baseline sanity check before we work with real
Morse signals.
"""

import os
import numpy as np
from scipy.io import wavfile


def generate_test_tone(
    output_path: str = "audio_samples/test_tone.wav",
    frequency: float = 700.0,
    duration: float = 2.0,
    sample_rate: int = 44100,
) -> None:
    """Generate a sine wave and save as 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    pcm = (wave * 32767).astype(np.int16)

    wavfile.write(output_path, sample_rate, pcm)

    print(f"Generated {output_path}")
    print(f"  Frequency:   {frequency} Hz")
    print(f"  Duration:    {duration} s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Total samples: {len(pcm)}")


if __name__ == "__main__":
    generate_test_tone()