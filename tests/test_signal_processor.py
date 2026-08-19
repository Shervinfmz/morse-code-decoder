"""
Verification script for SignalProcessor.

Loads the generated Morse audio, runs it through the processor,
and saves a 3-panel plot showing:
  1. Raw audio (the input)
  2. Bandpass-filtered audio
  3. Smoothed envelope (where Morse code shape is visible)
"""


import os
import sys

# Run this file directly from anywhere: python tests/test_x.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, saves to file
import matplotlib.pyplot as plt

from src.audio.file_source import FileSource
from src.processing.signal_processor import SignalProcessor


def load_full_audio(path: str) -> tuple:
    """Load the entire audio file into memory as a single numpy array."""
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
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_path = os.path.join(root, "audio_samples", "morse_hello_world.wav")
    plot_path = os.path.join(root, "audio_samples", "signal_processor_output.png")

    # Generate the fixture if it is missing, so a fresh clone needs no setup.
    if not os.path.exists(audio_path):
        from scripts.generate_morse_audio import generate_morse_audio
        print("test audio missing - generating it")
        generate_morse_audio("HELLO WORLD", output_path=audio_path, wpm=20)
        print()

    print(f"Loading {audio_path}")
    audio, sample_rate = load_full_audio(audio_path)
    duration = len(audio) / sample_rate
    print(f"  Duration: {duration:.2f} s")
    print(f"  Samples:  {len(audio)}")
    print()

    processor = SignalProcessor(sample_rate=sample_rate)
    print("SignalProcessor configured:")
    print(f"  Center frequency:  {processor.center_frequency} Hz")
    print(f"  Bandwidth:         {processor.bandwidth} Hz")
    print(f"  Filter order:      {processor.filter_order}")
    print(f"  Smoothing:         {processor.smoothing_ms} ms")
    print()

    print("Processing...")
    filtered = processor.bandpass(audio)
    envelope_raw = processor.envelope(filtered)
    smoothed = processor.smooth(envelope_raw)

    print(f"  Raw audio peak:        {np.max(np.abs(audio)):.4f}")
    print(f"  Filtered peak:         {np.max(np.abs(filtered)):.4f}")
    print(f"  Smoothed envelope peak: {np.max(smoothed):.4f}")
    print()

    time = np.arange(len(audio)) / sample_rate
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time, audio, color="#888888", linewidth=0.5)
    axes[0].set_title("Raw audio (input)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, filtered, color="#1f77b4", linewidth=0.5)
    axes[1].set_title("Bandpass filtered (~600-800 Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, smoothed, color="#2ca02c", linewidth=1.0)
    axes[2].set_title("Smoothed envelope (Morse code shape)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()

    print(f"Plot saved to {plot_path}")
    print()
    print("Open the PNG to verify:")
    print(f"  start {plot_path}")
    print()
    print("In the bottom plot you should see clearly:")
    print("  - HELLO as a group of dit/dah pulses")
    print("  - A long flat gap (inter-word)")
    print("  - WORLD as another group of dit/dah pulses")
    print()
    print("SignalProcessor verification: PASS (visual inspection required)")


if __name__ == "__main__":
    main()
