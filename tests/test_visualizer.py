"""
Verification script for the Visualizer classes.

Generates a Morse signal, renders all three views (oscilloscope, FFT,
waterfall) into one figure, and saves it as a PNG for inspection.
"""


import os
import sys

# Run this file directly from anywhere: python tests/test_x.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.generate_morse_audio import generate_morse_audio
from src.audio.file_source import FileSource
from src.processing.visualizer import OscilloscopeView, FFTView, WaterfallView


def load_full_audio(path: str):
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
    text = "CQ DE TEST"
    audio_path = "audio_samples/visualizer_test.wav"
    plot_path = "audio_samples/visualizer_output.png"

    print(f"Generating '{text}' at 600 Hz")
    generate_morse_audio(text, audio_path, wpm=20, frequency=600)

    audio, sample_rate = load_full_audio(audio_path)
    print(f"Loaded {len(audio)/sample_rate:.2f} s of audio")
    print()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    print("Rendering OscilloscopeView...")
    OscilloscopeView(sample_rate=sample_rate).render(audio, ax=axes[0])

    print("Rendering FFTView...")
    FFTView(sample_rate=sample_rate).render(audio, ax=axes[1])

    print("Rendering WaterfallView...")
    WaterfallView(sample_rate=sample_rate).render(audio, ax=axes[2])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()

    print()
    print(f"Saved all three views to {plot_path}")
    print(f"  Open it with:  start {plot_path}")
    print()
    print("Visualizer verification: PASS (visual inspection required)")


if __name__ == "__main__":
    main()
