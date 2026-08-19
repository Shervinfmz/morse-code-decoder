"""
Verification script for StreamDecoder.

Simulates a live audio stream: generated Morse audio is pushed through an
ArraySource one 1024-sample block at a time, exactly as a microphone or a
virtual audio cable would deliver it, and the decoder must produce the text
progressively - without ever seeing the whole recording at once.

PASS requires:
  1. text arrives in pieces while the stream is still running
  2. the final text matches what was transmitted
  3. the measured speed is close to the transmitted speed
"""

import os
import sys

# Run this file directly: python tests/test_x.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from scripts.generate_morse_audio import generate_morse_audio
from src.audio.file_source import FileSource
from src.audio.array_source import ArraySource
from src.processing.stream_decoder import StreamDecoder

PHRASE = "CQ DE MORSE DECODER"
WPM = 20
SAMPLE_RATE = 44100
CHUNK = 1024
TRAILING_SILENCE_SECONDS = 1.5


def load_audio(path: str) -> tuple:
    """Read a WAV file into one array via FileSource."""
    source = FileSource(filepath=path, chunk_size=4096)
    source.start()
    rate = source.sample_rate
    chunks = []
    while not source.is_finished:
        chunk = source.read_chunk()
        if len(chunk) > 0:
            chunks.append(chunk)
    source.stop()
    return np.concatenate(chunks), rate


def main() -> None:
    path = os.path.join(ROOT, "audio_samples", "stream_test.wav")
    print(f"Generating {WPM} WPM audio for: {PHRASE}")
    generate_morse_audio(PHRASE, output_path=path, wpm=WPM,
                         sample_rate=SAMPLE_RATE)
    print()

    audio, rate = load_audio(path)

    # The operator stops sending: trailing silence lets the last character settle.
    silence = np.zeros(int(TRAILING_SILENCE_SECONDS * rate), dtype=np.float32)
    audio = np.concatenate([audio, silence])
    print(f"Stream length:   {len(audio) / rate:.2f} s "
          f"({len(audio) // CHUNK} blocks of {CHUNK} samples)")

    # An ArraySource stands in for the live input: same interface, same blocks.
    source = ArraySource(audio, sample_rate=rate, chunk_size=CHUNK)
    decoder = StreamDecoder(sample_rate=rate)

    source.start()
    updates = []
    while not source.is_finished:
        chunk = source.read_chunk()
        if len(chunk) == 0:
            break
        decoder.feed(chunk)
        new_text = decoder.poll()
        if new_text:
            updates.append((decoder.stream_seconds, new_text))
    source.stop()

    # One last decode for anything still settling when the stream ended.
    final = decoder.poll(force=True)
    if final:
        updates.append((decoder.stream_seconds, final))

    print()
    print("Text as it arrived:")
    for when, text in updates:
        print(f"  [{when:6.2f} s]  {text!r}")

    decoded = decoder.text.strip()
    corrected = decoder.corrected_text.strip()

    print()
    print(f"Transmitted:     {PHRASE}")
    print(f"Decoded (raw):   {decoded}")
    print(f"Decoded (LM):    {corrected}")
    print(f"Carrier:         {decoder.carrier_hz:.0f} Hz")
    print(f"Speed:           {decoder.estimated_wpm:.1f} WPM (sent at {WPM})")
    print(f"Elements seen:   {len(decoder.elements)}")
    print()

    progressive = len(updates) > 1
    text_ok = corrected == PHRASE or decoded == PHRASE
    wpm_error = abs(decoder.estimated_wpm - WPM) / WPM * 100.0
    wpm_ok = wpm_error < 15.0

    print(f"  Progressive output (more than one update): "
          f"{'PASS' if progressive else 'FAIL'} ({len(updates)} updates)")
    print(f"  Final text matches transmission:           "
          f"{'PASS' if text_ok else 'FAIL'}")
    print(f"  Speed estimate within 15%:                 "
          f"{'PASS' if wpm_ok else 'FAIL'} ({wpm_error:.1f}% error)")
    print()

    if progressive and text_ok and wpm_ok:
        print("StreamDecoder verification: PASS")
    else:
        print("StreamDecoder verification: FAIL")


if __name__ == "__main__":
    main()
