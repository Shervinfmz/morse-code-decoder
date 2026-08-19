"""
Morse Code Decoder - command-line runner.

Runs the full decoding pipeline on an audio file and prints the result.

Usage:
    # Decode a specific audio file
    python -m run_decoder path/to/audio.wav

    # Generate a test signal and decode it (no file needed)
    python -m run_decoder --demo

    # Generate custom text as a test signal and decode it
    python -m run_decoder --text "HELLO WORLD" --wpm 20
"""

import sys
import argparse
import numpy as np

from src.audio.file_source import FileSource
from src.processing.signal_processor import SignalProcessor
from src.processing.morse_detector import MorseDetector
from src.processing.morse_decoder import MorseDecoder


def load_full_audio(path: str):
    """Load an entire audio file into a single numpy array via FileSource."""
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


def decode_audio_file(path: str) -> None:
    """Run the full pipeline on an audio file and print the decoded text."""
    print(f"Loading audio: {path}")
    audio, sample_rate = load_full_audio(path)

    if len(audio) == 0:
        print("ERROR: No audio data loaded. Check the file path and format.")
        return

    print(f"  Duration:    {len(audio) / sample_rate:.2f} s")
    print(f"  Sample rate: {sample_rate} Hz")
    print()

    print("Stage 0/3: Detecting carrier frequency...")
    from src.processing.frequency_detector import FrequencyDetector
    freq_detector = FrequencyDetector(sample_rate=sample_rate)
    carrier_freq = freq_detector.detect(audio)
    print(f"  Detected carrier: {carrier_freq:.1f} Hz")

    print("Stage 1/3: Signal processing (bandpass + envelope)...")
    processor = SignalProcessor(sample_rate=sample_rate, center_frequency=carrier_freq)
    envelope = processor.process(audio)

    print("Stage 2/3: Morse detection (dits, dahs, gaps)...")
    detector = MorseDetector(sample_rate=sample_rate)
    morse_string = detector.detect(envelope)

    print("Stage 3/3: Decoding (Morse -> text)...")
    decoder = MorseDecoder()
    decoded = decoder.decode(morse_string)
    print()

    print("-" * 50)
    print(f"Estimated WPM:  {detector.estimated_wpm:.1f}")
    print(f"Morse string:   {morse_string}")
    print(f"Decoded text:   {decoded}")
    print("-" * 50)


def decode_generated(text: str, wpm: int) -> None:
    """Generate a Morse audio signal from text, then decode it."""
    from scripts.generate_morse_audio import generate_morse_audio

    audio_path = "audio_samples/run_decoder_demo.wav"
    print(f"Generating test signal: '{text}' at {wpm} WPM")
    print()
    generate_morse_audio(text, audio_path, wpm=wpm)
    print()

    decode_audio_file(audio_path)

    print()
    if decoded_matches(text, audio_path):
        print("Round-trip check: PASS (decoded text matches input)")
    else:
        print("Round-trip check: decoded text differs from input (see above)")


def decoded_matches(original: str, audio_path: str) -> bool:
    """Helper: decode the audio and compare to the original text."""
    audio, sample_rate = load_full_audio(audio_path)
    processor = SignalProcessor(sample_rate=sample_rate)
    envelope = processor.process(audio)
    detector = MorseDetector(sample_rate=sample_rate)
    morse_string = detector.detect(envelope)
    decoder = MorseDecoder()
    return decoder.decode(morse_string) == original.upper()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Morse Code Decoder - run the full pipeline on an audio file."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        help="Path to an audio file (.wav or .mp3) to decode.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate and decode a default 'HELLO WORLD' test signal.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Generate and decode a custom text as a test signal.",
    )
    parser.add_argument(
        "--wpm",
        type=int,
        default=20,
        help="Words per minute for generated test signals (default 20).",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Morse Code Decoder")
    print("=" * 50)
    print()

    if args.text is not None:
        decode_generated(args.text, args.wpm)
    elif args.demo:
        decode_generated("HELLO WORLD", args.wpm)
    elif args.audio_file is not None:
        decode_audio_file(args.audio_file)
    else:
        print("No input specified. Choose one of:")
        print("  python -m run_decoder path/to/audio.wav")
        print("  python -m run_decoder --demo")
        print('  python -m run_decoder --text "HELLO WORLD" --wpm 20')
        sys.exit(1)


if __name__ == "__main__":
    main()
