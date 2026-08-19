"""
Generate Morse code audio from text and save as a WAV file.

Uses standard PARIS timing:
- 1 unit = 60 / (50 * WPM) seconds
- dit = 1 unit
- dah = 3 units
- intra-letter gap = 1 unit
- inter-letter gap = 3 units
- inter-word gap = 7 units

A short cosine ramp at each tone edge avoids audible clicks.
"""

import os
import numpy as np
from scipy.io import wavfile


MORSE_TABLE = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',   'E': '.',
    'F': '..-.',  'G': '--.',   'H': '....',  'I': '..',    'J': '.---',
    'K': '-.-',   'L': '.-..',  'M': '--',    'N': '-.',    'O': '---',
    'P': '.--.',  'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',  'Y': '-.--',
    'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', '/': '-..-.', '=': '-...-',
}


def text_to_morse(text: str) -> str:
    """Convert text to a readable Morse string ('.', '-', ' ' between letters, '/' between words)."""
    text = text.upper()
    words = text.split()
    encoded_words = []
    for word in words:
        letters = [MORSE_TABLE[ch] for ch in word if ch in MORSE_TABLE]
        encoded_words.append(' '.join(letters))
    return ' / '.join(encoded_words)


def _generate_tone(num_samples: int, frequency: float, sample_rate: int, amplitude: float) -> np.ndarray:
    """Generate a sine-wave burst with short cosine ramps at start and end."""
    t = np.arange(num_samples) / sample_rate
    wave = amplitude * np.sin(2 * np.pi * frequency * t)

    ramp_samples = min(int(0.005 * sample_rate), num_samples // 4)
    if ramp_samples > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))
        wave[:ramp_samples] *= ramp
        wave[-ramp_samples:] *= ramp[::-1]

    return wave.astype(np.float32)


def generate_morse_audio(
    text: str,
    output_path: str = "audio_samples/morse_hello_world.wav",
    wpm: int = 20,
    frequency: float = 700.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> None:
    """Generate Morse-code audio from text and save it as a 16-bit PCM WAV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    unit_seconds = 60.0 / (50.0 * wpm)
    unit_samples = int(sample_rate * unit_seconds)

    dit_tone = _generate_tone(unit_samples, frequency, sample_rate, amplitude)
    dah_tone = _generate_tone(unit_samples * 3, frequency, sample_rate, amplitude)
    intra_letter_gap = np.zeros(unit_samples, dtype=np.float32)
    inter_letter_gap = np.zeros(unit_samples * 3, dtype=np.float32)
    inter_word_gap = np.zeros(unit_samples * 7, dtype=np.float32)

    text_clean = text.upper()
    words = text_clean.split()

    pieces = []
    for w_idx, word in enumerate(words):
        if w_idx > 0:
            pieces.append(inter_word_gap)
        letters = [ch for ch in word if ch in MORSE_TABLE]
        for l_idx, letter in enumerate(letters):
            if l_idx > 0:
                pieces.append(inter_letter_gap)
            code = MORSE_TABLE[letter]
            for e_idx, element in enumerate(code):
                if e_idx > 0:
                    pieces.append(intra_letter_gap)
                if element == '.':
                    pieces.append(dit_tone)
                else:
                    pieces.append(dah_tone)

    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    pcm = (audio * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, pcm)

    duration = len(audio) / sample_rate
    print(f"Text:      {text}")
    print(f"Morse:     {text_to_morse(text)}")
    print(f"WPM:       {wpm}")
    print(f"Frequency: {frequency} Hz")
    print(f"Duration:  {duration:.2f} s")
    print(f"Saved to:  {output_path}")


if __name__ == "__main__":
    generate_morse_audio("Hi shervin. How are you?", "audio_samples/morse_hello_world.wav", wpm=20)

