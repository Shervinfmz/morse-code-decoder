"""
Build Morse_Decoder_AllInOne.ipynb from the src package.

Why this script exists
----------------------
The notebook and the package used to be two hand-maintained copies of the
same code. They drifted: by July the notebook was missing the language
model, the adaptive decoder, the unified pipeline and both GUIs.

So the notebook is now a BUILD ARTEFACT. The package under src/ is the only
source of truth. Run this script and the notebook is rebuilt from it, which
makes drift impossible.

Usage (from the project root):

    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace Morse_Decoder_AllInOne.ipynb

The first command writes the notebook. The second runs every cell so the
saved file contains real output for a reader who never executes it.

What the script changes when it copies a module in
--------------------------------------------------
1. `from src...` / `from scripts...` imports are dropped. In one notebook
   every class already shares a single namespace.
2. `matplotlib.use("TkAgg")` is dropped. The GUI classes build their canvases
   with FigureCanvasTkAgg directly and never touch pyplot, so they still work,
   while inline plotting in the notebook keeps working too.
3. Trailing `def main()` / `if __name__ == "__main__"` blocks are cut. The
   notebook has its own launcher cells.
4. app_view.py's colour constants are renamed with a VIEW_ prefix, because
   GREEN and CURSOR would otherwise collide with app_dashboard.py's palette
   and silently repaint the first GUI.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "Morse_Decoder_AllInOne.ipynb")

# Colour constants in app_view.py that clash with app_dashboard.py.
VIEW_RENAMES = ["GREEN", "DGREEN", "MGREEN", "CURSOR", "BLACK"]


def load_module(relpath, renames=None):
    """Read a source file and adapt it for life inside one notebook."""
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as handle:
        source = handle.read().replace("\r\n", "\n")

    # cut the command-line entry point, if any
    for marker in ("\ndef main():", "\nif __name__ =="):
        index = source.find(marker)
        if index != -1:
            source = source[:index]

    kept = []
    in_continuation = False
    for line in source.split("\n"):
        if in_continuation:
            # tail of a parenthesised import that was already dropped
            if ")" in line:
                in_continuation = False
            continue
        removable = (re.match(r"\s*from (src|scripts)[\w.]*\s+import", line)
                     or re.match(r"\s*import (src|scripts)\b", line)
                     or "matplotlib.use(" in line)
        if removable:
            # An indented import may be the only body of a try/except block.
            # Deleting it outright would leave an empty block, so leave a pass.
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                kept.append(" " * indent + "pass  # import dropped: one-notebook build")
            if line.count("(") > line.count(")"):
                in_continuation = True
            continue
        kept.append(line)
    source = "\n".join(kept).strip("\n")

    for name in (renames or []):
        source = re.sub(rf"\b{name}\b", f"VIEW_{name}", source)

    return source


def as_source(text):
    """Split into notebook source lines. Every line but the last keeps its \\n."""
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": as_source(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": as_source(text)}


def module_cell(relpath, renames=None):
    return code(load_module(relpath, renames))


# Each entry is (heading the banner goes in front of, banner text).
PART_DIVIDERS = [
    ("Morse table", """
# PART 1 - Audio input (Shervin)

From a file or a live device, to a clean on/off envelope ready for decoding.

| Block | Sections | In one sentence |
|---|---|---|
| The Morse alphabet | Morse table | One table, and the reverse built automatically from it |
| Audio input | AudioSource, FileSource, ArraySource, MicrophoneSource, PacedFileSource | One abstract rule, four implementations, and nothing downstream knows which it has |
| Making test audio | Morse audio generator | Text to Morse to sound, so the correct answer is always known |
| Finding the tone | FrequencyDetector | The FFT says which pitch, so the filter knows where to listen |
| Cleaning the signal | SignalProcessor | Bandpass filter, then envelope, then smoothing |
"""),

    ("MorseDetector", """
# PART 2 - Signal to text (Hanna)

From an envelope to readable characters, at any speed between 10 and 40 WPM.

| Block | Sections | In one sentence |
|---|---|---|
| Envelope to elements | MorseDetector | Threshold, run lengths, and finding the dit by the largest gap in the sorted durations |
| Elements to text | MorseDecoder | Table lookup; a pattern that matches nothing becomes `?` |
| Any speed, 10 to 40 WPM | Adaptive decoding | Two passes: measure the speed first, then decode with a window scaled to it |

Hanna also presents the verification and the limits, in Part 4.
"""),

    ("Vocabulary for the language model", """
# PART 3 - Model, streaming and display (Mayank)

Correcting what the decoder got wrong, decoding audio that has not finished
arriving, and putting it on screen.

| Block | Sections | In one sentence |
|---|---|---|
| The language model | Vocabulary, LanguageModel | Character n-grams (n = 3) plus a weighted edit distance over about 4917 words |
| One pipeline | The unified pipeline | Every application and every test calls the same function |
| Timing | Element and character timing | When each element and each character starts and ends |
| Live streaming | StreamDecoder | A rolling window re-decoded several times a second, and why half a second of latency is unavoidable |
| Visualisation | Visualizer | The project's second abstract base class, with three views |
"""),

    ("Run the pipeline end to end", """
# PART 4 - Results and demonstrations

Everything above is definitions. Everything below actually runs.

| Section | Presented by |
|---|---|
| Run the pipeline end to end | Hanna |
| The five pipeline views | Mayank |
| Accuracy across the speed range | Hanna |
| What the language model does | Mayank |
| Application 1 - MorseViewApp | Mayank |
| Decoding a live stream | Mayank |
| Application 2 - DashboardApp | Mayank |
| Summary, requirements and limits | Hanna |
"""),
]


# --------------------------------------------------------------- notebook

def renumber(cells):
    """Rewrite every '## N. Title' heading so the numbers run 0, 1, 2, ...

    A new section can be written with the placeholder '## X. Title' and it is
    numbered automatically, so sections can be added or removed anywhere
    without hand-editing every heading that follows.
    """
    counter = 0
    pattern = re.compile(r"^## (?:\d+|X)\. ", re.MULTILINE)
    for cell in cells:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        if not pattern.search(text):
            continue

        def replace(match):
            nonlocal counter
            new = f"## {counter}. "
            counter += 1
            return new

        cell["source"] = as_source(pattern.sub(replace, text))


def insert_part_dividers(cells):
    """
    Put a PART banner in front of the section that starts each block.

    The notebook is already in dependency order - a class cannot appear before
    the things it uses - and that order happens to group the work by person.
    These banners make the grouping visible without moving any code, so the
    section numbers in the exam notes stay correct.

    Banners use a single '#' so `renumber` leaves them alone; it only touches
    '## N.' headings.
    """
    pattern = re.compile(r"^## (?:\d+|X)\. (.+)$")
    for title, banner in reversed(PART_DIVIDERS):
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "markdown":
                continue
            first = "".join(cell["source"]).lstrip().split("\n", 1)[0]
            match = pattern.match(first)
            if match and match.group(1).strip().startswith(title):
                cells.insert(index, markdown(banner))
                break
    return cells


def build():
    cells = []

    cells.append(markdown("""
# Morse Code Decoder - All-in-One Notebook

Object-Oriented Programming, TH Koln (Prof. Dr. Peter Kern).

This notebook contains the complete decoder in one file: audio input, carrier
detection, filtering, element detection, Morse-to-text, the statistical
language model, and both graphical applications.

**This file is generated.** The package under `src/` is the source of truth.
Rebuild with `python scripts/build_notebook.py`. Do not edit the code cells by
hand - the next build overwrites them.

### The signal chain

    audio file -> FrequencyDetector -> SignalProcessor -> MorseDetector
               -> MorseDecoder -> LanguageModel -> text

### What this project does not do

The decoder is signal processing plus a statistical language model. It is not
a neural network, and it reports no confidence score, because a threshold and
a lookup table produce a symbol, not a probability. It decodes recorded
audio files; it is not a streaming receiver, so it reports no latency figure
either.
"""))

    cells.append(markdown("## 0. Setup"))
    cells.append(code("""
# Set to True and re-run the launcher cells at the bottom to open the GUIs.
# Left False so the whole notebook can run start to finish unattended.
LAUNCH_GUI = False

import os
import io
import sys
import math
import re
import wave
import contextlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert, spectrogram

os.makedirs("audio_samples", exist_ok=True)
print("Python", sys.version.split()[0], "| numpy", np.__version__)
"""))

    sections = [
        ("## 1. Morse table",
         "The single source of truth for character-to-pattern mapping, used by "
         "both the decoder and the audio generator.",
         "src/processing/morse_table.py", None),

        ("## 2. AudioSource - the input abstraction",
         "An abstract base class. Both input types implement the same four "
         "methods, so nothing downstream cares where samples come from.",
         "src/audio/audio_source.py", None),

        ("## 3. FileSource",
         "Reads .wav and .mp3 files and hands out fixed-size chunks.",
         "src/audio/file_source.py", None),

        ("## 4. ArraySource",
         "The second implementation of the same interface, serving samples "
         "that are already in memory. The test suite feeds a generated signal "
         "straight into the pipeline through it, with no temporary file on "
         "disk. Nothing downstream can tell the two sources apart, which is "
         "what the abstract base class is for.",
         "src/audio/array_source.py", None),

        ("## X. MicrophoneSource",
         "The live-input implementation of the same interface. It captures "
         "from any recording device the operating system exposes, which "
         "includes a virtual audio cable - the path that carries audio from a "
         "WebSDR receiver in a browser into this program with no acoustic "
         "step and therefore no added noise.",
         "src/audio/microphone_source.py", None),

        ("## X. PacedFileSource",
         "A file served at the speed the audio actually plays. Ask it for a "
         "chunk before the clock has reached it and it returns nothing, "
         "exactly as a microphone does. It exists so the streaming decoder "
         "can be exercised on a machine with no sound card, and it is the "
         "fourth implementation of AudioSource.",
         "src/audio/paced_file_source.py", None),

        ("## 5. Morse audio generator",
         "Turns text into a clean CW tone. Used to build test material with a "
         "known correct answer, which is what makes the accuracy tests below "
         "meaningful.",
         "scripts/generate_morse_audio.py", None),

        ("## 6. FrequencyDetector",
         "Finds the carrier tone by taking an FFT of the whole recording and "
         "picking the dominant bin. This is what lets the decoder work without "
         "being told the tone frequency in advance.",
         "src/processing/frequency_detector.py", None),

        ("## 7. SignalProcessor",
         "Bandpass filter around the carrier, then a Hilbert-transform "
         "envelope, then smoothing. The envelope is the on/off pattern the "
         "detector reads.",
         "src/processing/signal_processor.py", None),

        ("## 8. MorseDetector",
         "Thresholds the envelope, measures run lengths, estimates the dit "
         "length from the largest gap in the sorted durations, and classifies "
         "every element and gap. Speed in words per minute falls out of the "
         "measured dit length.",
         "src/processing/morse_detector.py", None),

        ("## 9. MorseDecoder",
         "Pattern to character, via the inverse table.",
         "src/processing/morse_decoder.py", None),

        ("## 10. Adaptive decoding",
         "A first pass estimates the dit length, which sets the smoothing "
         "window for a second pass. This is what makes one decoder work across "
         "10 to 40 words per minute instead of needing a hand-tuned constant.",
         "src/processing/adaptive_decoder.py", None),

        ("## 11. Vocabulary for the language model",
         "About 4900 common English words plus amateur-radio terms. Frozen in "
         "the repository, so nothing is downloaded at run time.",
         "src/processing/language_words_data.py", None),

        ("## 12. LanguageModel",
         "The statistical layer. Word-level correction by weighted edit "
         "distance, where substitutions that are common Morse confusions cost "
         "less, plus a character n-gram model that scores how English-like a "
         "candidate is. It is deliberately conservative: it never rewrites a "
         "word that is already valid.",
         "src/processing/language_model.py", None),

        ("## 13. The unified pipeline",
         "One function that every GUI, script and cell calls. Because there is "
         "a single entry point, the adaptive smoothing and the language model "
         "apply everywhere automatically.",
         "src/processing/decode_pipeline.py", None),

        ("## X. Element and character timing",
         "MorseDetector answers what was sent; this module answers when. It "
         "reports the start and end of every dit and dah, and of every "
         "decoded character. The file dashboard uses it to reveal text in "
         "step with playback, and the streaming decoder uses it to work out "
         "which characters are new.",
         "src/processing/timing.py", None),

        ("## X. StreamDecoder - decoding a live stream",
         "The file decoder waits for a complete recording; a live stream "
         "never ends. This class keeps a rolling window of the most recent "
         "audio, re-decodes it a few times a second, and emits only the "
         "characters that have settled. It is what makes real-time decoding "
         "possible without changing any of the signal processing above.",
         "src/processing/stream_decoder.py", None),

        ("## 14. Visualizer",
         "Oscilloscope, FFT and waterfall views behind one abstract interface.",
         "src/processing/visualizer.py", None),
    ]

    for heading, blurb, path, renames in sections:
        cells.append(markdown(f"{heading}\n\n{blurb}"))
        cells.append(module_cell(path, renames))

    # ------------------------------------------------------------ demos
    cells.append(markdown("""
## 15. Run the pipeline end to end

Generate audio from a known sentence, decode it, and compare. Because the
input text is known, the output can be checked rather than admired.
"""))
    cells.append(code("""
DEMO_TEXT = "MORSE DECODER WORKING"
DEMO_PATH = "audio_samples/notebook_demo.wav"

generate_morse_audio(DEMO_TEXT, DEMO_PATH, wpm=20, frequency=700)

source = FileSource(filepath=DEMO_PATH, chunk_size=1024)
source.start()
chunks = []
while not source.is_finished:
    chunk = source.read_chunk()
    if len(chunk) > 0:
        chunks.append(chunk)
source.stop()
audio = np.concatenate(chunks)
sample_rate = source.sample_rate

result = decode_pipeline(audio, sample_rate)

print("carrier detected :", round(result.carrier_hz), "Hz")
print("speed estimated  :", round(result.estimated_wpm, 1), "WPM")
print("smoothing chosen :", round(result.smoothing_ms, 1), "ms")
print()
print("raw decode       :", result.raw_text)
print("after correction :", result.corrected_text)
print("expected         :", DEMO_TEXT)
print()
print("match:", result.corrected_text.strip() == DEMO_TEXT)
"""))

    cells.append(markdown("""
## 16. The five pipeline views as static figures

The same five views the live dashboard scrolls: waveform, envelope with the
detected elements shaded, spectrum with the detected carrier marked, waterfall,
and the element timeline. Static here so the figures survive in the saved
notebook.
"""))
    cells.append(code("""
def element_segments(envelope, detector, sample_rate):
    \"\"\"Every on-period in the envelope, as (start_s, end_s, kind).

    Read straight from the detector output, so elements that never resolve
    into a character are still present.
    \"\"\"
    dit = getattr(detector, "dit_samples", None)
    if not dit:
        return []
    binary = (envelope > detector.threshold_ratio * np.max(envelope)).astype(np.int8)
    dah_threshold = getattr(detector, "dah_threshold", 2.0)
    segments, index, total = [], 0, len(binary)
    while index < total:
        if binary[index] == 1:
            start = index
            while index < total and binary[index] == 1:
                index += 1
            kind = "dit" if (index - start) < dah_threshold * dit else "dah"
            segments.append((start / sample_rate, index / sample_rate, kind))
        else:
            index += 1
    return segments


segments = element_segments(result.envelope, result.detector, sample_rate)
times = np.arange(len(audio)) / sample_rate
view = (times >= 0) & (times <= min(6.0, times[-1]))

figure, axes = plt.subplots(5, 1, figsize=(13, 11))
figure.patch.set_facecolor("#0d151e")

axes[0].plot(times[view], audio[view], color="#31e07a", linewidth=0.6)
axes[0].set_title("1. Time-domain signal")

envelope_norm = result.envelope / (np.max(result.envelope) + 1e-9)
axes[1].plot(times[view], envelope_norm[view], color="#f0b429", linewidth=0.9)
for start, end, kind in segments:
    if start > 6.0:
        break
    axes[1].axvspan(start, end, color="#22d3ee" if kind == "dit" else "#ff5fbf", alpha=0.35)
axes[1].set_title("2. Envelope with detected elements (cyan = dit, magenta = dah)")

spectrum = np.abs(np.fft.rfft(audio))
frequencies = np.fft.rfftfreq(len(audio), 1 / sample_rate)
band = frequencies <= 2000
axes[2].plot(frequencies[band], spectrum[band], color="#b388ff", linewidth=0.8)
axes[2].axvline(result.carrier_hz, color="#ff4d4d", linestyle="--")
axes[2].set_title(f"3. Spectrum, carrier detected at {result.carrier_hz:.0f} Hz")

freq_axis, time_axis, values = spectrogram(audio, fs=sample_rate, nperseg=1024, noverlap=512)
band = freq_axis <= 2000
axes[3].imshow(10 * np.log10(values[band, :] + 1e-12), aspect="auto", origin="lower",
               cmap="viridis",
               extent=[time_axis[0], time_axis[-1], freq_axis[band][0], freq_axis[band][-1]])
axes[3].set_title("4. Waterfall spectrogram")
axes[3].set_ylabel("Hz")

for start, end, kind in segments:
    if kind == "dit":
        axes[4].plot([(start + end) / 2], [0.5], marker="o", markersize=5, color="#22d3ee")
    else:
        axes[4].plot([start, end], [0.5, 0.5], linewidth=5, color="#ff5fbf")
axes[4].set_ylim(0, 1)
axes[4].set_yticks([])
axes[4].set_xlim(0, min(6.0, times[-1]))
axes[4].set_title(f"5. Element timeline - {len(segments)} elements detected")

for axis in axes:
    axis.set_facecolor("#05090d")
    axis.tick_params(colors="#7d93a8", labelsize=7)
    axis.title.set_color("#22d3ee")
    axis.title.set_size(9)

plt.tight_layout()
plt.show()

dits = sum(1 for _, _, kind in segments if kind == "dit")
dahs = len(segments) - dits
print(f"dits {dits} | dahs {dahs} | elements {len(segments)}")
"""))

    cells.append(markdown("""
## 17. Accuracy across the speed range

One decoder, no per-speed tuning. Every row generates fresh audio at that
speed and decodes it back.
"""))
    cells.append(code("""
PHRASE = "THE QUICK BROWN FOX"
rows = []
for wpm in range(10, 41, 5):
    path = f"audio_samples/_sweep_{wpm}.wav"
    with contextlib.redirect_stdout(io.StringIO()):   # the generator is chatty
        generate_morse_audio(PHRASE, path, wpm=wpm, frequency=700)

    source = FileSource(filepath=path, chunk_size=1024)
    source.start()
    pieces = []
    while not source.is_finished:
        chunk = source.read_chunk()
        if len(chunk) > 0:
            pieces.append(chunk)
    source.stop()

    outcome = decode_pipeline(np.concatenate(pieces), source.sample_rate)
    rows.append((wpm, round(outcome.estimated_wpm, 1), outcome.corrected_text.strip(),
                 outcome.corrected_text.strip() == PHRASE))

print()
print(f"{'set':>5} {'measured':>9}  {'ok':<5} decoded")
print("-" * 60)
for target, measured, text, correct in rows:
    print(f"{target:>5} {measured:>9}  {str(correct):<5} {text}")
print("-" * 60)
print(f"{sum(1 for r in rows if r[3])}/{len(rows)} correct")

for wpm in range(10, 41, 5):        # tidy up the scratch audio
    temp = f"audio_samples/_sweep_{wpm}.wav"
    if os.path.exists(temp):
        os.remove(temp)
"""))

    cells.append(markdown("""
## 18. What the language model does

The correction layer is the one statistical component. It is trained on text,
never on audio. Below it is given text with the kind of errors a marginal
signal produces.
"""))
    cells.append(code("""
model = get_language_model()
print("vocabulary size:", model.vocabulary_size)
print()
for damaged in ["M0RSE DEC0DER", "THE QU1CK BR0WN F0X", "CQ CQ DE DL1ABC", "HELLO WORLD"]:
    print(f"{damaged:<22} ->  {model.correct(damaged)}")
print()
print("Note: the model is conservative by design. It leaves valid words alone,")
print("and it can still be wrong on names and unusual words - see the word-level")
print("substitutions above before trusting it on unfamiliar text.")
"""))

    cells.append(markdown("""
## 19. Application 1 - MorseViewApp

The main decoding window: load a file, decode, and watch the text appear in
step with playback.

This cell opens a separate window and blocks the notebook kernel until that
window is closed. Set `LAUNCH_GUI = True` in the setup cell first.
"""))
    cells.append(markdown("### Source"))
    cells.append(module_cell("src/app_view.py", VIEW_RENAMES))
    cells.append(code("""
if LAUNCH_GUI:
    root = tk.Tk()
    MorseViewApp(root)
    root.mainloop()
else:
    print("LAUNCH_GUI is False - set it to True in the setup cell to open this window.")
"""))

    cells.append(markdown("""
## X. Decoding a live stream

Everything above decodes a complete recording. This section decodes audio
*while it arrives*, which is what a live input requires.

`PacedFileSource` releases a file at the speed it actually plays, so it stands
in for a microphone or a virtual audio cable: same interface, same block size,
same "nothing is ready yet" behaviour. The decoder has no idea it is not
listening to a radio.

Watch the timestamps in the output: text appears while the stream is still
running, not at the end.
"""))
    cells.append(code("""
from src.audio.paced_file_source import PacedFileSource
from src.processing.stream_decoder import StreamDecoder

STREAM_TEXT = "CQ DE MORSE DECODER K"
stream_path = "audio_samples/notebook_stream.wav"
generate_morse_audio(STREAM_TEXT, output_path=stream_path, wpm=20, sample_rate=44100)

# 6x real time so the notebook does not take a minute to run.
source = PacedFileSource(stream_path, chunk_size=1024, speed=6.0)
decoder = StreamDecoder(sample_rate=44100)

source.start()
updates = []
while not source.is_finished:
    chunk = source.read_chunk()
    if len(chunk):
        decoder.feed(chunk)
        new_text = decoder.poll()
        if new_text:
            updates.append((decoder.stream_seconds, new_text))
# Latency while the stream was actually running. Measure it before the
# artificial silence below, which would otherwise inflate the last reading.
typical_latency = decoder.character_latency_seconds

# The operator stops sending. The silence that follows is what proves the
# last character has ended, so feed a little of it before the final decode.
decoder.feed(np.zeros(int(1.5 * 44100), dtype=np.float32))
final = decoder.poll(force=True)
if final:
    updates.append((decoder.stream_seconds, final))
source.stop()

print("Text as it arrived, with the position in the stream:")
for when, piece in updates:
    print(f"  [{when:6.2f} s]  {piece!r}")

print()
print("Transmitted:", STREAM_TEXT)
print("Decoded:    ", decoder.text.strip())
print("Corrected:  ", decoder.corrected_text.strip())
print(f"Carrier:     {decoder.carrier_hz:.0f} Hz")
print(f"Speed:       {decoder.estimated_wpm:.1f} WPM")
print(f"Latency:     {typical_latency * 1000:.0f} ms behind the live edge")
print()
print("The decoder never saw the whole recording: it worked from a rolling")
print(f"{decoder.window_seconds:.0f} second window and committed a character only once the")
print("gap after it proved the character had ended. That wait is the latency.")
"""))

    cells.append(markdown("""
## X. Application 2 - DashboardApp, both modes in one window

The dashboard is one window with two sources of audio.

**Audio File** decodes the whole recording first, then replays it with the
text revealed in step with the audio. Because the entire signal is available
at once, this is the more accurate mode.

**Live Input** decodes audio as it arrives, from a microphone or from a
virtual audio cable carrying a WebSDR receiver. `Stream a File` runs that same
live path against a recording, at real-time speed, so it can be demonstrated
with no radio attached.

Both modes drive the same seven panels and the same decoding code. `app_stream`
is a two-line launcher that opens this window with Live Input pre-selected.
"""))
    cells.append(markdown("### Source"))
    cells.append(module_cell("src/app_dashboard.py"))
    cells.append(code("""
if LAUNCH_GUI:
    root = tk.Tk()
    DashboardApp(root, mode="file")   # or mode="live"
    root.mainloop()
else:
    print("LAUNCH_GUI is False - set it to True in the setup cell to open this window.")
"""))

    cells.append(markdown("""
## 21. Summary

| Requirement | Where it is met |
|---|---|
| Abstract base class with several implementations | `AudioSource`, with `FileSource`, `ArraySource`, `MicrophoneSource` and `PacedFileSource`; also `Visualizer` with three views |
| Signal processing chain | `FrequencyDetector`, `SignalProcessor`, `MorseDetector`, `MorseDecoder` |
| Works without being told the tone or the speed | carrier from the FFT, speed from the measured dit length |
| Intelligent interpretation layer | `LanguageModel`, word-level correction plus character n-grams |
| Visualisation | `Visualizer` with three views, plus the live dashboard |
| Live audio streaming, decoded in real time | `MicrophoneSource` -> `StreamDecoder` -> `DashboardApp` in live mode |
| Graphical application | `MorseViewApp`, and `DashboardApp` with a file mode and a live mode |

Honest limits, stated once more. No confidence score anywhere, because the
decoder compares a duration against a threshold and looks the pattern up in a
table, which produces a symbol rather than a probability. No latency figure in
file mode, because a complete recording is decoded and then replayed, so there
is nothing real to measure; live mode does report one, because there the wait
is real. One carrier at a time: the strongest peak in the FFT wins, so two
overlapping CW signals cannot be separated.
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    insert_part_dividers(cells)
    renumber(cells)

    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")

    code_cells = sum(1 for cell in cells if cell["cell_type"] == "code")
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)}")
    print(f"{len(cells)} cells ({code_cells} code)")


if __name__ == "__main__":
    build()
