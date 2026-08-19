# Morse Code Decoder — Progress Snapshot

**Course:** Object-Oriented Programming, TH Köln, Summer Semester 2026
**Supervisor:** Prof. Dr. Peter Kern
**Team:** Mayank Dinesh Mehta + Shervin Faramarzi Babadi + Hanna Leonie Pohl
**Repository:** Academic Git repository (original project repository)

---

## Current Status

- **Core implementation complete and integrated; final project modules include live streaming, adaptive decoding, timing, visualization, and language-model correction.**
- **Core decoder works end-to-end**: text → audio → text round-trips with 100% accuracy on clean signals at multiple WPM speeds
- **First presentation:** 29 June 2026 (Phase 12 milestone)
- **Final presentation:** 20 July 2026

---

## Project Architecture (high-level)

```
AudioSource (abstract)
   ├── FileSource          (reads .wav, .mp3)
   └── MicrophoneSource    (live capture via sounddevice)
          ↓
SignalProcessor            (bandpass filter + envelope detection)
          ↓
MorseDetector              (threshold, classify dits/dahs, adaptive WPM)
          ↓
MorseDecoder               (lookup table → text)
```

Full class diagram lives in `docs/architecture.md` and renders as a Mermaid diagram on GitLab.

---

## Phase-by-Phase Summary

### Phase 1 — Architecture sketch
- Created `docs/architecture.md` with class diagram (Mermaid) and module responsibilities
- Defined data flow: audio → filter → envelope → detect → decode → text
- Established `src/` for product code, `scripts/` for tooling, `tests/` for verification

### Phase 2 — Python environment
- Set up virtual environment with Python 3.11.9 (chose 3.11 over 3.13 for library stability)
- Installed core dependencies: `numpy`, `scipy`, `sounddevice`, `matplotlib`, `pydub`
- Configured `.gitignore` to exclude `.venv/`, `audio_samples/`, `__pycache__/`

### Phase 3 — AudioSource skeleton
- Created abstract base class `AudioSource` using Python's `ABC`
- Defined contract: `start()`, `stop()`, `read_chunk()`, `sample_rate`, `chunk_size`
- Stubs for `MicrophoneSource` and `FileSource` (concrete subclasses)
- Demonstrates clean OOP inheritance from the start

### Phase 4 — FileSource implementation
- Reads `.wav` files via `scipy.io.wavfile`
- Supports `.mp3` via `pydub` (requires ffmpeg)
- Stereo audio averaged to mono
- All formats normalized to float32 in [-1.0, 1.0]
- Exposes `duration_seconds`, `total_samples`, `is_finished` properties
- **Verified:** loads test tone, returns chunks correctly

### Phase 5 — MicrophoneSource implementation
- Uses `sounddevice.InputStream` with callback API
- Thread-safe queue handoff between audio callback thread and main thread
- Drops chunks if consumer can't keep up (prevents blocking the audio thread)
- `list_devices()` static method for hardware enumeration
- **Verified:** captures 3 seconds of live audio, reports RMS and peak amplitude

### Phase 6 — Morse audio generator (testing tool)
- Script: `scripts/generate_morse_audio.py`
- Converts arbitrary text to Morse code audio at any WPM
- Uses standard PARIS timing: 1 unit = 60/(50·WPM) seconds
- Applies cosine ramps at tone edges (no audible clicks)
- Output: 16-bit PCM `.wav` files
- **Purpose:** provides ground-truth signals for testing every downstream stage

### Phase 7 — SignalProcessor
- Class: `src/processing/signal_processor.py`
- Butterworth bandpass filter (default 600–800 Hz, configurable)
- Envelope detection via Hilbert transform (cleaner than rectify + lowpass)
- Moving-average smoothing (20 ms default)
- **Verified:** 3-panel matplotlib plot showing raw → filtered → envelope. Envelope clearly shows Morse on/off pattern as square-wave-like pulses.

### Phase 8 — MorseDetector
- Class: `src/processing/morse_detector.py`
- Thresholds envelope at 30% of peak → binary on/off signal
- Extracts run lengths (durations of consecutive "on" and "off" periods)
- **Adaptive WPM estimation:** sorts on-durations, finds biggest jump to separate dits from dahs. Works at any WPM without assuming the rate in advance.
- Classifies gaps as intra-letter (no separator), inter-letter (space), or inter-word (triple space)
- **Verified:** detects exact expected Morse string for "HELLO WORLD" at 20 WPM

### Phase 9 — MorseDecoder + End-to-End Pipeline
- Shared module `src/processing/morse_table.py` holds the Morse table (single source of truth)
- `MorseDecoder` converts Morse string → text via lookup
- Unknown patterns become `?` placeholder
- **End-to-end test verified at 4 phrases:**
  - "HELLO WORLD" @ 20 WPM → PASS
  - "CQ DE TEST" @ 20 WPM → PASS
  - "THE QUICK BROWN FOX" @ 15 WPM → PASS
  - "ABC 123" @ 25 WPM → PASS

---

## What's Working Right Now

You can run this command and watch the entire pipeline decode generated Morse audio:
```
python -m tests.test_end_to_end
```

Result: 4/4 test cases pass with 100% character accuracy.

---

## What's Still To Do

| Phase | Description | Required for |
|-------|-------------|--------------|
| 10 | _(merged into Phase 9's end-to-end test)_ | — |
| 11 | Visualizer (oscilloscope / FFT / waterfall views) | First presentation |
| 12 | GUI window showing real-time decoded text | First presentation |
| 13 | Adaptive WPM under signal drift | Robustness |
| 14 | Noise handling | Robustness |
| 15 | N-gram language model (the "ML" requirement) | Final presentation |
| 16 | Real-world testing (Morse Code Ninja audio at varying WPM/noise) | Final presentation |
| 17 | Code review prep (type hints, docstrings, refactoring) | Final presentation |
| 18 | Documentation (README, design decisions, architecture update) | Final presentation |
| 19 | Demo screencast | Final presentation |

---

## Folder Structure

```
morse-code-decoder/
├── .venv/                    # virtual environment (gitignored)
├── audio_samples/            # generated test audio (gitignored)
├── docs/
│   └── architecture.md       # class diagram and module responsibilities
├── scripts/
│   ├── generate_test_tone.py
│   └── generate_morse_audio.py
├── src/
│   ├── audio/
│   │   ├── audio_source.py
│   │   ├── file_source.py
│   │   └── microphone_source.py
│   └── processing/
│       ├── morse_table.py
│       ├── signal_processor.py
│       ├── morse_detector.py
│       └── morse_decoder.py
├── tests/
│   ├── test_file_source.py
│   ├── test_microphone_source.py
│   ├── test_signal_processor.py
│   ├── test_morse_detector.py
│   └── test_end_to_end.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions (for teammate to get running)

1. Clone the repo:
   ```
   git clone original academic repository
   cd morse-code-decoder
   ```

2. Create the virtual environment (requires Python 3.11.9):
   ```
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the end-to-end test:
   ```
   python -m tests.test_end_to_end
   ```

   Expected: `4/4 test cases passed`.

**Important:** always run scripts from project root using `python -m <package>.<module>`. Do not use VS Code's Run button or `python path/to/file.py` — both break the import paths.

---

## Known Tech Debt (to clean up in Phase 17)

- Morse table is duplicated between `scripts/generate_morse_audio.py` and `src/processing/morse_table.py`. Generator should import from the shared module.
- `pydub` runtime warning appears on every run because it's imported at the top of `file_source.py` instead of lazily inside the `.mp3` branch.
- WPM estimation has a small systematic underestimate (~5-7%) due to envelope smoothing stretching the "on" periods at the edges.
- No formal unit tests (`pytest`). Current "tests" are verification scripts. Pytest setup can come in cleanup phase.

---

## Key Design Decisions (worth defending in the presentation)

1. **Abstract AudioSource with file and microphone subclasses** — the rest of the pipeline doesn't care where audio comes from. Adding a new input type means adding a new subclass; nothing downstream changes. Textbook OOP.

2. **Generator built alongside decoder** — gives us ground-truth test signals where we know the exact expected output. Without this, we couldn't tell if a decoder failure was the decoder's fault or noise in the input.

3. **Hilbert transform for envelope detection** — single-step, mathematically clean, avoids the parameter tuning of rectify-and-lowpass approaches.

4. **Adaptive dit-length estimation** — we don't assume WPM in advance. The algorithm finds the natural gap between dit-length and dah-length on-periods and adapts to whatever signal it receives. This is what makes "Variable Speed Detection" actually work.

5. **Shared Morse table module** — both encoder and decoder will import from one source of truth (currently only decoder does; generator refactor pending).

---

## How to Talk About This to Your Teammate

Three things they need to know:

1. **The architecture is built and locked.** They should not redesign anything that's already done. They should plug into the existing classes.

2. **The core works.** Right now we have a functional decoder. Everything else (visualizer, GUI, robustness, language model) is additive.

3. **Their next contribution.** Pick one of Phase 11 (Visualizer) or Phase 12 (GUI). I'd suggest the **Visualizer** first — it's more isolated and doesn't depend on GUI choices. The visualizer will plug into the existing `Visualizer` abstract class shown in `docs/architecture.md`.

---

## Project Timeline

- ✅ **Week 1 (May 25-31):** Architecture + audio I/O (Phases 1-5)
- ✅ **Week 2 (Jun 1-7):** Signal processing + Morse pipeline (Phases 6-9)
- 🟡 **Week 3-4 (Jun 8-21):** Visualizer + GUI (Phases 11-12)
- 🟡 **Week 5 (Jun 22-28):** Robustness + presentation prep (Phases 13-14)
- 🎯 **June 29:** First presentation
- 🟡 **Week 6-7 (Jun 30 - Jul 12):** Language model + real-world testing (Phases 15-16)
- 🟡 **Week 8 (Jul 13-19):** Polish, docs, screencast (Phases 17-19)
- 🎯 **July 20:** Final presentation
