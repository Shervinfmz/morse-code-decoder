# Architecture

## Overview
The Morse Code Decoder follows a modular object-oriented design. Each module has a single responsibility: audio input, signal processing, Morse detection, decoding, language correction, visualization, and orchestration.

## Class Diagram

```mermaid
classDiagram
    class AudioSource {
        <<abstract>>
        +int sample_rate
        +int chunk_size
        +start() void
        +stop() void
        +read_chunk() ndarray
    }
    class MicrophoneSource {
        +int device_id
    }
    class FileSource {
        +str filepath
    }
    AudioSource <|-- MicrophoneSource
    AudioSource <|-- FileSource

    class SignalProcessor {
        +bandpass_filter(signal) ndarray
        +envelope_detection(signal) ndarray
        +reduce_noise(signal) ndarray
    }

    class MorseDetector {
        +float current_wpm
        +estimate_wpm(envelope) float
        +detect_elements(envelope) list
    }

    class MorseDecoder {
        +dict morse_table
        +decode(elements) str
    }

    class LanguageModel {
        +train(corpus) void
        +correct(text) str
    }

    class Visualizer {
        <<abstract>>
        +update(data) void
    }
    class OscilloscopeView
    class FFTView
    class WaterfallView
    Visualizer <|-- OscilloscopeView
    Visualizer <|-- FFTView
    Visualizer <|-- WaterfallView

    class DecoderApp {
        +run() void
    }
    DecoderApp --> AudioSource
    DecoderApp --> SignalProcessor
    DecoderApp --> MorseDetector
    DecoderApp --> MorseDecoder
    DecoderApp --> LanguageModel
    DecoderApp --> Visualizer
```

## Module Responsibilities

### AudioSource (abstract)
- Defines the interface for any audio input.
- Subclasses: `MicrophoneSource` (live mic via sounddevice), `FileSource` (mp3/wav playback).
- Outputs raw audio chunks as numpy arrays.

### SignalProcessor
- Applies bandpass filter centered on the CW tone frequency (typically 600–800 Hz).
- Extracts the envelope (amplitude over time) via rectification + low-pass, or Hilbert transform.
- Optional adaptive noise reduction.

### MorseDetector
- Thresholds the envelope to identify on/off periods.
- Estimates words-per-minute (WPM) adaptively.
- Classifies each pulse as dit, dah, element-gap, letter-gap, or word-gap.

### MorseDecoder
- Maps dit/dah patterns to characters using the standard Morse table.
- Handles prosigns (CQ, K, AR, SK).

### LanguageModel
- Character-level N-gram model trained on amateur radio conversations and English text.
- Scores ambiguous decoder outputs and applies probabilistic corrections.

### Visualizer (abstract)
- Common interface for live signal displays.
- Subclasses: `OscilloscopeView` (time domain), `FFTView` (frequency domain), `WaterfallView` (spectrogram).

### DecoderApp
- Top-level orchestrator. Wires audio source, processor, detector, decoder, language model, and visualizer.
- Owns the GUI window that displays decoded text in real-time.

## Data Flow

```
AudioSource --> SignalProcessor --> MorseDetector --> MorseDecoder --> LanguageModel --> UI
                       |
                       +------> Visualizer
```

## Out of Scope
- Multi-language support (English only).
- Direct WebSDR integration (use virtual audio cable workaround).
- Mobile UI.