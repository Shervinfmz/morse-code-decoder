"""
Morse Code Decoder - "Morse View" style single-window GUI (file input).

Layout inspired by classic Morse decoder tools (e.g. Mitov Morse View):
  - Input source     : load an audio file (.wav / .mp3)
  - Decoder controller: Play + Decode / Stop, Clear
  - Decoder settings  : threshold, bandpass low/high, smoothing
                        (AUTOMATIC by default; optional overrides)
  - Stream view       : green-on-black on/off waveform with a moving playback
                        cursor that tracks the audio position
  - Decoder output    : RAW decoded text (revealed in sync with playback),
                        CORRECTED text (language model), and the Morse string

Decoding uses the unified decode_pipeline: automatic carrier detection,
two-pass adaptive smoothing (so 10-40 WPM all decode well), and the N-gram
language model for error correction. The GUI shows both the raw decode and the
corrected text side by side.

Flow:
  1. Load a file   -> the waveform is shown (no playback yet).
  2. Play + Decode -> audio plays; the decoded text appears gradually in time
                      with the sound; a vertical cursor sweeps the waveform;
                      the corrected text and Morse fill in.
  3. Stop          -> freezes where it is.

Run from the project root:
    python -m src.app_view
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

from src.audio.file_source import FileSource
from src.processing.decode_pipeline import decode_pipeline
from src.processing.morse_table import INVERSE_MORSE_TABLE

GREEN = "#00FF00"
DGREEN = "#1a4d1a"
MGREEN = "#00AA00"
CURSOR = "#FF4444"
BLACK = "#000000"


class MorseViewApp:
    """Single-window file decoder: settings, stream view w/ cursor, sync reveal, correction."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Morse Code Decoder - Morse View")
        self.root.geometry("1150x900")

        self.sample_rate = 44100
        self.audio = None

        # decode results (computed on Play / load)
        self.full_decoded = ""       # raw decoded text
        self.full_corrected = ""     # language-model corrected text
        self.full_morse = ""
        self.char_times = []
        self.duration_s = 0.0
        self.carrier = 0.0
        self.wpm = 0.0
        self.smoothing_ms = 0.0
        self.was_corrected = False
        self.last_envelope = None

        # playback state
        self.is_playing = False
        self.play_start_ms = None
        self._play_id = None
        self._cursor_line = None
        self._stream_ax = None
        self._resume_pos_s = 0.0     # where to resume playback from (seconds)
        self._decoded_once = False   # whether the current audio is already decoded

        self._build_ui()

    # ---------- UI ----------

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        src = ttk.LabelFrame(top, text="Input source", padding=8)
        src.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        ttk.Button(src, text="Load audio file...", command=self.on_load_file).pack(fill=tk.X, pady=2)
        self.file_label = ttk.Label(src, text="(no file)", width=28)
        self.file_label.pack(anchor=tk.W, pady=2)

        ctrl = ttk.LabelFrame(top, text="Decoder controller", padding=8)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.play_button = ttk.Button(ctrl, text="Play + Decode", command=self.on_play, state=tk.DISABLED)
        self.play_button.pack(fill=tk.X, pady=2)
        self.stop_button = ttk.Button(ctrl, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Clear", command=self.on_clear).pack(fill=tk.X, pady=2)
        self.status_label = ttk.Label(ctrl, text="Load a file, then Play + Decode.", width=34, wraplength=240)
        self.status_label.pack(anchor=tk.W, pady=6)

        setp = ttk.LabelFrame(top, text="Decoder settings (auto by default)", padding=8)
        setp.pack(side=tk.LEFT, fill=tk.Y)
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(setp, text="Automatic (recommended)", variable=self.auto_var,
                        command=self._toggle_auto).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.correct_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(setp, text="Language correction",
                        variable=self.correct_var).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        self.threshold_var = tk.StringVar(value="0.30")
        self.bp_low_var = tk.StringVar(value="auto")
        self.bp_high_var = tk.StringVar(value="auto")
        rows = [("Threshold (0-1):", self.threshold_var),
                ("Bandpass low (Hz):", self.bp_low_var),
                ("Bandpass high (Hz):", self.bp_high_var)]
        self._setting_entries = []
        for i, (lbl, var) in enumerate(rows, start=2):
            ttk.Label(setp, text=lbl).grid(row=i, column=0, sticky=tk.W, pady=1)
            e = ttk.Entry(setp, textvariable=var, width=10)
            e.grid(row=i, column=1, sticky=tk.W, padx=4, pady=1)
            self._setting_entries.append(e)
        self._toggle_auto()

        mid = ttk.Frame(self.root, padding=8)
        mid.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(11, 3.4), dpi=88, facecolor=BLACK)
        self.canvas = FigureCanvasTkAgg(self.figure, master=mid)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_stream()

        out = ttk.Frame(self.root, padding=8)
        out.pack(fill=tk.X)
        ttk.Label(out, text="Raw decoded text (as detected):").pack(anchor=tk.W)
        self.text_output = tk.Text(out, height=2, wrap=tk.WORD, font=("Consolas", 12))
        self.text_output.pack(fill=tk.X, pady=(0, 6))
        self.text_output.configure(state=tk.DISABLED)

        ttk.Label(out, text="Corrected text (language model):").pack(anchor=tk.W)
        self.corrected_output = tk.Text(out, height=2, wrap=tk.WORD, font=("Consolas", 14))
        self.corrected_output.pack(fill=tk.X, pady=(0, 6))
        self.corrected_output.configure(state=tk.DISABLED)

        ttk.Label(out, text="Morse view:").pack(anchor=tk.W)
        self.morse_output = tk.Text(out, height=2, wrap=tk.WORD, font=("Consolas", 11))
        self.morse_output.pack(fill=tk.X)
        self.morse_output.configure(state=tk.DISABLED)

    def _toggle_auto(self):
        state = tk.DISABLED if self.auto_var.get() else tk.NORMAL
        for e in self._setting_entries:
            e.config(state=state)

    # ---------- input ----------

    def on_load_file(self):
        path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.audio, self.sample_rate = self._load_audio(path)
        except Exception as exc:
            self._fail(f"Could not read file: {exc}")
            return
        if len(self.audio) == 0:
            self._fail("File loaded but had no audio.")
            return
        name = os.path.basename(path)
        self.file_label.config(text=name)
        self.duration_s = len(self.audio) / self.sample_rate
        self._resume_pos_s = 0.0      # new file -> start fresh
        self._decoded_once = False    # new file -> needs decoding
        self._clear_text()
        self._draw_stream(self.audio)
        self.status_label.config(text=f"Loaded {name}. Click Play + Decode.")
        self.play_button.config(state=tk.NORMAL if SOUNDDEVICE_AVAILABLE else tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        if not SOUNDDEVICE_AVAILABLE:
            self._decode_static()

    # ---------- decode (via unified pipeline) ----------

    def _run_pipeline(self):
        """Decode self.audio with the unified pipeline; store all results."""
        result = decode_pipeline(
            self.audio, self.sample_rate,
            center_frequency=self._manual_carrier(),
            threshold_ratio=self._threshold(),
            apply_correction=self.correct_var.get(),
        )
        self.carrier = result.carrier_hz
        self.smoothing_ms = result.smoothing_ms
        self.wpm = result.estimated_wpm
        self.last_envelope = result.envelope
        self.full_morse = result.morse
        self.full_decoded = result.raw_text
        self.full_corrected = result.corrected_text
        self.was_corrected = result.was_corrected
        self.char_times = self._compute_char_times(result.envelope, result.detector)

    def _decode_static(self):
        """Decode and show the full result without playback (fallback)."""
        try:
            self._run_pipeline()
        except Exception as exc:
            self._fail(f"Decoding failed: {exc}")
            return
        self._draw_stream(self.audio, self.last_envelope)
        if not self.full_morse:
            self.status_label.config(text=f"No Morse found (carrier {self.carrier:.0f} Hz).")
            return
        self._set_text(self.text_output, self.full_decoded)
        self._set_text(self.corrected_output, self.full_corrected)
        self._set_text(self.morse_output, self.full_morse)
        self._status_summary()

    def _compute_char_times(self, envelope, detector):
        dit = getattr(detector, "dit_samples", None)
        if not dit:
            return []
        threshold = detector.threshold_ratio * np.max(envelope)
        binary = (envelope > threshold).astype(np.int8)
        dah_thr = getattr(detector, "dah_threshold", 2.0)
        letter_gap = getattr(detector, "letter_gap_threshold", 2.0) * dit
        word_gap = getattr(detector, "word_gap_threshold", 5.0) * dit
        n = len(binary); i = 0; char_times = []; current = []
        while i < n and binary[i] == 0:
            i += 1

        def flush(end):
            if not current:
                return
            ch = INVERSE_MORSE_TABLE.get("".join(current), "?")
            char_times.append((end / self.sample_rate, ch))
            current.clear()

        while i < n:
            s = i
            while i < n and binary[i] == 1:
                i += 1
            on_len = i - s
            current.append("." if on_len < dah_thr * dit else "-")
            letter_end = i
            s = i
            while i < n and binary[i] == 0:
                i += 1
            gap_len = i - s
            if i >= n:
                flush(letter_end); break
            if gap_len >= word_gap:
                flush(letter_end); char_times.append((letter_end / self.sample_rate, " "))
            elif gap_len >= letter_gap:
                flush(letter_end)
        return char_times

    def _status_summary(self):
        extra = f", smoothing {self.smoothing_ms:.0f}ms"
        corr = "  (corrected)" if self.was_corrected else ""
        self.status_label.config(
            text=f"Carrier {self.carrier:.0f} Hz, ~{self.wpm:.1f} WPM{extra}.{corr}")

    # ---------- playback ----------

    def on_play(self):
        if self.audio is None or not SOUNDDEVICE_AVAILABLE or self.is_playing:
            return

        # Decode only once per loaded file. On resume we reuse the decode.
        if not self._decoded_once:
            self._clear_text()
            self.status_label.config(text="Decoding...")
            self.root.update_idletasks()
            try:
                self._run_pipeline()
            except Exception as exc:
                self._fail(f"Decoding failed: {exc}")
                return
            self._draw_stream(self.audio, self.last_envelope)
            if not self.full_morse:
                self.status_label.config(text=f"No Morse found (carrier {self.carrier:.0f} Hz).")
                return
            self._decoded_once = True

        # During playback, only the RAW text reveals in sync with the audio.
        # The corrected text is filled in at the end (see _finish_playback),
        # so the viewer sees the raw decode first, then the cleaned-up result.
        self.is_playing = True
        self.play_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Resume from where we stopped (or from the start if at the beginning).
        start_s = self._resume_pos_s
        if start_s >= self.duration_s - 0.05:
            start_s = 0.0  # was at/near the end -> restart from beginning
        start_sample = int(start_s * self.sample_rate)

        sd.stop()
        sd.play(self.audio[start_sample:], self.sample_rate)
        # Offset the reveal clock so elapsed time reflects the resume point.
        self.play_start_ms = self._now_ms() - int(start_s * 1000)
        self._reveal_tick()

    def _reveal_tick(self):
        if not self.is_playing:
            return
        elapsed = (self._now_ms() - self.play_start_ms) / 1000.0
        revealed = "".join(ch for (t, ch) in self.char_times if t <= elapsed)
        self._set_text(self.text_output, revealed)
        if self.duration_s > 0:
            frac = min(1.0, elapsed / self.duration_s)
            self._set_text(self.morse_output, self.full_morse[:int(frac * len(self.full_morse))])
        self._move_cursor(elapsed)
        self.status_label.config(text=f"Playing... {elapsed:.1f} / {self.duration_s:.1f} s")
        if elapsed >= self.duration_s:
            self._finish_playback()
            return
        self._play_id = self.root.after(80, self._reveal_tick)

    def _finish_playback(self):
        self.is_playing = False
        if self._play_id is not None:
            self.root.after_cancel(self._play_id)
            self._play_id = None
        if SOUNDDEVICE_AVAILABLE:
            sd.stop()
        self._resume_pos_s = 0.0  # finished -> next Play starts from the beginning
        self._set_text(self.text_output, self.full_decoded)
        self._set_text(self.corrected_output, self.full_corrected)
        self._set_text(self.morse_output, self.full_morse)
        self._move_cursor(self.duration_s)
        self.play_button.config(state=tk.NORMAL if SOUNDDEVICE_AVAILABLE else tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self._status_summary()

    def on_stop(self):
        if not self.is_playing:
            return
        self.is_playing = False
        if self._play_id is not None:
            self.root.after_cancel(self._play_id)
            self._play_id = None
        if SOUNDDEVICE_AVAILABLE:
            sd.stop()
        # Remember where we stopped, so Play resumes from here.
        self._resume_pos_s = (self._now_ms() - self.play_start_ms) / 1000.0
        self.play_button.config(state=tk.NORMAL if SOUNDDEVICE_AVAILABLE else tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text=f"Stopped at {self._resume_pos_s:.1f} s. Play to resume.")

    # ---------- decode parameters ----------

    def _manual_carrier(self):
        """Return a manual carrier if the user set one, else None (auto-detect)."""
        if self.auto_var.get():
            return None
        try:
            lo = float(self.bp_low_var.get()); hi = float(self.bp_high_var.get())
            return (lo + hi) / 2.0
        except ValueError:
            return None

    def _threshold(self):
        if self.auto_var.get():
            return 0.30
        try:
            return float(self.threshold_var.get())
        except ValueError:
            return 0.30

    # ---------- stream view ----------

    def _draw_empty_stream(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, facecolor=BLACK)
        ax.text(0.5, 0.5, "Stream view", color=GREEN, ha="center", va="center", fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MGREEN)
        self._stream_ax = ax
        self._cursor_line = None
        self.canvas.draw()

    def _draw_stream(self, audio, envelope=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111, facecolor=BLACK)
        sr = self.sample_rate
        if envelope is not None:
            sig = envelope / (np.max(envelope) + 1e-9)
        else:
            sig = np.abs(audio) / (np.max(np.abs(audio)) + 1e-9)
        max_points = 8000
        step = max(1, len(sig) // max_points)
        sig_ds = sig[::step]
        t = np.arange(len(sig_ds)) * step / sr
        ax.plot(t, sig_ds, color=GREEN, linewidth=0.9)
        ax.grid(True, color=DGREEN, linewidth=0.6)
        ax.set_xlabel("Time (s)", color=GREEN)
        ax.set_ylabel("Level", color=GREEN)
        ax.tick_params(colors=GREEN)
        ax.set_title("Stream view", color=GREEN, fontsize=12)
        ax.set_xlim(0, max(t[-1] if len(t) else 1, 1))
        for s in ax.spines.values():
            s.set_color(MGREEN)
        self.figure.tight_layout(pad=1.5)
        self._stream_ax = ax
        self._cursor_line = ax.axvline(x=0, color=CURSOR, linewidth=1.5)
        self._cursor_line.set_visible(False)
        self.canvas.draw()

    def _move_cursor(self, t_seconds):
        if self._cursor_line is None or self._stream_ax is None:
            return
        self._cursor_line.set_visible(True)
        self._cursor_line.set_xdata([t_seconds, t_seconds])
        self.canvas.draw_idle()

    # ---------- helpers ----------

    def on_clear(self):
        if self.is_playing:
            self.on_stop()
        self._clear_text()
        self.audio = None
        self._resume_pos_s = 0.0
        self._decoded_once = False
        self.file_label.config(text="(no file)")
        self.status_label.config(text="Cleared. Load a file.")
        self.play_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self._draw_empty_stream()

    def _clear_text(self):
        self._set_text(self.text_output, "")
        self._set_text(self.corrected_output, "")
        self._set_text(self.morse_output, "")

    def _fail(self, message):
        self._clear_text()
        self.status_label.config(text=f"FAILED: {message}")
        messagebox.showerror("Error", message)

    def _set_text(self, widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def _now_ms(self):
        return self.root.tk.call("clock", "milliseconds")

    def _load_audio(self, path):
        source = FileSource(filepath=path, chunk_size=1024)
        source.start()
        sr = source.sample_rate
        chunks = []
        while not source.is_finished:
            c = source.read_chunk()
            if len(c) > 0:
                chunks.append(c)
        source.stop()
        if not chunks:
            return np.array([], dtype=np.float32), sr
        return np.concatenate(chunks), sr


def main():
    root = tk.Tk()
    MorseViewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
