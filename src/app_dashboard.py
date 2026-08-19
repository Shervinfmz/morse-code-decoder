"""
Morse Code Decoder - pipeline dashboard.

One window, two sources of audio.

  FILE MODE   Load a recording. The whole file is decoded first, then replayed
              with the text revealed in step with the audio. Because the entire
              signal is available at once, this is the more accurate mode: the
              threshold sees the true peak and the speed estimate sees every
              element in the recording.

  LIVE MODE   Decode audio as it arrives, from a microphone or from a virtual
              audio cable carrying a WebSDR receiver in a browser. Nothing is
              known in advance. A rolling window of recent audio is re-decoded
              several times a second, and a character is released only once the
              gap after it proves that it has ended.

Both modes drive the same seven panels and the same decoding code. What differs
is where the samples come from and how much of the signal exists when decoding
starts:

    file:  FileSource                        -> decode_pipeline -> replay
    live:  MicrophoneSource, PacedFileSource -> StreamDecoder   -> as it arrives

All four sources are AudioSource subclasses, so neither decoder knows or cares
which one it was handed.

Honest metrics only
-------------------
No confidence score in either mode: the detector compares a duration against a
threshold and looks the pattern up in a table, which produces a symbol, not a
probability.

No latency figure in file mode, because a recording is decoded and then
replayed - there is nothing real to measure. Live mode does show one, because
there the wait is real.

Run from the project root:
    python -m src.app_dashboard      opens in file mode
    python -m src.app_stream         opens in live mode
    python main.py dashboard | stream
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from scipy.signal import spectrogram, butter, sosfiltfilt

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import PolyCollection, LineCollection

try:
    import sounddevice as sd
    PLAYBACK_AVAILABLE = True
except Exception:
    sd = None
    PLAYBACK_AVAILABLE = False

try:
    from src.audio.microphone_source import MicrophoneSource
    AUDIO_INPUT_AVAILABLE = True
    AUDIO_INPUT_ERROR = ""
except Exception as exc:
    MicrophoneSource = None
    AUDIO_INPUT_AVAILABLE = False
    AUDIO_INPUT_ERROR = str(exc)

from src.audio.file_source import FileSource
from src.audio.paced_file_source import PacedFileSource
from src.processing.decode_pipeline import decode_pipeline, get_language_model
from src.processing.stream_decoder import StreamDecoder
from src.processing.timing import element_segments, character_events

# ---------------------------------------------------------------- appearance
BG = "#080d13"
PANEL_BG = "#0d151e"
PLOT_BG = "#05090d"
BORDER = "#1b2a38"
FG = "#d5e2f0"
MUTED = "#7d93a8"
CYAN = "#22d3ee"
GREEN = "#22c55e"
AMBER = "#f0b429"
VIOLET = "#b388ff"
DIT_C = "#22d3ee"
DAH_C = "#ff5fbf"
CURSOR = "#ff4d4d"
WAVE_C = "#31e07a"
GRID_C = "#1d2b39"

# ---------------------------------------------------------------- behaviour
DEFAULT_SAMPLE_RATE = 44100   # replaced by whatever the chosen source reports
CHUNK = 1024
WINDOW_S = 6.0                # seconds of signal visible at once
TICK_MS = 110                 # display refresh interval
WAVE_POINTS = 1100
FFT_N = 8192
PLOT_DPI = 74
RAIL_W = 158
DISPLAY_SECONDS = 12.0        # recent audio kept for drawing in live mode
WATERFALL_COLUMNS = 900
STAGGER_HEAVY = True          # refresh FFT and waterfall on alternate frames


class DashboardApp:
    """Morse decoding dashboard with a file mode and a live mode."""

    def __init__(self, root: tk.Tk, mode: str = "file"):
        self.root = root
        self.root.title("Morse Code Decoder - Pipeline Dashboard")
        self.root.geometry("1500x950")
        self.root.minsize(1100, 720)
        self.root.configure(bg=BG)

        self.mode = "file"
        self.sample_rate = DEFAULT_SAMPLE_RATE

        # ---- file mode state
        self.audio = None
        self.duration_s = 0.0
        self.envelope = None
        self.env_norm = None
        self.carrier = 0.0
        self.wpm = 0.0
        self.snr_db = 0.0
        self.symbols = []
        self.char_events = []
        self.full_raw = ""
        self.full_corrected = ""
        self.n_dits = 0
        self.n_dahs = 0
        self._decoded_once = False
        self._spec_db = None
        self._spec_t = None
        self._spec_f = None
        self._spec_vmin = -100.0
        self._spec_vmax = 0.0
        self._fft_ref = 1.0
        self.is_playing = False
        self.play_start_ms = None
        self._pos_s = 0.0

        # ---- live mode state
        self.decoder = StreamDecoder(sample_rate=self.sample_rate)
        self.source = None
        self.is_running = False
        self._worker = None
        self._worker_stop = threading.Event()
        self._text_queue = queue.Queue()
        self._started_at = None
        self._display = np.zeros(0, dtype=np.float32)
        self._display_start_s = 0.0
        self._wf_pending = np.zeros(0, dtype=np.float32)
        self._wf_columns = None
        self._wf_times = []
        self._wf_freqs = None
        self._devices = []

        # ---- shared
        self._tick_id = None
        self._logged_upto = 0
        self._word_cache = {}
        self._frame = 0
        self._draw_errors = 0
        self._water_img = None
        self.panels = {}

        self._build_ui()
        self.set_mode(mode)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self._build_topbar()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # The sidebar packs first, against the right edge. Tk allocates space
        # in packing order, so a sidebar packed after an expanding frame is the
        # first thing starved on a narrow or DPI-scaled display.
        side = tk.Frame(body, bg=BG, width=300)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        side.pack_propagate(False)
        self._build_sidebar(side)

        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_panels(left)
        self._build_output(left)

        self._build_statusbar()

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill=tk.X, padx=10, pady=(8, 4))
        tk.Label(bar, text="MORSE CODE DECODER", bg=BG, fg=CYAN,
                 font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)

        self.live_badge = tk.Label(bar, text="  IDLE  ", bg="#16222e", fg=MUTED,
                                   font=("Segoe UI", 9, "bold"))
        self.live_badge.pack(side=tk.RIGHT, padx=(10, 0))
        self.carrier_top = tk.Label(bar, text="Carrier:  --", bg=BG, fg=CURSOR,
                                    font=("Consolas", 10, "bold"))
        self.carrier_top.pack(side=tk.RIGHT, padx=10)
        self.wpm_top = tk.Label(bar, text="WPM:  --", bg=BG, fg=AMBER,
                                font=("Consolas", 10, "bold"))
        self.wpm_top.pack(side=tk.RIGHT, padx=10)

        modebar = tk.Frame(self.root, bg=BG)
        modebar.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Label(modebar, text="Audio source:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 8))
        self.mode_var = tk.StringVar(value="file")
        self.mode_buttons = {}
        for key, text in (("file", "Audio File"), ("live", "Live Input")):
            button = tk.Radiobutton(
                modebar, text=text, value=key, variable=self.mode_var,
                command=lambda k=key: self.set_mode(k), indicatoron=False,
                width=14, padx=6, pady=3, font=("Segoe UI", 9, "bold"),
                bg="#16222e", fg=MUTED, selectcolor="#15803d",
                activebackground="#1d2f42", activeforeground="white",
                relief=tk.FLAT, borderwidth=0)
            button.pack(side=tk.LEFT, padx=2)
            self.mode_buttons[key] = button
        self.mode_hint = tk.Label(modebar, text="", bg=BG, fg=MUTED,
                                  font=("Segoe UI", 8))
        self.mode_hint.pack(side=tk.LEFT, padx=(14, 0))

        # One controls row per mode; only the active one is shown.
        self.control_area = tk.Frame(self.root, bg=BG)
        self.control_area.pack(fill=tk.X, padx=10, pady=(0, 6))
        self._build_file_controls()
        self._build_live_controls()

    def _build_file_controls(self):
        row = tk.Frame(self.control_area, bg=BG)
        self.file_controls = row

        self.load_btn = self._button(row, "Load Audio", self.on_load, "#243447")
        self.play_btn = self._button(row, "Play", self.on_play, "#15803d",
                                     state=tk.DISABLED)
        self.pause_btn = self._button(row, "Pause", self.on_pause, "#a16207",
                                      state=tk.DISABLED)
        self.file_stop_btn = self._button(row, "Stop", self.on_stop_file,
                                          "#b91c1c", state=tk.DISABLED)

        tk.Label(row, text="File:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(16, 4))
        self.file_label = tk.Label(row, text="(none)", bg=BG, fg=FG,
                                   font=("Consolas", 10))
        self.file_label.pack(side=tk.LEFT)

        tk.Label(row, text="Time:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(16, 4))
        self.time_label = tk.Label(row, text="00:00:00 / 00:00:00", bg=BG,
                                   fg=FG, font=("Consolas", 10))
        self.time_label.pack(side=tk.LEFT)

        self.progress = tk.Canvas(row, height=6, bg="#16222e",
                                  highlightthickness=0)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)
        self._progress_fill = self.progress.create_rectangle(0, 0, 0, 6,
                                                             fill=CYAN, width=0)

    def _build_live_controls(self):
        row = tk.Frame(self.control_area, bg=BG)
        self.live_controls = row

        tk.Label(row, text="Input device:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 6))
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(row, textvariable=self.device_var,
                                       width=42, state="readonly")
        self.device_box.pack(side=tk.LEFT, padx=(0, 8))

        self.refresh_btn = self._button(row, "Refresh", self._populate_devices,
                                        "#243447")
        self.start_btn = self._button(row, "Start Listening", self.on_start,
                                      "#15803d")
        self.live_stop_btn = self._button(row, "Stop", self.on_stop_live,
                                          "#b91c1c", state=tk.DISABLED)
        self.simulate_btn = self._button(row, "Stream a File", self.on_simulate,
                                         "#4c1d95")

        tk.Label(row, text="Elapsed:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(16, 4))
        self.elapsed_label = tk.Label(row, text="00:00:00", bg=BG, fg=FG,
                                      font=("Consolas", 10))
        self.elapsed_label.pack(side=tk.LEFT)

    def _button(self, parent, text, command, colour, state=tk.NORMAL):
        button = tk.Button(parent, text=text, command=command, bg=colour,
                           fg="white", activebackground=colour,
                           activeforeground="white", relief=tk.FLAT, padx=14,
                           pady=4, state=state, font=("Segoe UI", 9, "bold"),
                           disabledforeground="#5a6b7c")
        button.pack(side=tk.LEFT, padx=3)
        return button

    # -- panels ----------------------------------------------------------

    def _build_panels(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill=tk.BOTH, expand=True)
        wrap.columnconfigure(0, weight=1)
        specs = [
            ("wave", 1, "TIME-DOMAIN SIGNAL", "", 3, False),
            ("env", 2, "ENVELOPE +\nMORSE DETECTOR", "(Cyan=DIT, Magenta=DAH)",
             3, False),
            ("fft", 3, "FFT SPECTRUM", "(Carrier Detection)", 3, False),
            ("water", 4, "WATERFALL\nSPECTROGRAM", "", 3, True),
            ("timeline", 5, "MORSE TIMELINE", "(DIT = dot, DAH = bar)", 2, False),
        ]
        for key, number, title, subtitle, weight, colorbar in specs:
            self.panels[key] = self._make_panel(wrap, number - 1, number, title,
                                                subtitle, weight, colorbar)
        rail = self.panels["wave"]["rail"]
        tk.Label(rail, text="Amplitude", bg=PANEL_BG, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor=tk.W, padx=10, pady=(6, 0))
        self.amp_label = tk.Label(rail, text="--", bg=PANEL_BG, fg=FG,
                                  font=("Consolas", 8))
        self.amp_label.pack(anchor=tk.W, padx=10)

    def _make_panel(self, parent, row, number, title, subtitle, weight, colorbar):
        parent.rowconfigure(row, weight=weight)
        frame = tk.Frame(parent, bg=PANEL_BG, highlightbackground=BORDER,
                         highlightthickness=1)
        frame.grid(row=row, column=0, sticky="nsew", pady=2)

        rail = tk.Frame(frame, bg=PANEL_BG, width=RAIL_W)
        rail.pack(side=tk.LEFT, fill=tk.Y)
        rail.pack_propagate(False)
        head = tk.Frame(rail, bg=PANEL_BG)
        head.pack(fill=tk.X, padx=8, pady=(8, 0))
        tk.Label(head, text=str(number), bg=PANEL_BG, fg=CYAN,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        tk.Label(head, text=title, bg=PANEL_BG, fg=CYAN, justify=tk.LEFT,
                 font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(8, 0))
        if subtitle:
            tk.Label(rail, text=subtitle, bg=PANEL_BG, fg=MUTED, justify=tk.LEFT,
                     wraplength=RAIL_W - 20,
                     font=("Segoe UI", 8)).pack(anchor=tk.W, padx=(28, 6))

        plot = tk.Frame(frame, bg=PLOT_BG)
        plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        figure = Figure(figsize=(7.0, 1.7), dpi=PLOT_DPI, facecolor=PANEL_BG)
        if colorbar:
            axes = figure.add_axes([0.045, 0.24, 0.885, 0.70])
            cax = figure.add_axes([0.945, 0.24, 0.012, 0.70])
        else:
            axes = figure.add_axes([0.045, 0.24, 0.945, 0.70])
            cax = None
        self._style_axes(axes)
        canvas = FigureCanvasTkAgg(figure, master=plot)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        live = tk.Label(plot, text="IDLE", bg=PLOT_BG, fg=MUTED,
                        font=("Segoe UI", 8, "bold"))
        live.place(relx=1.0, rely=0.0, x=-8, y=3, anchor="ne")
        return {"frame": frame, "rail": rail, "fig": figure, "ax": axes,
                "cax": cax, "canvas": canvas, "live": live, "row": row}

    def _style_axes(self, axes):
        axes.set_facecolor(PLOT_BG)
        axes.tick_params(colors=MUTED, labelsize=6.5, length=2)
        axes.grid(True, color=GRID_C, linewidth=0.4, alpha=0.7)
        for spine in axes.spines.values():
            spine.set_color(BORDER)

    def _build_output(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill=tk.X, pady=(4, 0))

        box6 = tk.Frame(row, bg=PANEL_BG, highlightbackground=BORDER,
                        highlightthickness=1)
        box6.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self._panel_header(box6, 6, "DECODED TEXT", GREEN)
        self.text_output = tk.Text(box6, height=3, bg=PLOT_BG, fg="#43ff5d",
                                   font=("Consolas", 13), relief=tk.FLAT,
                                   wrap=tk.WORD)
        self.text_output.pack(fill=tk.X, padx=8, pady=(2, 0))
        self.text_output.configure(state=tk.DISABLED)
        tk.Label(box6, text="Language-model corrected", bg=PANEL_BG, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor=tk.W, padx=8, pady=(4, 0))
        self.corrected_output = tk.Text(box6, height=2, bg=PLOT_BG, fg="#9be7ff",
                                        font=("Consolas", 12), relief=tk.FLAT,
                                        wrap=tk.WORD)
        self.corrected_output.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.corrected_output.configure(state=tk.DISABLED)
        buttons = tk.Frame(box6, bg=PANEL_BG)
        buttons.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._button(buttons, "Clear Text", self.on_clear_text, "#243447")
        self._button(buttons, "Copy Text", self.on_copy_text, "#243447")

        box7 = tk.Frame(row, bg=PANEL_BG, highlightbackground=BORDER,
                        highlightthickness=1)
        box7.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))
        self._panel_header(box7, 7, "DECODE LOG", AMBER)
        logwrap = tk.Frame(box7, bg=PANEL_BG)
        logwrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 8))
        scroll = tk.Scrollbar(logwrap, bg=PANEL_BG, troughcolor=PLOT_BG, width=10)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_output = tk.Text(logwrap, height=9, bg=PLOT_BG, fg="#43ff5d",
                                  font=("Consolas", 9), relief=tk.FLAT,
                                  yscrollcommand=scroll.set)
        self.log_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_output.configure(state=tk.DISABLED)
        scroll.config(command=self.log_output.yview)

    def _panel_header(self, parent, number, title, colour):
        head = tk.Frame(parent, bg=PANEL_BG)
        head.pack(fill=tk.X, padx=8, pady=(6, 0))
        tk.Label(head, text=str(number), bg=PANEL_BG, fg=CYAN,
                 font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(head, text=title, bg=PANEL_BG, fg=colour,
                 font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(8, 0))

    # -- sidebar ---------------------------------------------------------

    def _build_sidebar(self, parent):
        stats = self._section(parent, "SIGNAL & DECODER STATISTICS")
        self.stat_labels = {}
        for key, name in [("quality", "Carrier Quality"),
                          ("wpm", "Speed (WPM)"),
                          ("carrier", "Carrier Frequency"),
                          ("snr", "SNR"),
                          ("dits", "Dits Detected"),
                          ("dahs", "Dahs Detected"),
                          ("elements", "Elements Total"),
                          ("chars", "Characters"),
                          ("words", "Words"),
                          ("latency", "Character Latency"),
                          ("level", "Input Level")]:
            self.stat_labels[key] = self._stat_row(stats, name)
        tk.Frame(stats, bg=PANEL_BG, height=4).pack()

        ctrl = self._section(parent, "DECODER CONTROLS")
        tk.Label(ctrl, text="Detection threshold (fraction of envelope peak)",
                 bg=PANEL_BG, fg=MUTED, font=("Segoe UI", 8), wraplength=270,
                 justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=(4, 0))
        self.threshold_var = tk.DoubleVar(value=0.30)
        tk.Scale(ctrl, variable=self.threshold_var, from_=0.10, to=0.60,
                 resolution=0.01, orient=tk.HORIZONTAL, bg=PANEL_BG, fg=FG,
                 troughcolor="#16222e", highlightthickness=0,
                 sliderrelief=tk.FLAT, activebackground=CYAN,
                 font=("Consolas", 8), length=256).pack(padx=8)

        self.auto_carrier = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Auto detect carrier",
                       variable=self.auto_carrier,
                       command=self._sync_carrier_entry, bg=PANEL_BG, fg=FG,
                       selectcolor=PLOT_BG, activebackground=PANEL_BG,
                       activeforeground=FG, font=("Segoe UI", 9),
                       anchor=tk.W).pack(fill=tk.X, padx=6, pady=(4, 0))
        crow = tk.Frame(ctrl, bg=PANEL_BG)
        crow.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Label(crow, text="Manual carrier (Hz)", bg=PANEL_BG, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.carrier_entry = tk.Entry(crow, width=7, bg=PLOT_BG, fg=FG,
                                      insertbackground=FG, relief=tk.FLAT,
                                      font=("Consolas", 9))
        self.carrier_entry.pack(side=tk.RIGHT)
        self.carrier_entry.insert(0, "700")
        self.carrier_entry.config(state=tk.DISABLED)

        tk.Label(ctrl, text="WPM: adaptive. Speed is measured from the dit "
                            "length in the signal, so no manual range is used.",
                 bg=PANEL_BG, fg=MUTED, font=("Segoe UI", 8), wraplength=270,
                 justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=(2, 2))
        applyrow = tk.Frame(ctrl, bg=PANEL_BG)
        applyrow.pack(fill=tk.X, pady=(0, 6))
        self.redecode_btn = self._button(applyrow, "Apply & Re-decode",
                                         self.on_redecode, "#243447")

        self.help_section = self._section(parent, "HOW TO USE A WEBSDR")
        tk.Label(self.help_section, text=(
            "1. Open a WebSDR page and tune to a CW signal.\n"
            "     14100 kHz carries automated beacons around the\n"
            "     clock. 7000-7040 kHz has live operators.\n"
            "2. Pick a loopback device above: Stereo Mix, or\n"
            "     CABLE Output after installing VB-CABLE.\n"
            "3. Press Start Listening.\n"
            "4. Watch Carrier Quality. Green SIGNAL means a real\n"
            "     CW tone is present. Red NOISE means the receiver\n"
            "     is on an empty spot, and nothing is decoded while\n"
            "     it is red, so retune until it turns green."),
            bg=PANEL_BG, fg=MUTED, font=("Segoe UI", 8), justify=tk.LEFT,
            wraplength=268).pack(anchor=tk.W, padx=8, pady=(4, 8))

        self.note_label = tk.Label(parent, text="", bg=BG, fg="#4f6273",
                                   font=("Segoe UI", 8), wraplength=288,
                                   justify=tk.LEFT)
        self.note_label.pack(anchor=tk.W, padx=2, pady=(6, 0))

    def _section(self, parent, title):
        outer = tk.Frame(parent, bg=PANEL_BG, highlightbackground=BORDER,
                         highlightthickness=1)
        outer.pack(fill=tk.X, pady=(0, 8))
        tk.Label(outer, text=title, bg="#132030", fg=CYAN,
                 font=("Segoe UI", 8, "bold"), anchor=tk.W).pack(
            fill=tk.X, padx=1, pady=1, ipady=3, ipadx=8)
        return outer

    def _stat_row(self, parent, name):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill=tk.X, padx=10, pady=1)
        tk.Label(row, text=name, bg=PANEL_BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        value = tk.Label(row, text="--", bg=PANEL_BG, fg=CYAN,
                         font=("Consolas", 9, "bold"))
        value.pack(side=tk.RIGHT)
        return value

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg="#0b131c", highlightbackground=BORDER,
                       highlightthickness=1)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_fields = {}
        for key, name in [("status", "Status"), ("source", "Source"),
                          ("rate", "Sample Rate"), ("passes", "Decode Passes")]:
            cell = tk.Frame(bar, bg="#0b131c")
            cell.pack(side=tk.LEFT, padx=14, pady=3)
            tk.Label(cell, text=f"{name}:", bg="#0b131c", fg=MUTED,
                     font=("Segoe UI", 8)).pack(side=tk.LEFT)
            value = tk.Label(cell, text="--", bg="#0b131c", fg=AMBER,
                             font=("Consolas", 9))
            value.pack(side=tk.LEFT, padx=(5, 0))
            self.status_fields[key] = value
        self.status_fields["status"].config(text="idle")

    # ---------------------------------------------------------------- modes

    def set_mode(self, mode: str):
        """Switch modes, stopping whatever the other mode was doing."""
        if mode not in ("file", "live"):
            return
        if self.is_playing:
            self.on_stop_file()
        if self.is_running:
            self.on_stop_live()

        self.mode = mode
        self.mode_var.set(mode)
        for key, button in self.mode_buttons.items():
            button.config(fg="white" if key == mode else MUTED,
                          bg="#15803d" if key == mode else "#16222e")

        self.file_controls.pack_forget()
        self.live_controls.pack_forget()
        if mode == "file":
            self.file_controls.pack(fill=tk.X)
            self.help_section.pack_forget()
            self.redecode_btn.config(state=tk.NORMAL)
            self.mode_hint.config(
                text="Decodes the whole recording first, then replays it in step.")
            self.note_label.config(text=(
                "Every figure is measured from the signal. No confidence score, "
                "because the decoder produces symbols rather than probabilities. "
                "No latency either: this mode decodes a complete recording "
                "before replaying it, so there is nothing real to measure."))
        else:
            self.live_controls.pack(fill=tk.X)
            self.help_section.pack(fill=tk.X, pady=(0, 8))
            self.redecode_btn.config(state=tk.DISABLED)
            self.mode_hint.config(
                text="Decodes audio as it arrives. Nothing is known in advance.")
            self.note_label.config(text=(
                "Every figure is measured from the live signal. Character "
                "latency is how far behind the live edge a character was "
                "reported: the wait needed for the following gap to prove it "
                "had ended. This is the one place a latency figure is real."))
            self._populate_devices()
        self.status_fields["status"].config(text="idle")
        self._reset_stats()

    def _reset_stats(self):
        for label in self.stat_labels.values():
            label.config(text="--", fg=CYAN)
        self.wpm_top.config(text="WPM:  --")
        self.carrier_top.config(text="Carrier:  --")
        self.status_fields["passes"].config(text="--")
        self._set_live(False)

    def _fft_max_hz(self):
        """Top of the spectrum display: 2 kHz, or Nyquist if the rate is low."""
        return min(2000.0, self.sample_rate / 2.0)

    # -------------------------------------------------------------- FILE MODE

    def on_load(self):
        path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*.*")])
        if path:
            self.load_file(path)

    def load_file(self, path):
        """Load a recording. Also used by scripts and tests."""
        try:
            self.audio, rate = self._read_audio(path)
        except Exception as exc:
            self._fail(f"Could not read file: {exc}")
            return
        if len(self.audio) == 0:
            self._fail("File loaded but contained no audio.")
            return

        # Adopt the file's own rate. An 8 kHz recording read as 44.1 kHz puts
        # every frequency and every duration out by a factor of five, which
        # moves the carrier outside the search band entirely.
        self.sample_rate = int(rate)
        self.file_label.config(text=os.path.basename(path))
        self.duration_s = len(self.audio) / self.sample_rate
        self._pos_s = 0.0
        self._decoded_once = False
        self._clear_all_text()
        self.status_fields["status"].config(text="loaded")
        self.status_fields["source"].config(text=os.path.basename(path)[:30])
        self.status_fields["rate"].config(text=f"{self.sample_rate} Hz")
        self._update_time(0.0)
        self.play_btn.config(state=tk.NORMAL)
        self.file_stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)

    def _read_audio(self, path):
        source = FileSource(filepath=path, chunk_size=4096)
        source.start()
        rate = source.sample_rate
        chunks = []
        while not source.is_finished:
            chunk = source.read_chunk()
            if len(chunk) > 0:
                chunks.append(chunk)
        source.stop()
        if not chunks:
            return np.array([], dtype=np.float32), rate
        return np.concatenate(chunks), rate

    def decode_now(self):
        """Decode the whole recording and prepare every panel for playback."""
        centre = None
        if not self.auto_carrier.get():
            try:
                centre = float(self.carrier_entry.get())
            except ValueError:
                self._fail("Manual carrier must be a number in Hz.")
                return
        result = decode_pipeline(self.audio, self.sample_rate,
                                 center_frequency=centre,
                                 threshold_ratio=float(self.threshold_var.get()))

        self.envelope = result.envelope
        self.env_norm = self.envelope / (np.max(self.envelope) + 1e-9)
        self.carrier = result.carrier_hz
        self.wpm = result.estimated_wpm
        self.snr_db = self._compute_snr(self.audio, self.carrier)
        self.full_raw = result.raw_text
        self.full_corrected = result.corrected_text

        # Timing comes from src/processing/timing.py, the module the live
        # decoder also uses, so both modes measure elements identically.
        self.symbols = element_segments(result.envelope, result.detector,
                                        self.sample_rate)
        self.char_events = character_events(result.envelope, result.detector,
                                            self.sample_rate)
        self.n_dits = sum(1 for e in self.symbols if e.kind == "dit")
        self.n_dahs = sum(1 for e in self.symbols if e.kind == "dah")

        self._prepare_spectra()
        self._prepare_panels()
        self._update_file_stats(self.full_raw, self.full_corrected)
        self._decoded_once = True

    def _compute_snr(self, audio, carrier, bandwidth=100.0):
        nyquist = self.sample_rate / 2.0
        low = max(0.001, (carrier - bandwidth / 2) / nyquist)
        high = min(0.999, (carrier + bandwidth / 2) / nyquist)
        if not 0 < low < high < 1:
            return 0.0
        sos = butter(4, [low, high], btype="band", output="sos")
        signal = sosfiltfilt(sos, audio)
        noise_power = float(np.mean((audio - signal) ** 2))
        if noise_power < 1e-12:
            return 99.0
        return float(10 * np.log10(float(np.mean(signal ** 2))
                                   / (noise_power + 1e-12)))

    def _prepare_spectra(self):
        """Pre-compute the spectrogram once; each frame slices it per window."""
        nperseg = int(min(1024, max(64, len(self.audio) // 4)))
        freqs, times, values = spectrogram(self.audio, fs=self.sample_rate,
                                           nperseg=nperseg,
                                           noverlap=nperseg * 3 // 4,
                                           window="hann")
        mask = freqs <= self._fft_max_hz()
        self._spec_f = freqs[mask]
        self._spec_t = times
        self._spec_db = 10 * np.log10(values[mask, :] + 1e-12)
        self._spec_vmax = float(np.percentile(self._spec_db, 99.9))
        self._spec_vmin = self._spec_vmax - 80.0

        # A fixed reference so the live FFT panel reads in stable dB instead of
        # being renormalised every frame. Silence must look like silence.
        block = int(min(len(self.audio), FFT_N))
        window = np.hanning(block)
        peak = 0.0
        for start in range(0, max(1, len(self.audio) - block + 1), block):
            segment = self.audio[start:start + block]
            if len(segment) < block:
                break
            magnitude = np.abs(np.fft.rfft(segment * window, n=FFT_N))
            peak = max(peak, float(np.max(magnitude)))
        self._fft_ref = peak if peak > 0 else 1.0

    def on_play(self):
        if self.mode != "file" or self.audio is None or self.is_playing:
            return
        if not self._decoded_once:
            self.status_fields["status"].config(text="decoding...")
            self.root.update_idletasks()
            try:
                self.decode_now()
            except Exception as exc:
                self._fail(f"Decoding failed: {exc}")
                return

        start = self._pos_s
        if start >= self.duration_s - 0.05:
            start = 0.0
        if start <= 0.001:
            self._clear_all_text()

        self.is_playing = True
        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.file_stop_btn.config(state=tk.NORMAL)
        self._set_live(True)

        played = False
        if PLAYBACK_AVAILABLE:
            try:
                sd.stop()
                sd.play(self.audio[int(start * self.sample_rate):],
                        self.sample_rate)
                played = True
            except Exception:
                played = False
        self.status_fields["status"].config(
            text="decoding (audio playing)" if played
            else "decoding (no audio device, visuals only)")
        self.play_start_ms = self._now_ms() - int(start * 1000)
        self._tick_file()

    def _tick_file(self):
        if not self.is_playing:
            return
        elapsed = (self._now_ms() - self.play_start_ms) / 1000.0
        self._pos_s = elapsed

        try:
            self._redraw_file(elapsed)
        except Exception as exc:
            self._draw_errors += 1
            if self._draw_errors <= 3:
                print(f"[dashboard] _redraw_file failed: {exc}")
        self._update_time(elapsed)

        revealed = [e for e in self.char_events if e.end_seconds <= elapsed]
        raw = "".join(event.character for event in revealed)
        corrected = self._correct_live(raw)
        self._set_text(self.text_output, raw)
        self._set_text(self.corrected_output, corrected)
        self._update_file_stats(raw, corrected, elapsed)

        while self._logged_upto < len(revealed):
            event = revealed[self._logged_upto]
            if event.character != " ":
                self._log(f"[{self._fmt_time(event.end_seconds, millis=True)}]  "
                          f"{event.morse:<8}  {event.character}")
            self._logged_upto += 1

        if elapsed >= self.duration_s:
            self._finish_file()
            return
        self._tick_id = self.root.after(TICK_MS, self._tick_file)

    def on_pause(self):
        if not self.is_playing:
            return
        self.is_playing = False
        self._cancel_tick()
        self._stop_playback()
        self._pos_s = (self._now_ms() - self.play_start_ms) / 1000.0
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self._set_live(False)
        self.status_fields["status"].config(text=f"paused at {self._pos_s:.1f}s")

    def on_stop_file(self):
        if self.is_playing:
            self.is_playing = False
            self._cancel_tick()
        self._stop_playback()
        self._pos_s = 0.0
        self.play_btn.config(
            state=tk.NORMAL if self.audio is not None else tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.file_stop_btn.config(state=tk.DISABLED)
        self._set_live(False)
        self._update_time(0.0)
        self.status_fields["status"].config(text="stopped")

    def _finish_file(self):
        self.is_playing = False
        self._cancel_tick()
        self._stop_playback()
        self._pos_s = 0.0
        self._set_text(self.text_output, self.full_raw)
        self._set_text(self.corrected_output, self.full_corrected)
        self._update_file_stats(self.full_raw, self.full_corrected)
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.file_stop_btn.config(state=tk.DISABLED)
        self._set_live(False)
        self.status_fields["status"].config(text="done")

    def on_redecode(self):
        if self.mode != "file" or self.audio is None:
            return
        if self.is_playing:
            self.on_pause()
        self.status_fields["status"].config(text="re-decoding...")
        self.root.update_idletasks()
        try:
            self.decode_now()
        except Exception as exc:
            self._fail(f"Decoding failed: {exc}")
            return
        self._pos_s = 0.0
        self._clear_all_text()
        self._set_text(self.text_output, self.full_raw)
        self._set_text(self.corrected_output, self.full_corrected)
        self._update_time(0.0)
        self.status_fields["status"].config(
            text=f"re-decoded at threshold {self.threshold_var.get():.2f}")

    def _redraw_file(self, elapsed):
        self._frame += 1
        even = (self._frame % 2 == 0)
        do_fft = (not STAGGER_HEAVY) or even
        do_water = (not STAGGER_HEAVY) or (not even)

        rate = self.sample_rate
        right = min(elapsed, self.duration_s)
        left = max(0.0, right - WINDOW_S)
        i0 = int(left * rate)
        i1 = min(int(right * rate), len(self.audio))

        if i1 > i0 + 1:
            segment = self.audio[i0:i1]
            step = max(1, len(segment) // WAVE_POINTS)
            xs = np.arange(i0, i1, step) / rate
            self._wave_line.set_data(xs, segment[::step])
            self.panels["wave"]["ax"].set_xlim(left, left + WINDOW_S)
            self.amp_label.config(
                text=f"{float(np.min(segment)):+.3f} to "
                     f"{float(np.max(segment)):+.3f}")
            self.panels["wave"]["canvas"].draw_idle()

            step = max(1, (i1 - i0) // WAVE_POINTS)
            xs = np.arange(i0, i1, step) / rate
            self._env_line.set_data(xs, self.env_norm[i0:i1][::step])
            dit_boxes, dah_boxes = [], []
            for element in self.symbols:
                start, end = element.start_seconds, element.end_seconds
                if end < left or start > right:
                    continue
                box = [(start, 0.0), (end, 0.0), (end, 1.0), (start, 1.0)]
                (dit_boxes if element.kind == "dit" else dah_boxes).append(box)
            self._env_dits.set_verts(dit_boxes)
            self._env_dahs.set_verts(dah_boxes)
            self.panels["env"]["ax"].set_xlim(left, left + WINDOW_S)
            self.panels["env"]["canvas"].draw_idle()

        if do_fft and i1 > 64:
            block = self.audio[max(0, i1 - FFT_N):i1]
            if len(block) >= 64:
                window = np.hanning(len(block))
                magnitude = np.abs(np.fft.rfft(block * window, n=FFT_N))
                freqs = np.fft.rfftfreq(FFT_N, 1.0 / rate)
                mask = freqs <= self._fft_max_hz()
                decibels = 20 * np.log10(magnitude[mask] / self._fft_ref + 1e-12)
                self._fft_line.set_data(freqs[mask], np.clip(decibels, -100, 5))
                self.panels["fft"]["canvas"].draw_idle()

        if do_water and self._spec_t is not None and self._water_img is not None:
            c0 = int(np.searchsorted(self._spec_t, left))
            c1 = int(np.searchsorted(self._spec_t, right))
            if c1 > c0 + 1:
                self._water_img.set_data(self._spec_db[:, c0:c1])
                self._water_img.set_extent([self._spec_t[c0], self._spec_t[c1 - 1],
                                            self._spec_f[0], self._spec_f[-1]])
                self.panels["water"]["ax"].set_xlim(left, left + WINDOW_S)
                self.panels["water"]["canvas"].draw_idle()

        dit_x = [(e.start_seconds + e.end_seconds) / 2 for e in self.symbols
                 if e.kind == "dit" and e.start_seconds <= right
                 and e.end_seconds >= left]
        dah_segments = [[(e.start_seconds, 0.5), (e.end_seconds, 0.5)]
                        for e in self.symbols
                        if e.kind == "dah" and e.start_seconds <= right
                        and e.end_seconds >= left]
        self._tl_dits.set_data(dit_x, [0.5] * len(dit_x))
        self._tl_dahs.set_segments(dah_segments)
        self.panels["timeline"]["ax"].set_xlim(left, left + WINDOW_S)
        self.panels["timeline"]["canvas"].draw_idle()

    def _update_file_stats(self, raw, corrected, elapsed=None):
        text = corrected if corrected else raw
        chars = len(text.replace(" ", ""))
        words = len([w for w in text.split(" ") if w])
        if elapsed is None:
            dits, dahs = self.n_dits, self.n_dahs
        else:
            dits = sum(1 for e in self.symbols
                       if e.kind == "dit" and e.start_seconds <= elapsed)
            dahs = sum(1 for e in self.symbols
                       if e.kind == "dah" and e.start_seconds <= elapsed)

        self.stat_labels["quality"].config(text="n/a (file)", fg=MUTED)
        self.stat_labels["wpm"].config(text=f"{self.wpm:.1f}", fg=CYAN)
        self.stat_labels["carrier"].config(text=f"{self.carrier:.0f} Hz", fg=CYAN)
        self.stat_labels["snr"].config(text=f"{self.snr_db:.1f} dB", fg=CYAN)
        self.stat_labels["dits"].config(text=str(dits), fg=CYAN)
        self.stat_labels["dahs"].config(text=str(dahs), fg=CYAN)
        self.stat_labels["elements"].config(text=str(dits + dahs), fg=CYAN)
        self.stat_labels["chars"].config(text=str(chars), fg=CYAN)
        self.stat_labels["words"].config(text=str(words), fg=CYAN)
        self.stat_labels["latency"].config(text="n/a (replay)", fg=MUTED)
        self.wpm_top.config(text=f"WPM:  {self.wpm:.1f}")
        self.carrier_top.config(text=f"Carrier:  {self.carrier:.0f} Hz")

    def _update_time(self, elapsed):
        self.time_label.config(
            text=f"{self._fmt_time(elapsed)} / {self._fmt_time(self.duration_s)}")
        width = self.progress.winfo_width() or 1
        fraction = 0.0 if self.duration_s <= 0 else min(1.0, elapsed / self.duration_s)
        self.progress.coords(self._progress_fill, 0, 0, width * fraction, 6)

    # -------------------------------------------------------------- LIVE MODE

    def _populate_devices(self):
        if not AUDIO_INPUT_AVAILABLE:
            self.device_box["values"] = ["(no audio input available)"]
            self.device_box.current(0)
            self.start_btn.config(state=tk.DISABLED)
            self.status_fields["status"].config(text="no audio input library")
            return
        try:
            devices = MicrophoneSource.input_devices()
        except Exception as exc:
            devices = []
            self.status_fields["status"].config(text=f"device query failed: {exc}")

        self._devices = devices
        if not devices:
            self.device_box["values"] = ["(no input devices found)"]
            self.device_box.current(0)
            self.start_btn.config(state=tk.DISABLED)
            return

        self.device_box["values"] = [f"[{i}] {n}" for i, n in devices]
        preferred = 0
        for position, (_, name) in enumerate(devices):
            lowered = name.lower()
            if ("cable" in lowered or "voicemeeter" in lowered
                    or "virtual" in lowered or "stereo mix" in lowered
                    or "what u hear" in lowered or "loopback" in lowered):
                preferred = position
                break
        self.device_box.current(preferred)
        self.start_btn.config(state=tk.NORMAL)

    def _selected_device_id(self):
        label = self.device_var.get()
        for index, name in self._devices:
            if label == f"[{index}] {name}":
                return index, name
        return None, "default"

    def on_start(self):
        """Listen to a live input device."""
        if self.is_running or not AUDIO_INPUT_AVAILABLE:
            return
        device_id, device_name = self._selected_device_id()
        try:
            source = MicrophoneSource(sample_rate=DEFAULT_SAMPLE_RATE,
                                      chunk_size=CHUNK, device_id=device_id)
            source.start()
        except Exception as exc:
            messagebox.showerror("Could not open input",
                                 self._input_help(exc, device_name))
            self.status_fields["status"].config(text="input failed")
            return
        self._begin_live(source, device_name, play_audio=False)

    def on_simulate(self):
        """
        Replay a recording at real-time speed through the live decoder.

        The audio is already known, so this is not a live signal. It exercises
        the streaming path with no radio attached, and the status bar says so
        for as long as it runs.
        """
        if self.is_running:
            return
        path = filedialog.askopenfilename(
            title="Stream a file through the live decoder",
            filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*.*")])
        if not path:
            return
        try:
            source = PacedFileSource(path, chunk_size=CHUNK)
            source.start()
        except Exception as exc:
            messagebox.showerror("Could not open file", str(exc))
            return
        self._begin_live(source, f"FILE (simulated): {os.path.basename(path)}",
                         play_audio=True)

    def _begin_live(self, source, source_name, play_audio):
        """Start the streaming decoder against any AudioSource."""
        self.source = source

        # Adopt the rate the source actually delivers. An 8 kHz recording
        # decoded as 44.1 kHz puts every frequency out by a factor of five,
        # which pushes the carrier outside the search band entirely.
        rate = int(getattr(source, "sample_rate", DEFAULT_SAMPLE_RATE)
                   or DEFAULT_SAMPLE_RATE)
        self.sample_rate = rate
        self.decoder = StreamDecoder(sample_rate=rate)

        self._reset_display()
        self._clear_all_text()
        self._prepare_panels()

        self.is_running = True
        self._started_at = time.time()
        self.start_btn.config(state=tk.DISABLED)
        self.simulate_btn.config(state=tk.DISABLED)
        self.live_stop_btn.config(state=tk.NORMAL)
        self.device_box.config(state=tk.DISABLED)
        self._set_live(True)
        self.status_fields["rate"].config(text=f"{rate} Hz")
        self.status_fields["source"].config(text=source_name[:30])
        self.status_fields["status"].config(
            text="streaming from file (not a live signal)"
            if play_audio else "listening")

        # A simulated stream is played aloud so the Morse can be heard while
        # the text appears. A real device must NOT be played back: the audio
        # would be fed straight into the loopback being recorded.
        if play_audio and PLAYBACK_AVAILABLE:
            samples = getattr(source, "_samples", None)
            if samples is not None:
                try:
                    sd.stop()
                    sd.play(samples, rate)
                except Exception:
                    pass

        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._decode_worker, daemon=True)
        self._worker.start()
        self._tick_live()

    @staticmethod
    def _input_help(exc, device_name):
        """Turn a PortAudio error into something the user can act on."""
        detail = str(exc)
        message = f"Could not open '{device_name}'.\n\n{detail}\n\n"
        if "-9996" in detail or "Invalid device" in detail:
            message += (
                "Windows lists disabled recording devices, but they cannot be "
                "opened. This one is almost certainly disabled.\n\n"
                "  1. Right-click the speaker icon in the taskbar\n"
                "  2. Sound settings > More sound settings\n"
                "  3. Recording tab\n"
                "  4. Right-click the empty area, tick 'Show Disabled Devices'\n"
                "  5. Right-click the device > Enable\n"
                "  6. Come back here and press Refresh\n\n"
                "You can also press 'Stream a File' to run the live decoder on "
                "a recording, which needs no audio device at all.")
        elif "-9997" in detail or "Invalid sample rate" in detail:
            message += ("This device does not support 44100 Hz. Set its Default "
                        "Format to 44100 Hz in the same Recording tab, or "
                        "choose another device.")
        elif "-9985" in detail or "Device unavailable" in detail:
            message += "Another program is using this device. Close it and retry."
        return message

    def on_stop_live(self):
        if not self.is_running:
            return
        self.is_running = False
        self._worker_stop.set()
        self._cancel_tick()
        self._stop_playback()
        if self.source is not None:
            try:
                self.source.stop()
            except Exception:
                pass
            self.source = None
        self.start_btn.config(
            state=tk.NORMAL if AUDIO_INPUT_AVAILABLE else tk.DISABLED)
        self.simulate_btn.config(state=tk.NORMAL)
        self.live_stop_btn.config(state=tk.DISABLED)
        self.device_box.config(state="readonly")
        self._set_live(False)
        self.status_fields["status"].config(text="stopped")

    def _decode_worker(self):
        """Run the expensive decode off the Tk thread."""
        while not self._worker_stop.is_set():
            try:
                new_text = self.decoder.poll()
                if new_text:
                    self._text_queue.put(new_text)
            except Exception:
                pass
            time.sleep(0.05)

    def _tick_live(self):
        if not self.is_running:
            return

        fed = []
        for _ in range(64):
            if self.source is None:
                break
            chunk = self.source.read_chunk()
            if len(chunk) == 0:
                break
            fed.append(chunk)
        if fed:
            block = np.concatenate(fed)
            self.decoder.feed(block)
            self._append_display(block)

        while True:
            try:
                self._text_queue.get_nowait()
            except queue.Empty:
                break

        # Draw defensively. A failure in one panel must not kill the timer:
        # if the exception escaped this callback the next frame would never be
        # scheduled and the whole display would freeze silently.
        for step in (self._refresh_live_text, self._redraw_live,
                     self._update_live_stats):
            try:
                step()
            except Exception as exc:
                self._draw_errors += 1
                if self._draw_errors <= 3:
                    print(f"[dashboard] {step.__name__} failed: {exc}")

        self._tick_id = self.root.after(TICK_MS, self._tick_live)

    def _append_display(self, block):
        self._display = np.concatenate([self._display, block])
        max_samples = int(DISPLAY_SECONDS * self.sample_rate)
        excess = len(self._display) - max_samples
        if excess > 0:
            self._display = self._display[excess:]
            self._display_start_s += excess / self.sample_rate
        self._append_waterfall(block)

    def _append_waterfall(self, block):
        """Grow the spectrogram a few columns at a time as audio arrives."""
        self._wf_pending = np.concatenate([self._wf_pending, block])
        if len(self._wf_pending) < 4096:
            return
        nperseg = int(min(1024, len(self._wf_pending) // 2))
        freqs, times, values = spectrogram(self._wf_pending, fs=self.sample_rate,
                                           nperseg=nperseg,
                                           noverlap=nperseg * 3 // 4,
                                           window="hann")
        mask = freqs <= self._fft_max_hz()
        self._wf_freqs = freqs[mask]
        columns = 10 * np.log10(values[mask, :] + 1e-12)
        start = (self.decoder.stream_seconds
                 - len(self._wf_pending) / self.sample_rate)
        if self._wf_columns is None or self._wf_columns.shape[0] != columns.shape[0]:
            self._wf_columns = columns
            self._wf_times = list(start + times)
        else:
            self._wf_columns = np.concatenate([self._wf_columns, columns], axis=1)
            self._wf_times.extend(start + times)
        if self._wf_columns.shape[1] > WATERFALL_COLUMNS:
            cut = self._wf_columns.shape[1] - WATERFALL_COLUMNS
            self._wf_columns = self._wf_columns[:, cut:]
            self._wf_times = self._wf_times[cut:]
        self._wf_pending = self._wf_pending[-(nperseg * 3 // 4):]

    def _reset_display(self):
        self._display = np.zeros(0, dtype=np.float32)
        self._display_start_s = 0.0
        self._wf_pending = np.zeros(0, dtype=np.float32)
        self._wf_columns = None
        self._wf_times = []

    def _redraw_live(self):
        if len(self._display) < 128:
            return
        right = self.decoder.stream_seconds
        left = max(0.0, right - WINDOW_S)
        rate = self.sample_rate
        fft_max = self._fft_max_hz()

        start_index = max(0, int((left - self._display_start_s) * rate))
        segment = self._display[start_index:]
        if len(segment) > 1:
            step = max(1, len(segment) // WAVE_POINTS)
            times = (self._display_start_s
                     + (start_index + np.arange(0, len(segment), step)) / rate)
            self._wave_line.set_data(times, segment[::step])
            limit = max(0.05, float(np.max(np.abs(segment))) * 1.2)
            self.panels["wave"]["ax"].set_ylim(-limit, limit)
            self.panels["wave"]["ax"].set_xlim(left, left + WINDOW_S)
            self.amp_label.config(
                text=f"{float(np.min(segment)):+.3f} to "
                     f"{float(np.max(segment)):+.3f}")
            self.panels["wave"]["canvas"].draw_idle()

        envelope = self.decoder.envelope
        if envelope is not None and len(envelope) > 1:
            peak = float(np.max(envelope)) or 1.0
            step = max(1, len(envelope) // WAVE_POINTS)
            times = (self.decoder.envelope_start_seconds
                     + np.arange(0, len(envelope), step) / rate)
            self._env_line.set_data(times, envelope[::step] / peak)
            dit_boxes, dah_boxes = [], []
            for element in self.decoder.elements:
                if element.end_seconds < left or element.start_seconds > right:
                    continue
                box = [(element.start_seconds, 0.0), (element.end_seconds, 0.0),
                       (element.end_seconds, 1.0), (element.start_seconds, 1.0)]
                (dit_boxes if element.kind == "dit" else dah_boxes).append(box)
            self._env_dits.set_verts(dit_boxes)
            self._env_dahs.set_verts(dah_boxes)
            self.panels["env"]["ax"].set_xlim(left, left + WINDOW_S)
            self.panels["env"]["canvas"].draw_idle()

        block = self._display[-FFT_N:]
        if len(block) >= 1024:
            window = np.hanning(len(block))
            magnitude = np.abs(np.fft.rfft(block * window, n=FFT_N))
            freqs = np.fft.rfftfreq(FFT_N, 1.0 / rate)
            mask = freqs <= fft_max
            reference = float(np.max(magnitude)) or 1.0
            decibels = 20 * np.log10(magnitude[mask] / reference + 1e-12)
            self._fft_line.set_data(freqs[mask], np.clip(decibels, -100, 5))
            self.panels["fft"]["ax"].set_xlim(0, fft_max)
            if self.decoder.carrier_hz > 0:
                self._fft_marker.set_xdata([self.decoder.carrier_hz] * 2)
            self.panels["fft"]["canvas"].draw_idle()

        if self._wf_columns is not None and self._wf_columns.shape[1] > 2:
            times = np.asarray(self._wf_times)
            keep = times >= left
            if np.any(keep):
                data = self._wf_columns[:, keep]
                visible = times[keep]
                axes = self.panels["water"]["ax"]
                vmax = float(np.percentile(data, 99.5))
                if self._water_img is None:
                    self._water_img = axes.imshow(
                        data, aspect="auto", origin="lower", cmap="viridis",
                        vmin=vmax - 80, vmax=vmax,
                        extent=[visible[0], visible[-1],
                                self._wf_freqs[0], self._wf_freqs[-1]],
                        interpolation="nearest")
                    colourbar = self.panels["water"]["fig"].colorbar(
                        self._water_img, cax=self.panels["water"]["cax"])
                    colourbar.ax.tick_params(colors=MUTED, labelsize=5.5, length=2)
                    colourbar.outline.set_edgecolor(BORDER)
                else:
                    self._water_img.set_data(data)
                    self._water_img.set_clim(vmax - 80, vmax)
                    self._water_img.set_extent([visible[0], visible[-1],
                                                self._wf_freqs[0],
                                                self._wf_freqs[-1]])
                axes.set_xlim(left, left + WINDOW_S)
                self.panels["water"]["canvas"].draw_idle()

        dit_x = [(e.start_seconds + e.end_seconds) / 2
                 for e in self.decoder.elements
                 if e.kind == "dit" and e.end_seconds >= left]
        dah_segments = [[(e.start_seconds, 0.5), (e.end_seconds, 0.5)]
                        for e in self.decoder.elements
                        if e.kind == "dah" and e.end_seconds >= left]
        self._tl_dits.set_data(dit_x, [0.5] * len(dit_x))
        self._tl_dahs.set_segments(dah_segments)
        self.panels["timeline"]["ax"].set_xlim(left, left + WINDOW_S)
        self.panels["timeline"]["canvas"].draw_idle()

    def _refresh_live_text(self):
        self._set_text(self.text_output, self.decoder.text)
        self._set_text(self.corrected_output, self.decoder.corrected_text)
        events = self.decoder.events
        while self._logged_upto < len(events):
            event = events[self._logged_upto]
            if event.character != " ":
                self._log(f"[{self._fmt_time(event.end_seconds, millis=True)}]  "
                          f"{event.morse:<8}  {event.character}")
            self._logged_upto += 1

    def _update_live_stats(self):
        text = self.decoder.corrected_text or self.decoder.text
        chars = len(text.replace(" ", ""))
        words = len([w for w in text.split(" ") if w])
        dits = sum(1 for e in self.decoder.elements if e.kind == "dit")
        dahs = sum(1 for e in self.decoder.elements if e.kind == "dah")

        # Input level answers the first question when nothing decodes: is any
        # audio arriving at all? Silence here means the loopback is not
        # carrying sound, which is a Windows routing problem, not a tuning one.
        if len(self._display):
            recent = self._display[-int(0.5 * self.sample_rate):]
            peak = float(np.max(np.abs(recent))) if len(recent) else 0.0
        else:
            peak = 0.0
        if peak <= 1e-5:
            self.stat_labels["level"].config(text="SILENT", fg=CURSOR)
        else:
            decibels = 20 * np.log10(peak)
            colour = GREEN if peak > 0.01 else AMBER
            self.stat_labels["level"].config(text=f"{decibels:6.1f} dBFS", fg=colour)

        quality = self.decoder.carrier_quality
        if quality <= 0.0:
            self.stat_labels["quality"].config(text="--", fg=MUTED)
        elif self.decoder.signal_present:
            self.stat_labels["quality"].config(text=f"{quality:.2f}  SIGNAL",
                                               fg=GREEN)
        else:
            self.stat_labels["quality"].config(text=f"{quality:.2f}  NOISE",
                                               fg=CURSOR)
        self.stat_labels["wpm"].config(text=f"{self.decoder.estimated_wpm:.1f}",
                                       fg=CYAN)
        self.stat_labels["carrier"].config(
            text=f"{self.decoder.carrier_hz:.0f} Hz", fg=CYAN)
        self.stat_labels["snr"].config(text="n/a (live)", fg=MUTED)
        self.stat_labels["dits"].config(text=str(dits), fg=CYAN)
        self.stat_labels["dahs"].config(text=str(dahs), fg=CYAN)
        self.stat_labels["elements"].config(text=str(dits + dahs), fg=CYAN)
        self.stat_labels["chars"].config(text=str(chars), fg=CYAN)
        self.stat_labels["words"].config(text=str(words), fg=CYAN)
        self.stat_labels["latency"].config(
            text=f"{self.decoder.character_latency_seconds * 1000:.0f} ms",
            fg=CYAN)

        self.wpm_top.config(text=f"WPM:  {self.decoder.estimated_wpm:.1f}")
        self.carrier_top.config(text=f"Carrier:  {self.decoder.carrier_hz:.0f} Hz")
        passes = getattr(self.decoder, "decode_count",
                         getattr(self.decoder, "_decode_count", 0))
        self.status_fields["passes"].config(text=str(passes))
        if self._started_at:
            self.elapsed_label.config(
                text=self._fmt_time(time.time() - self._started_at))

        if self.source is not None and not isinstance(self.source, PacedFileSource):
            if peak <= 1e-5:
                self.status_fields["status"].config(
                    text="listening (SILENT - no audio reaching this device)")
            elif self.decoder.carrier_quality <= 0.0:
                self.status_fields["status"].config(text="listening (no audio yet)")
            elif self.decoder.signal_present:
                self.status_fields["status"].config(
                    text="listening (CW signal found)")
            else:
                self.status_fields["status"].config(
                    text="listening (noise only - retune to a CW signal)")

    # ------------------------------------------------------- shared plotting

    def _prepare_panels(self):
        for key in ("wave", "env", "fft", "water", "timeline"):
            axes = self.panels[key]["ax"]
            axes.clear()
            self._style_axes(axes)
        self.panels["water"]["cax"].clear()
        fft_max = self._fft_max_hz()

        axes = self.panels["wave"]["ax"]
        (self._wave_line,) = axes.plot([], [], color=WAVE_C, linewidth=0.7)
        if self.mode == "file" and self.audio is not None and len(self.audio):
            limit = float(np.max(np.abs(self.audio))) * 1.15 + 1e-6
        else:
            limit = 1.0
        axes.set_ylim(-limit, limit)

        axes = self.panels["env"]["ax"]
        (self._env_line,) = axes.plot([], [], color=AMBER, linewidth=0.9)
        self._env_dits = PolyCollection([], facecolors=DIT_C, alpha=0.40,
                                        edgecolors="none")
        self._env_dahs = PolyCollection([], facecolors=DAH_C, alpha=0.40,
                                        edgecolors="none")
        axes.add_collection(self._env_dits)
        axes.add_collection(self._env_dahs)
        axes.set_ylim(0, 1.05)

        axes = self.panels["fft"]["ax"]
        (self._fft_line,) = axes.plot([], [], color=VIOLET, linewidth=0.8)
        self._fft_marker = axes.axvline(
            self.carrier if self.mode == "file" else 0.0,
            color=CURSOR, linestyle="--", linewidth=1.1)
        axes.set_xlim(0, fft_max)
        axes.set_ylim(-100, 5)
        axes.set_ylabel("dB", color=MUTED, fontsize=6)

        axes = self.panels["water"]["ax"]
        axes.set_ylabel("Hz", color=MUTED, fontsize=6)
        if self.mode == "file" and self._spec_db is not None:
            self._water_img = axes.imshow(
                self._spec_db[:, :2], aspect="auto", origin="lower",
                cmap="viridis", vmin=self._spec_vmin, vmax=self._spec_vmax,
                extent=[0, WINDOW_S, self._spec_f[0], self._spec_f[-1]],
                interpolation="nearest")
            colourbar = self.panels["water"]["fig"].colorbar(
                self._water_img, cax=self.panels["water"]["cax"])
            colourbar.ax.tick_params(colors=MUTED, labelsize=5.5, length=2)
            colourbar.outline.set_edgecolor(BORDER)
        else:
            self._water_img = None

        axes = self.panels["timeline"]["ax"]
        (self._tl_dits,) = axes.plot([], [], linestyle="none", marker="o",
                                     markersize=5, color=DIT_C)
        self._tl_dahs = LineCollection([], colors=DAH_C, linewidths=5)
        axes.add_collection(self._tl_dahs)
        axes.set_ylim(0, 1)
        axes.set_yticks([])

        self._frame = 0
        for key in ("wave", "env", "fft", "water", "timeline"):
            self.panels[key]["canvas"].draw()

    # -------------------------------------------------------------- helpers

    def _correct_live(self, raw):
        """Correct completed words as they finish; leave the current one raw."""
        if not raw:
            return ""
        parts = raw.split(" ")
        finished, current = parts[:-1], parts[-1]
        model = get_language_model()
        out = []
        for word in finished:
            if not word:
                out.append(word)
                continue
            if word not in self._word_cache:
                self._word_cache[word] = model.correct_word(word)
            out.append(self._word_cache[word])
        out.append(current)
        return " ".join(out)

    def _cancel_tick(self):
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None

    @staticmethod
    def _stop_playback():
        if PLAYBACK_AVAILABLE:
            try:
                sd.stop()
            except Exception:
                pass

    @staticmethod
    def _fmt_time(seconds, millis=False):
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if millis:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"

    def _set_live(self, live):
        self.live_badge.config(text="  LIVE  " if live else "  IDLE  ",
                               fg=GREEN if live else MUTED,
                               bg="#0c2a18" if live else "#16222e")
        for panel in self.panels.values():
            panel["live"].config(text="LIVE" if live else "IDLE",
                                 fg=GREEN if live else MUTED)

    def _sync_carrier_entry(self):
        self.carrier_entry.config(
            state=tk.DISABLED if self.auto_carrier.get() else tk.NORMAL)

    def _log(self, line):
        self.log_output.configure(state=tk.NORMAL)
        self.log_output.insert(tk.END, line + "\n")
        self.log_output.see(tk.END)
        self.log_output.configure(state=tk.DISABLED)

    def on_clear_text(self):
        self._clear_all_text()
        if self.mode == "live":
            self.decoder.reset()

    def on_copy_text(self):
        text = self.corrected_output.get("1.0", tk.END).strip()
        if not text:
            text = self.text_output.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_fields["status"].config(text="text copied to clipboard")

    def _clear_all_text(self):
        self._set_text(self.text_output, "")
        self._set_text(self.corrected_output, "")
        self.log_output.configure(state=tk.NORMAL)
        self.log_output.delete("1.0", tk.END)
        self.log_output.configure(state=tk.DISABLED)
        self._logged_upto = 0
        self._word_cache = {}

    def _set_text(self, widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _fail(self, message):
        self.status_fields["status"].config(text="error")
        messagebox.showerror("Error", message)

    def _now_ms(self):
        return self.root.tk.call("clock", "milliseconds")


def main(mode: str = "file"):
    root = tk.Tk()
    app = DashboardApp(root, mode=mode)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (app.on_stop_file(), app.on_stop_live(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
