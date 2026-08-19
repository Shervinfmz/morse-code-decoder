"""
Morse Code Decoder - live stream dashboard.

Decodes Morse in real time from a live audio input while it is arriving. The
input can be any recording device the operating system exposes:

  * a **virtual audio cable** (VB-CABLE, VoiceMeeter), which carries audio
    digitally from another program - a WebSDR receiver in a browser, for
    example - with no acoustic path and therefore no added noise
  * a physical **microphone**

This is the streaming counterpart to `app_dashboard.py`. The dashboard decodes
a complete file and then replays it; this window decodes audio it has never
seen before, as it arrives, and can run indefinitely.

How it works
------------
    MicrophoneSource  ->  StreamDecoder  ->  this window
      (audio thread)      (worker thread)      (Tk thread)

  * `MicrophoneSource` captures blocks on its own thread and queues them.
  * The Tk timer drains that queue and calls `StreamDecoder.feed()`, which is
    cheap - it only appends to a rolling buffer.
  * A worker thread calls `StreamDecoder.poll()`, which is expensive, so the
    display never stalls while a decode is running.
  * New characters come back through a queue and are drawn on the Tk thread.

Honest description of the latency
---------------------------------
Characters appear roughly half a second after they are sent. That is not
inefficiency: a character is not finished until the gap that follows proves it
is finished. The decoder waits about eight dit lengths before committing, and
that wait is the latency. This is the one place in the project where a latency
figure is meaningful, and the status bar shows the measured value.

Run from the project root:
    python -m src.app_stream
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from scipy.signal import spectrogram

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
except Exception as exc:  # PortAudio or sounddevice missing
    MicrophoneSource = None
    AUDIO_INPUT_AVAILABLE = False
    AUDIO_INPUT_ERROR = str(exc)

try:
    import sounddevice as sd
    PLAYBACK_AVAILABLE = True
except Exception:
    sd = None
    PLAYBACK_AVAILABLE = False

from src.audio.paced_file_source import PacedFileSource
from src.processing.stream_decoder import StreamDecoder

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
DEFAULT_SAMPLE_RATE = 44100  # used until a source reports its own rate
CHUNK = 1024
WINDOW_S = 6.0          # seconds of signal on screen
TICK_MS = 110           # display refresh interval
WAVE_POINTS = 1100
FFT_N = 8192
FFT_MAX_HZ = 2000.0
PLOT_DPI = 74
RAIL_W = 158
DISPLAY_SECONDS = 12.0  # how much recent audio is kept for drawing
WATERFALL_COLUMNS = 900


class StreamApp:
    """Live streaming Morse decoder window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Morse Code Decoder - Live Stream Dashboard")
        self.root.geometry("1500x950")
        self.root.minsize(1100, 720)
        self.root.configure(bg=BG)

        # The real rate comes from whatever source is opened. A recording made
        # from a WebSDR is often 8 kHz, and decoding it as 44.1 kHz would put
        # a 750 Hz carrier at 4 kHz and make every timing measurement wrong.
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.decoder = StreamDecoder(sample_rate=self.sample_rate)
        self.source = None
        self.is_running = False

        self._tick_id = None
        self._worker = None
        self._worker_stop = threading.Event()
        self._text_queue = queue.Queue()
        self._logged_upto = 0
        self._started_at = None
        self._last_decode_latency = 0.0

        # rolling display state
        self._display = np.zeros(0, dtype=np.float32)
        self._display_start_s = 0.0
        self._wf_pending = np.zeros(0, dtype=np.float32)
        self._wf_columns = None
        self._wf_times = []
        self._wf_freqs = None

        self.panels = {}
        self._build_ui()
        self._populate_devices()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self._build_topbar()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

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
        tk.Label(bar, text="MORSE CODE DECODER  -  LIVE STREAM DASHBOARD",
                 bg=BG, fg=CYAN, font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)

        self.live_badge = tk.Label(bar, text="  IDLE  ", bg="#16222e", fg=MUTED,
                                   font=("Segoe UI", 9, "bold"))
        self.live_badge.pack(side=tk.RIGHT, padx=(10, 0))
        self.carrier_top = tk.Label(bar, text="Carrier:  --", bg=BG, fg=CURSOR,
                                    font=("Consolas", 10, "bold"))
        self.carrier_top.pack(side=tk.RIGHT, padx=10)
        self.wpm_top = tk.Label(bar, text="WPM:  --", bg=BG, fg=AMBER,
                                font=("Consolas", 10, "bold"))
        self.wpm_top.pack(side=tk.RIGHT, padx=10)

        ctl = tk.Frame(self.root, bg=BG)
        ctl.pack(fill=tk.X, padx=10, pady=(0, 6))

        tk.Label(ctl, text="Input device:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 6))
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(ctl, textvariable=self.device_var,
                                       width=46, state="readonly")
        self.device_box.pack(side=tk.LEFT, padx=(0, 8))

        self.refresh_btn = self._button(ctl, "Refresh", self._populate_devices, "#243447")
        self.start_btn = self._button(ctl, "Start Listening", self.on_start, "#15803d")
        self.stop_btn = self._button(ctl, "Stop", self.on_stop, "#b91c1c",
                                     state=tk.DISABLED)
        self.simulate_btn = self._button(ctl, "Stream a File", self.on_simulate, "#4c1d95")

        # Hearing the file while it decodes makes the demo far easier to
        # follow. A live device is already audible elsewhere, so this only
        # applies to "Stream a File".
        self.play_audio = tk.BooleanVar(value=PLAYBACK_AVAILABLE)
        tk.Checkbutton(ctl, text="Play sound", variable=self.play_audio,
                       bg=BG, fg=FG, selectcolor=PLOT_BG, activebackground=BG,
                       activeforeground=FG, font=("Segoe UI", 9),
                       state=tk.NORMAL if PLAYBACK_AVAILABLE else tk.DISABLED
                       ).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(ctl, text="Elapsed:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(16, 4))
        self.elapsed_label = tk.Label(ctl, text="00:00:00", bg=BG, fg=FG,
                                      font=("Consolas", 10))
        self.elapsed_label.pack(side=tk.LEFT)

    def _button(self, parent, text, command, colour, state=tk.NORMAL):
        button = tk.Button(parent, text=text, command=command, bg=colour, fg="white",
                           activebackground=colour, activeforeground="white",
                           relief=tk.FLAT, padx=14, pady=4, state=state,
                           font=("Segoe UI", 9, "bold"), disabledforeground="#5a6b7c")
        button.pack(side=tk.LEFT, padx=3)
        return button

    def _build_panels(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill=tk.BOTH, expand=True)
        wrap.columnconfigure(0, weight=1)
        specs = [
            ("wave", 1, "TIME-DOMAIN SIGNAL", "(Live input)", 3, False),
            ("env", 2, "ENVELOPE +\nMORSE DETECTOR", "(Cyan=DIT, Magenta=DAH)", 3, False),
            ("fft", 3, "FFT SPECTRUM", "(Carrier Detection)", 3, False),
            ("water", 4, "WATERFALL\nSPECTROGRAM", "", 3, True),
            ("timeline", 5, "MORSE TIMELINE", "(DIT = dot, DAH = bar)", 2, False),
        ]
        for key, number, title, subtitle, weight, colorbar in specs:
            self.panels[key] = self._make_panel(wrap, number - 1, number, title,
                                                subtitle, weight, colorbar)

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
        return {"frame": frame, "fig": figure, "ax": axes, "cax": cax,
                "canvas": canvas, "live": live}

    def _style_axes(self, axes):
        axes.set_facecolor(PLOT_BG)
        axes.tick_params(colors=MUTED, labelsize=6.5, length=2)
        axes.grid(True, color=GRID_C, linewidth=0.4, alpha=0.7)
        for spine in axes.spines.values():
            spine.set_color(BORDER)

    def _build_output(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill=tk.X, pady=(4, 0))

        box6 = tk.Frame(row, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        box6.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self._panel_header(box6, 6, "DECODED TEXT (Live stream)", GREEN)
        self.text_output = tk.Text(box6, height=3, bg=PLOT_BG, fg="#43ff5d",
                                   font=("Consolas", 13), relief=tk.FLAT, wrap=tk.WORD)
        self.text_output.pack(fill=tk.X, padx=8, pady=(2, 0))
        self.text_output.configure(state=tk.DISABLED)
        tk.Label(box6, text="Language-model corrected", bg=PANEL_BG, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor=tk.W, padx=8, pady=(4, 0))
        self.corrected_output = tk.Text(box6, height=2, bg=PLOT_BG, fg="#9be7ff",
                                        font=("Consolas", 12), relief=tk.FLAT, wrap=tk.WORD)
        self.corrected_output.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.corrected_output.configure(state=tk.DISABLED)
        buttons = tk.Frame(box6, bg=PANEL_BG)
        buttons.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._button(buttons, "Clear Text", self.on_clear, "#243447")
        self._button(buttons, "Copy Text", self.on_copy, "#243447")

        box7 = tk.Frame(row, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        box7.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))
        self._panel_header(box7, 7, "DECODE LOG (Live stream)", AMBER)
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

    def _build_sidebar(self, parent):
        stats = self._section(parent, "LIVE STREAM STATISTICS")
        self.stat_labels = {}
        for key, name in [("quality", "Carrier Quality"), ("wpm", "Speed (WPM)"),
                          ("carrier", "Carrier Frequency"),
                          ("elements", "Elements Detected"), ("chars", "Characters"),
                          ("words", "Words"), ("buffer", "Decode Window"),
                          ("latency", "Character Latency"),
                          ("decodetime", "Decode Pass Time")]:
            self.stat_labels[key] = self._stat_row(stats, name)
        tk.Frame(stats, bg=PANEL_BG, height=4).pack()

        info = self._section(parent, "HOW TO USE A WEBSDR")
        tk.Label(info, text=(
            "1. Open a WebSDR page and tune to a CW signal.\n"
            "2. Pick a loopback device above:\n"
            "     - Stereo Mix, if your sound card has it, or\n"
            "     - CABLE Output, after installing VB-CABLE\n"
            "     (set Windows playback to CABLE Input first).\n"
            "3. Press Start Listening.\n"
            "4. Watch Carrier Quality above. Green SIGNAL means a\n"
            "     real CW tone is present. Red NOISE means the\n"
            "     receiver is on an empty spot - retune until it\n"
            "     turns green. Nothing is decoded while it is red.\n\n"
            "A loopback carries the audio digitally, so nothing "
            "is lost to room noise. A microphone works too, but "
            "a speaker-to-microphone path adds noise and is far "
            "less reliable."),
            bg=PANEL_BG, fg=MUTED, font=("Segoe UI", 8), justify=tk.LEFT,
            wraplength=262).pack(anchor=tk.W, padx=8, pady=(4, 8))

        tk.Label(parent, text=(
            "Every figure above is measured from the live signal. Character "
            "latency is how far behind the live edge a character was reported "
            "- the wait needed for the following gap to prove it had ended. "
            "Decode pass time is how long one decode of the window took."),
            bg=BG, fg="#4f6273", font=("Segoe UI", 8),
            wraplength=286, justify=tk.LEFT).pack(anchor=tk.W, padx=2, pady=(6, 0))

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
        for key, name in [("status", "Status"), ("device", "Device"),
                          ("rate", "Sample Rate"), ("decodes", "Decode Passes")]:
            cell = tk.Frame(bar, bg="#0b131c")
            cell.pack(side=tk.LEFT, padx=14, pady=3)
            tk.Label(cell, text=f"{name}:", bg="#0b131c", fg=MUTED,
                     font=("Segoe UI", 8)).pack(side=tk.LEFT)
            value = tk.Label(cell, text="--", bg="#0b131c", fg=AMBER,
                             font=("Consolas", 9))
            value.pack(side=tk.LEFT, padx=(5, 0))
            self.status_fields[key] = value
        self.status_fields["status"].config(text="idle")
        self.status_fields["rate"].config(text=f"{self.sample_rate} Hz")

    # ------------------------------------------------------------- devices

    def _populate_devices(self):
        """Fill the dropdown with every input device the system exposes."""
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

        labels = [f"[{index}] {name}" for index, name in devices]
        self.device_box["values"] = labels
        # Prefer a virtual cable if one is installed - that is the WebSDR path.
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
        for index, name in getattr(self, "_devices", []):
            if label == f"[{index}] {name}":
                return index, name
        return None, "default"

    # ------------------------------------------------------------ listening

    def on_start(self):
        """Listen to a live input device."""
        if self.is_running or not AUDIO_INPUT_AVAILABLE:
            return
        device_id, device_name = self._selected_device_id()
        try:
            source = MicrophoneSource(sample_rate=DEFAULT_SAMPLE_RATE,
                                      chunk_size=CHUNK,
                                      device_id=device_id)
            source.start()
        except Exception as exc:
            messagebox.showerror("Could not open input", self._input_help(exc, device_name))
            self.status_fields["status"].config(text="input failed")
            return
        self._begin(source, device_name)

    @staticmethod
    def _input_help(exc, device_name):
        """Turn a PortAudio error into something the user can act on."""
        detail = str(exc)
        message = f"Could not open '{device_name}'.\n\n{detail}\n\n"
        if "-9996" in detail or "Invalid device" in detail:
            message += (
                "Windows lists disabled recording devices, but they cannot be "
                "opened. This one is almost certainly disabled.\n\n"
                "To enable it:\n"
                "  1. Right-click the speaker icon in the taskbar\n"
                "  2. Sound settings > More sound settings\n"
                "  3. Recording tab\n"
                "  4. Right-click in the empty area, tick 'Show Disabled "
                "Devices'\n"
                "  5. Right-click the device > Enable\n"
                "  6. Come back here and press Refresh\n\n"
                "You can also press 'Stream a File' to run the live decoder "
                "on a recording, which needs no audio device at all.")
        elif "-9997" in detail or "Invalid sample rate" in detail:
            message += ("This device does not support 44100 Hz. Set its "
                        "Default Format to 44100 Hz in the same Recording tab, "
                        "or choose a different device.")
        elif "-9985" in detail or "Device unavailable" in detail:
            message += "Another program is using this device. Close it and try again."
        return message

    def on_simulate(self):
        """
        Replay a file at real-time speed through the live decoder.

        The audio is already known, so this is not a live signal. It exists to
        exercise the streaming path on a machine with no radio attached, and
        the status bar says so plainly while it runs.
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
        import os as _os
        self._begin(source, f"FILE (simulated): {_os.path.basename(path)}")

        # Play the same file through the speakers, started at the same moment
        # the paced source starts serving it, so sound and display stay in step.
        if PLAYBACK_AVAILABLE and self.play_audio.get():
            try:
                samples = getattr(source, "_samples", None)
                if samples is not None:
                    sd.stop()
                    sd.play(samples, source.sample_rate)
            except Exception:
                pass

        # Play the file aloud as it streams, so the Morse can be heard while
        # the text appears. Listening to a live device must NOT do this: the
        # audio would be fed straight back into the loopback being recorded.
        if PLAYBACK_AVAILABLE and source._samples is not None:
            try:
                sd.stop()
                sd.play(source._samples, source.sample_rate)
            except Exception:
                pass

    def _begin(self, source, device_name):
        """Start the decode loop against any AudioSource."""
        self.source = source
        # Adopt the rate the source actually delivers. An 8 kHz recording
        # decoded as 44.1 kHz gets every frequency and duration wrong by a
        # factor of five.
        rate = int(getattr(source, "sample_rate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE)
        if rate != self.sample_rate:
            self.sample_rate = rate
            self.decoder = StreamDecoder(sample_rate=rate)
        self.decoder.reset()
        self._reset_display()
        self._clear_text_widgets()
        self._prepare_panels()

        self.is_running = True
        self._started_at = time.time()
        self.start_btn.config(state=tk.DISABLED)
        self.simulate_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.device_box.config(state=tk.DISABLED)
        self._set_live(True)
        self.status_fields["rate"].config(text=f"{self.sample_rate} Hz")
        simulated = device_name.startswith("FILE")
        self.status_fields["status"].config(
            text="streaming from file (not a live signal)" if simulated else "listening")
        self.status_fields["device"].config(text=device_name[:34])

        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._decode_worker, daemon=True)
        self._worker.start()
        self._tick()

    def on_stop(self):
        if not self.is_running:
            return
        self.is_running = False
        self._worker_stop.set()
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        if PLAYBACK_AVAILABLE:
            try:
                sd.stop()
            except Exception:
                pass
        if PLAYBACK_AVAILABLE:
            try:
                sd.stop()
            except Exception:
                pass
        if self.source is not None:
            try:
                self.source.stop()
            except Exception:
                pass
            self.source = None
        self.start_btn.config(state=tk.NORMAL if AUDIO_INPUT_AVAILABLE else tk.DISABLED)
        self.simulate_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.device_box.config(state="readonly")
        self._set_live(False)
        self.status_fields["status"].config(text="stopped")

    def _decode_worker(self):
        """Run the expensive decode off the Tk thread."""
        while not self._worker_stop.is_set():
            try:
                started = time.time()
                new_text = self.decoder.poll()
                if new_text:
                    self._last_decode_latency = time.time() - started
                    self._text_queue.put(new_text)
            except Exception:
                pass
            time.sleep(0.05)

    # ---------------------------------------------------------------- ticks

    def _tick(self):
        if not self.is_running:
            return

        # 1. drain captured audio into the decoder and the display buffer
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

        # 2. drain decoded text produced by the worker
        while True:
            try:
                self._text_queue.get_nowait()
            except queue.Empty:
                break

        self._refresh_text()
        self._redraw()
        self._update_stats()

        self._tick_id = self.root.after(TICK_MS, self._tick)

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
        freqs, times, values = spectrogram(self._wf_pending, fs=self.sample_rate,
                                           nperseg=1024, noverlap=768, window="hann")
        mask = freqs <= min(FFT_MAX_HZ, self.sample_rate / 2.0)
        self._wf_freqs = freqs[mask]
        columns = 10 * np.log10(values[mask, :] + 1e-12)
        start = self.decoder.stream_seconds - len(self._wf_pending) / self.sample_rate
        if self._wf_columns is None:
            self._wf_columns = columns
        else:
            self._wf_columns = np.concatenate([self._wf_columns, columns], axis=1)
        self._wf_times.extend(start + times)
        if self._wf_columns.shape[1] > WATERFALL_COLUMNS:
            cut = self._wf_columns.shape[1] - WATERFALL_COLUMNS
            self._wf_columns = self._wf_columns[:, cut:]
            self._wf_times = self._wf_times[cut:]
        # keep the overlap so the next batch joins seamlessly
        self._wf_pending = self._wf_pending[-768:]

    def _reset_display(self):
        self._display = np.zeros(0, dtype=np.float32)
        self._display_start_s = 0.0
        self._wf_pending = np.zeros(0, dtype=np.float32)
        self._wf_columns = None
        self._wf_times = []
        self._logged_upto = 0

    # -------------------------------------------------------------- drawing

    def _prepare_panels(self):
        for key in ("wave", "env", "fft", "water", "timeline"):
            axes = self.panels[key]["ax"]
            axes.clear()
            self._style_axes(axes)

        axes = self.panels["wave"]["ax"]
        (self._wave_line,) = axes.plot([], [], color=WAVE_C, linewidth=0.7)
        axes.set_ylim(-1.0, 1.0)

        axes = self.panels["env"]["ax"]
        (self._env_line,) = axes.plot([], [], color=AMBER, linewidth=0.9)
        self._env_dits = PolyCollection([], facecolors=DIT_C, alpha=0.40, edgecolors="none")
        self._env_dahs = PolyCollection([], facecolors=DAH_C, alpha=0.40, edgecolors="none")
        axes.add_collection(self._env_dits)
        axes.add_collection(self._env_dahs)
        axes.set_ylim(0, 1.05)

        axes = self.panels["fft"]["ax"]
        (self._fft_line,) = axes.plot([], [], color=VIOLET, linewidth=0.8)
        self._fft_marker = axes.axvline(0, color=CURSOR, linestyle="--", linewidth=1.1)
        axes.set_xlim(0, min(FFT_MAX_HZ, self.sample_rate / 2.0))
        axes.set_ylim(-100, 5)
        axes.set_ylabel("dB", color=MUTED, fontsize=6)

        axes = self.panels["water"]["ax"]
        self._water_img = None
        axes.set_ylabel("Hz", color=MUTED, fontsize=6)

        axes = self.panels["timeline"]["ax"]
        (self._tl_dits,) = axes.plot([], [], linestyle="none", marker="o",
                                     markersize=5, color=DIT_C)
        self._tl_dahs = LineCollection([], colors=DAH_C, linewidths=5)
        axes.add_collection(self._tl_dahs)
        axes.set_ylim(0, 1)
        axes.set_yticks([])

        for key in ("wave", "env", "fft", "water", "timeline"):
            self.panels[key]["canvas"].draw()

    def _redraw(self):
        if len(self._display) < 128:
            return
        right = self.decoder.stream_seconds
        left = max(0.0, right - WINDOW_S)

        # 1 - waveform
        start_index = max(0, int((left - self._display_start_s) * self.sample_rate))
        segment = self._display[start_index:]
        if len(segment) > 1:
            step = max(1, len(segment) // WAVE_POINTS)
            times = (self._display_start_s
                     + (start_index + np.arange(0, len(segment), step)) / self.sample_rate)
            self._wave_line.set_data(times, segment[::step])
            limit = max(0.05, float(np.max(np.abs(segment))) * 1.2)
            self.panels["wave"]["ax"].set_ylim(-limit, limit)
            self.panels["wave"]["ax"].set_xlim(left, left + WINDOW_S)
            self.panels["wave"]["canvas"].draw_idle()

        # 2 - envelope from the most recent decode, plus detected elements
        envelope = self.decoder.envelope
        if envelope is not None and len(envelope) > 1:
            peak = float(np.max(envelope)) or 1.0
            step = max(1, len(envelope) // WAVE_POINTS)
            times = (self.decoder.envelope_start_seconds
                     + np.arange(0, len(envelope), step) / self.sample_rate)
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

        # 3 - live spectrum of the newest audio
        block = self._display[-FFT_N:]
        if len(block) >= 1024:
            window = np.hanning(len(block))
            magnitude = np.abs(np.fft.rfft(block * window, n=FFT_N))
            freqs = np.fft.rfftfreq(FFT_N, 1.0 / self.sample_rate)
            mask = freqs <= min(FFT_MAX_HZ, self.sample_rate / 2.0)
            reference = float(np.max(magnitude)) or 1.0
            decibels = 20 * np.log10(magnitude[mask] / reference + 1e-12)
            self._fft_line.set_data(freqs[mask], np.clip(decibels, -100, 5))
            marker_hz = self.decoder.carrier_hz or self.decoder.peak_hz
            if marker_hz > 0:
                self._fft_marker.set_xdata([marker_hz] * 2)
                self._fft_marker.set_color(
                    CURSOR if self.decoder.signal_present else MUTED)
            self.panels["fft"]["canvas"].draw_idle()

        # 4 - waterfall
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
                                                self._wf_freqs[0], self._wf_freqs[-1]])
                axes.set_xlim(left, left + WINDOW_S)
                self.panels["water"]["canvas"].draw_idle()

        # 5 - timeline of every detected element
        dit_x = [(e.start_seconds + e.end_seconds) / 2 for e in self.decoder.elements
                 if e.kind == "dit" and e.end_seconds >= left]
        dah_segments = [[(e.start_seconds, 0.5), (e.end_seconds, 0.5)]
                        for e in self.decoder.elements
                        if e.kind == "dah" and e.end_seconds >= left]
        self._tl_dits.set_data(dit_x, [0.5] * len(dit_x))
        self._tl_dahs.set_segments(dah_segments)
        self.panels["timeline"]["ax"].set_xlim(left, left + WINDOW_S)
        self.panels["timeline"]["canvas"].draw_idle()

    # ----------------------------------------------------------------- text

    def _refresh_text(self):
        self._set_text(self.text_output, self.decoder.text)
        self._set_text(self.corrected_output, self.decoder.corrected_text)
        events = self.decoder.events
        while self._logged_upto < len(events):
            event = events[self._logged_upto]
            if event.character != " ":
                self._log(f"[{self._fmt_time(event.end_seconds, millis=True)}]  "
                          f"{event.morse:<8}  {event.character}")
            self._logged_upto += 1

    def _update_stats(self):
        text = self.decoder.corrected_text or self.decoder.text
        chars = len(text.replace(" ", ""))
        words = len([w for w in text.split(" ") if w])
        # Carrier quality is the tuning aid: green means a real CW tone is
        # present, red means the window is looking at noise and is not
        # decoding at all.
        quality = self.decoder.carrier_quality
        if quality <= 0.0:
            self.stat_labels["quality"].config(text="--", fg=MUTED)
        elif self.decoder.signal_present:
            self.stat_labels["quality"].config(text=f"{quality:.2f}  SIGNAL", fg=GREEN)
        else:
            self.stat_labels["quality"].config(text=f"{quality:.2f}  NOISE", fg=CURSOR)
        self.stat_labels["wpm"].config(text=f"{self.decoder.estimated_wpm:.1f}")
        # While the window is too noisy to decode there is no confirmed
        # carrier, but showing where the energy actually is helps the operator
        # tune towards a signal.
        if self.decoder.carrier_hz > 0:
            self.stat_labels["carrier"].config(
                text=f"{self.decoder.carrier_hz:.0f} Hz", fg=CYAN)
        elif self.decoder.peak_hz > 0:
            self.stat_labels["carrier"].config(
                text=f"~{self.decoder.peak_hz:.0f} Hz", fg=MUTED)
        else:
            self.stat_labels["carrier"].config(text="--", fg=CYAN)
        self.stat_labels["elements"].config(text=str(len(self.decoder.elements)))
        self.stat_labels["chars"].config(text=str(chars))
        self.stat_labels["words"].config(text=str(words))
        self.stat_labels["buffer"].config(text=f"{self.decoder.window_seconds:.0f} s")
        self.stat_labels["latency"].config(
            text=f"{self.decoder.character_latency_seconds * 1000:.0f} ms")
        self.stat_labels["decodetime"].config(
            text=f"{self._last_decode_latency * 1000:.0f} ms")
        self.wpm_top.config(text=f"WPM:  {self.decoder.estimated_wpm:.1f}")
        self.carrier_top.config(text=f"Carrier:  {self.decoder.carrier_hz:.0f} Hz")
        self.status_fields["decodes"].config(text=str(self.decoder._decode_count))
        if self._started_at:
            self.elapsed_label.config(
                text=self._fmt_time(time.time() - self._started_at))
        if self.is_running and not self.source.__class__.__name__.startswith("Paced"):
            if self.decoder.carrier_quality <= 0.0:
                self.status_fields["status"].config(text="listening (no audio yet)")
            elif self.decoder.signal_present:
                self.status_fields["status"].config(text="listening (CW signal found)")
            else:
                self.status_fields["status"].config(
                    text="listening (noise only - retune to a CW signal)")

    # -------------------------------------------------------------- helpers

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

    def _log(self, line):
        self.log_output.configure(state=tk.NORMAL)
        self.log_output.insert(tk.END, line + "\n")
        self.log_output.see(tk.END)
        self.log_output.configure(state=tk.DISABLED)

    def _set_text(self, widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _clear_text_widgets(self):
        self._set_text(self.text_output, "")
        self._set_text(self.corrected_output, "")
        self.log_output.configure(state=tk.NORMAL)
        self.log_output.delete("1.0", tk.END)
        self.log_output.configure(state=tk.DISABLED)
        self._logged_upto = 0

    def on_clear(self):
        self.decoder.reset()
        self._clear_text_widgets()

    def on_copy(self):
        text = self.decoder.corrected_text.strip() or self.decoder.text.strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_fields["status"].config(text="text copied to clipboard")


def main():
    root = tk.Tk()
    app = StreamApp(root)
    if not AUDIO_INPUT_AVAILABLE:
        messagebox.showwarning(
            "No audio input",
            "Live capture needs the sounddevice package and the PortAudio "
            f"library.\n\nDetails: {AUDIO_INPUT_ERROR}\n\n"
            "The window will open, but Start is disabled.")
    root.protocol("WM_DELETE_WINDOW", lambda: (app.on_stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
