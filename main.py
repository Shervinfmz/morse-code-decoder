"""
Morse Code Decoder - application entry point.

Launches one of the three graphical applications, or lists them if no choice
is given. Each window can also be started directly:

    python -m src.app_view        decode a file, text revealed in sync
    python -m src.app_dashboard   the same, with the full signal pipeline shown
    python -m src.app_stream      decode a LIVE audio stream as it arrives

Usage:
    python main.py                 show the menu
    python main.py view
    python main.py dashboard
    python main.py stream
"""

import sys

APPS = {
    "view": ("src.app_view",
             "Decode an audio file. Text is revealed in step with playback."),
    "dashboard": ("src.app_dashboard",
                  "Decode a file and show the whole signal pipeline: waveform, "
                  "envelope, FFT, waterfall, element timeline."),
    "stream": ("src.app_stream",
               "Decode a LIVE audio stream as it arrives, from a microphone or "
               "a virtual audio cable carrying a WebSDR receiver."),
}


def print_menu() -> None:
    print("Morse Code Decoder")
    print("=" * 60)
    print("Choose an application:")
    print()
    for name, (module, description) in APPS.items():
        print(f"  {name:<10} {description}")
        print(f"  {'':<10} python main.py {name}")
        print()
    print("Command line decoding without a window:")
    print("  python -m run_decoder --demo")


def main() -> None:
    if len(sys.argv) < 2:
        print_menu()
        return

    choice = sys.argv[1].strip().lower()
    if choice not in APPS:
        print(f"Unknown application: {choice!r}")
        print()
        print_menu()
        sys.exit(1)

    module_name = APPS[choice][0]
    import importlib
    module = importlib.import_module(module_name)
    module.main()


if __name__ == "__main__":
    main()
