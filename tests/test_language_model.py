"""
Verification for the LanguageModel (intelligent decoding layer).

Checks two things the model must get right:
  1. It FIXES common Morse decode errors (digit-for-letter confusions,
     single wrong characters) into valid words.
  2. It PRESERVES text that is already correct - it must not "over-correct"
     valid words into different ones.

Run from the project root:
    python -m tests.test_language_model
"""

import os
import sys

# Run this file directly: python tests/test_x.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.language_model import LanguageModel


def main():
    lm = LanguageModel()
    print("LanguageModel verification")
    print("=" * 60)
    print(f"Vocabulary size: {lm.vocabulary_size} words")
    print()

    # 1. Correction cases: corrupted -> expected clean text
    fix_cases = [
        ("HELL0 W0RLD", "HELLO WORLD"),
        ("MICR0S0FT", "MICROSOFT"),
        ("THE QUICK BR0WN F0X", "THE QUICK BROWN FOX"),
        ("THANK Y0U F0R THE C0NTACT", "THANK YOU FOR THE CONTACT"),
        ("PLEASE SEND Y0UR NAME", "PLEASE SEND YOUR NAME"),
        ("CQ CQ CQ DE", "CQ CQ CQ DE"),
        ("THIS IS MICR0S0FT", "THIS IS MICROSOFT"),
        # "?" marks a Morse pattern the decoder could not identify. It is a
        # missing letter, not punctuation, so it must be filled in.
        ("MICROSOFT COMPUTERS ?ROGRAM", "MICROSOFT COMPUTERS PROGRAM"),
        ("P?OGRAM THE PLATF?RM", "PROGRAM THE PLATFORM"),
        ("MOR?E DEC?DER", "MORSE DECODER"),
        # A real distress call must survive intact.
        ("S0S TITANIC SINKING", "SOS TITANIC SINKING"),
    ]
    print("Correction cases (corrupted -> clean):")
    fix_ok = 0
    for corrupted, expected in fix_cases:
        out = lm.correct(corrupted)
        ok = out == expected
        fix_ok += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {corrupted!r} -> {out!r}")
        if not ok:
            print(f"        expected: {expected!r}")

    # 2. Preservation cases: valid words must stay unchanged
    # Numbers, punctuation and plurals must survive untouched. A digit is one
    # cheap substitution from a letter (4/H, 0/O), so "40" used to become "HO".
    # A token that is mostly unknown carries too little information to correct:
    # "??" sits within reach of dozens of two-letter words.
    # Words that are correct English but are NOT in the 4,917-word vocabulary.
    # Each sits one ordinary letter away from a word that IS in it, so before
    # the threshold was lowered below 1.0 every one of these was rewritten:
    # SOS -> SON, SINKING -> SINGING, SINK -> SING.
    preserve = ["SOS", "SINKING", "SINK", "TITANIC", "ICEBERG",
                "40", "73", "2026", "THIS,", "FREQUENCY.", "JUMPS", "??", "A?",
                "PYTHON", "DECODER", "SIGNAL", "ANTENNA", "MORSE", "WEATHER",
                "STATION", "MESSAGE", "RECEIVED", "WORLD", "HELLO", "A", "I"]
    print("\nPreservation cases (valid words must NOT change):")
    pres_ok = 0
    for word in preserve:
        out = lm.correct_word(word)
        ok = out == word
        pres_ok += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {word!r} -> {out!r}")

    print("\n" + "=" * 60)
    total = len(fix_cases) + len(preserve)
    passed = fix_ok + pres_ok
    print(f"Correction: {fix_ok}/{len(fix_cases)}   "
          f"Preservation: {pres_ok}/{len(preserve)}   "
          f"Total: {passed}/{total}")
    if passed == total:
        print("LanguageModel verification: PASS")
    else:
        print("LanguageModel verification: FAIL")


if __name__ == "__main__":
    main()
