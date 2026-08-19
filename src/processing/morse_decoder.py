"""
Morse code decoder.

Converts a Morse string (dits, dahs, and gaps) into readable text using
a lookup table.
"""

from src.processing.morse_table import INVERSE_MORSE_TABLE


class MorseDecoder:
    """Convert a Morse-code string to text via lookup table."""

    def __init__(self, unknown_marker: str = "?"):
        """
        Parameters
        ----------
        unknown_marker : str
            Character inserted when an unrecognized Morse pattern is encountered.
        """
        self.unknown_marker = unknown_marker

    def decode(self, morse: str) -> str:
        """
        Decode a Morse string into text.

        Input format:
            - '.' = dit, '-' = dah
            - single space between letters within a word
            - triple space between words

        Example:
            input:  ".... . .-.. .-.. ---   .-- --- .-. .-.. -.."
            output: "HELLO WORLD"

        Unknown patterns become `self.unknown_marker`.
        """
        if not morse:
            return ""

        words = morse.split("   ")
        decoded_words = []
        for word in words:
            letters = word.split(" ")
            decoded_letters = []
            for letter in letters:
                if not letter:
                    continue
                decoded_letters.append(
                    INVERSE_MORSE_TABLE.get(letter, self.unknown_marker)
                )
            decoded_words.append("".join(decoded_letters))

        return " ".join(decoded_words)
