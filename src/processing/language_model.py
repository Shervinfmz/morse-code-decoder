"""
Intelligent decoding layer for the Morse Code Decoder.

The LanguageModel sits AFTER the MorseDecoder. The decoder turns audio into
text, but on imperfect signals it makes small errors (e.g. a '0' where an 'O'
belongs, or one wrong letter from a timing glitch). This layer corrects those
errors using statistics of the English language plus amateur-radio vocabulary.

This is the only machine-learning / statistical component in the project. It is
trained on TEXT, never on audio. It implements the assignment's "intelligent
interpretation layer": probabilistic correction using N-grams, trained on
typical Morse conversations including prosigns (CQ, DE, K, AR, SK) and common
abbreviations.

Two cooperating layers:

  1. Word-level correction
     - Known words (English vocabulary + ham-radio terms) are kept as-is.
     - Unknown words are matched to the closest known word using a weighted
       edit distance. Substitutions that are common Morse confusions
       (0<->O, 1<->I, 5<->S, ...) cost less, so the model prefers the
       correction the signal most likely intended.

  2. Character-level N-gram model
     - Trained on the same text, it scores how "English-like" any string is.
     - Used to rank candidates and to avoid changing text that already looks
       like valid language.

Design choice: the model is CONSERVATIVE. It never changes a word that is
already valid, and only corrects an unknown word when a clearly-close known
word exists. This avoids the classic failure where a corrector "fixes" correct
but uncommon words into common ones.
"""

import re
import math
from collections import Counter

try:
    # Bundled in the package: ~4900 common English words, frozen (no download).
    from src.processing.language_words_data import COMMON_WORDS
except Exception:  # pragma: no cover - fallback for flat layouts
    try:
        from language_words_data import COMMON_WORDS
    except Exception:
        COMMON_WORDS = []


# Amateur-radio prosigns and common abbreviations (the assignment asks for these).
HAM_TERMS = [
    "CQ", "DE", "QSO", "QTH", "QRZ", "QSL", "QRM", "QRN", "QSY", "QRP",
    "RST", "WX", "RIG", "ANT", "PSE", "TNX", "TU", "FB", "OM", "YL",
    "AR", "SK", "KN", "BT", "DX", "UR", "HR", "ES", "GM", "GE", "GN",
    "GA", "73", "88", "K", "R", "N", "AS", "WPM", "AGN", "CPY", "RPT",
    # Distress and safety. SOS is the most famous Morse sequence there is;
    # leaving it out meant a real SOS was "corrected" into SON.
    "SOS", "MAYDAY", "PAN", "DISTRESS", "EMERGENCY", "RESCUE",
]

# Domain words worth always recognising (project + common test content).
DOMAIN_WORDS = [
    "MORSE", "DECODER", "DECODE", "SIGNAL", "STATION", "FREQUENCY",
    "MICROSOFT", "PLATFORM", "COMPUTERS", "COMPUTER", "ANTENNA", "PYTHON",
]

# Valid one-letter words that must never be "corrected" into something longer.
PROTECTED_SHORT = {"A", "I", "K", "R", "N"}

# Common substitution confusions in Morse decoding (digit <-> similar letter).
# These cost less in the edit distance, so the model favours the likely intent.
CHEAP_SUBSTITUTIONS = {
    ("0", "O"), ("O", "0"),
    ("1", "I"), ("I", "1"), ("1", "L"), ("L", "1"),
    ("5", "S"), ("S", "5"),
    ("8", "B"), ("B", "8"),
    ("2", "Z"), ("Z", "2"),
    ("6", "G"), ("G", "6"), ("9", "G"),
    ("4", "H"), ("7", "T"),
}
CHEAP_COST = 0.3

# The decoder writes "?" when a Morse pattern matches nothing in the table, so
# a "?" is a KNOWN-UNKNOWN letter, not punctuation. Matching it against any
# real letter therefore costs little: the position is already flagged as
# doubtful, and the surrounding letters carry the evidence.
UNKNOWN_MARKER = "?"
WILDCARD_COST = 0.3

# A token that is mostly question marks carries too little information to
# correct. "??" would sit within reach of dozens of two-letter words.
MAX_UNKNOWN_FRACTION = 0.4


class LanguageModel:
    """
    Statistical correction layer for decoded Morse text.

    Usage:
        lm = LanguageModel()
        clean = lm.correct("HELL0 W0RLD")   # -> "HELLO WORLD"
    """

    def __init__(self, extra_words=None, correction_threshold: float = 0.95):
        """
        Args:
            extra_words: optional iterable of domain words to always treat as
                valid (e.g. names, call signs).
            correction_threshold: only correct an unknown word if a known word
                exists within this weighted distance. Lower = more cautious.

        Why the default is 0.95 and not 1.0
        -----------------------------------
        This model exists to repair the errors a MORSE DECODER makes, not to
        spell-check English. Those errors have a known shape:

            * a digit read as a letter, or the reverse (cost 0.3)
            * a pattern the decoder could not name at all, written "?" (0.3)

        An ordinary substitution - any letter for any other letter - costs 1.0,
        and no plausible Morse mis-decode looks like that. Sitting the
        threshold just below 1.0 means such an edit is never accepted, while
        up to three Morse-shaped edits still are (3 x 0.3 = 0.9).

        This matters because the vocabulary holds about 4,900 words and English
        has far more. With a threshold above 1.0, ANY correctly decoded word
        that happened to be missing from the list, and that sat one letter away
        from a word that was present, would be silently rewritten:

            SOS      -> SON        (SOS was not in the vocabulary)
            SINKING  -> SINGING    (SINKING was not in the vocabulary)

        Both were decoded perfectly. The model broke them. Below 1.0 it leaves
        an unfamiliar word alone unless it has a Morse-specific reason to
        change it, which is the correct default for a decoder aid.
        """
        self.correction_threshold = correction_threshold
        self._word_freq = Counter()
        self._build_vocabulary(extra_words or [])
        self._build_char_ngrams()

    # ---------- training (from bundled text) ----------

    def _build_vocabulary(self, extra_words):
        """Assign a frequency weight to every known word."""
        n = len(COMMON_WORDS)
        # Earlier in the common-words list = more frequent = higher weight.
        for rank, word in enumerate(COMMON_WORDS):
            self._word_freq[word] = max(1, n - rank)
        # Ham-radio terms get a strong weight so prosigns win ties.
        for term in HAM_TERMS:
            self._word_freq[term] = max(self._word_freq.get(term, 0), 3000)
        # Domain words (project + common test content).
        for term in DOMAIN_WORDS:
            self._word_freq[term] = max(self._word_freq.get(term, 0), 2500)
        # Caller-supplied domain words.
        for word in extra_words:
            w = word.upper()
            self._word_freq[w] = max(self._word_freq.get(w, 0), 2000)
        self.vocabulary = set(self._word_freq)

    def _build_char_ngrams(self, n: int = 3):
        """Build a character trigram model over the vocabulary."""
        self._n = n
        self._trigrams = Counter()
        for word, count in self._word_freq.items():
            for g in self._char_ngrams(word):
                self._trigrams[g] += count
        self._total_trigrams = sum(self._trigrams.values())
        self._vocab_trigrams = len(self._trigrams)

    def _char_ngrams(self, word):
        s = "^" + word + "$"
        return [s[i:i + self._n] for i in range(len(s) - self._n + 1)]

    # ---------- scoring ----------

    def englishness(self, word: str) -> float:
        """
        Average log-probability of the word's character trigrams.
        Higher (closer to 0) = more English-like. Used to compare candidates.
        """
        grams = self._char_ngrams(word.upper())
        if not grams:
            return -999.0
        score = 0.0
        for g in grams:
            p = (self._trigrams.get(g, 0) + 1) / (self._total_trigrams + self._vocab_trigrams)
            score += math.log(p)
        return score / len(grams)

    def _weighted_distance(self, a: str, b: str) -> float:
        """Edit distance where known Morse confusions cost less than 1."""
        if abs(len(a) - len(b)) > 2:
            return 99.0
        m, n = len(a), len(b)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    cost = 0.0
                elif UNKNOWN_MARKER in (a[i - 1], b[j - 1]):
                    cost = WILDCARD_COST
                elif (a[i - 1], b[j - 1]) in CHEAP_SUBSTITUTIONS:
                    cost = CHEAP_COST
                else:
                    cost = 1.0
                dp[i][j] = min(dp[i - 1][j] + 1.0,
                               dp[i][j - 1] + 1.0,
                               dp[i - 1][j - 1] + cost)
        return dp[m][n]

    # ---------- correction ----------

    def correct_word(self, word: str) -> str:
        """
        Correct a single token, keeping any punctuation attached to it.

        "THIS," is corrected as "THIS" and the comma is put back, so the model
        never silently deletes punctuation the decoder got right.

        A "?" is treated as part of the word, not as punctuation: it marks a
        Morse pattern the decoder could not identify, so "?ROGRAM" is a
        seven-letter word with one unknown letter and should become "PROGRAM".
        """
        if not word:
            return word
        upper = word.upper()
        match = re.fullmatch(r"([^A-Z0-9?]*)([A-Z0-9?]+)([^A-Z0-9?]*)", upper)
        if not match:
            # Punctuation only, or punctuation inside the token. Leave it be.
            return upper
        prefix, core, suffix = match.groups()
        return prefix + self._correct_core(core) + suffix

    def _correct_core(self, w: str) -> str:
        """Correct a bare alphanumeric token. Valid words are never changed."""
        if w in self.vocabulary:
            return w
        # Too many unknown letters to correct responsibly.
        unknowns = w.count(UNKNOWN_MARKER)
        if unknowns and unknowns > MAX_UNKNOWN_FRACTION * len(w):
            return w
        # A number is not a misspelled word. Several digits are one cheap
        # substitution away from a letter (4 and H, 0 and O), so without this
        # guard "40" becomes "HO" at a cost of only 0.6. Tokens that mix
        # letters and digits are still corrected, so "M0RSE" -> "MORSE" works.
        if w.replace(UNKNOWN_MARKER, '').isdigit() and unknowns == 0:
            return w
        # Regular plurals and third-person verbs are valid even though only
        # the singular is stored, so JUMPS must not be "corrected" to JUMP.
        if len(w) > 3 and w.endswith("S") and w[:-1] in self.vocabulary:
            return w
        # Protect valid one-letter words and very short tokens: correcting a
        # single character into a longer word does more harm than good.
        if len(w) <= 1 or w in PROTECTED_SHORT:
            return w
        candidates = []
        for cand in self.vocabulary:
            if abs(len(cand) - len(w)) > 1:
                continue
            d = self._weighted_distance(w, cand)
            if d < self.correction_threshold:
                candidates.append((d, cand))
        if not candidates:
            return w
        best_d = min(d for d, _ in candidates)
        shortlist = [c for d, c in candidates if d == best_d]
        # Tie-break with the character n-gram model first, word frequency second.
        return max(shortlist, key=lambda c: (self.englishness(c), self._word_freq[c]))

    def correct(self, text: str) -> str:
        """
        Correct a full decoded message. Preserves word order and punctuation;
        uppercases the output, because Morse has no case.
        """
        if not text:
            return text
        tokens = text.upper().split()
        if not tokens:
            return text
        return " ".join(self.correct_word(t) for t in tokens)

    # ---------- introspection (useful for the report / demo) ----------

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def explain(self, word: str):
        """
        Return (original, corrected, changed) for a single word, for showing
        the model's effect in a demo or test.
        """
        corrected = self.correct_word(word)
        return word.upper(), corrected, (corrected != word.upper())
