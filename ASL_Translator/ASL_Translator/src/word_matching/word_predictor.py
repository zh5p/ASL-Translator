from typing import Optional, List, Tuple
from enum import Enum

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


class PredictState(Enum):
    """State of word prediction — determines UI action."""
    EXACT_WORD = "exact"       # Auto-commit the word
    PREFIX_MATCH = "prefix"   # Show completion suggestions
    FUZZY_FALLBACK = "fuzzy"   # Show "did you mean?" suggestion
    NO_MATCH = "none"          # No suggestion


# ASL fingerspelling confusion pairs — geometrically similar hand poses
CONFUSION_MAP = {
    'H': {'I', 'E'},
    'I': {'H', 'E'},
    'M': {'N'},
    'N': {'M'},
    'S': {'T'},
    'T': {'S'},
    'K': {'P'},
    'P': {'K'},
    'U': {'V'},
    'V': {'U'},
    'A': {'S'},
    'S': {'A', 'T'},
    'O': {'Q', 'C'},
    'Q': {'O', 'G'},
    'G': {'Q', 'O'},
    'C': {'O', 'G'},
}


class WordPredictor:
    """Real-time word prediction for ASL fingerspelling.

    Algorithm:
      1. Exact word match → commit immediately
      2. Trie prefix match → show completions (assist, not auto-commit)
      3. Confusion variant → O(1) fuzzy fallback (max 1 substitution via confusion pairs)
      4. No match → let user continue spelling

    Key improvements over WordSegmenter:
    - Confusion-aware matching (H/I cost 0.5 vs 1.0)
    - Precomputed confusion variants for O(1) fuzzy lookup
    - Prefix completions shown as suggestions (not auto-committed)
    - Only auto-commit on exact word match
    - Stale buffer detection (3s timeout)
    """

    def __init__(self, dictionary, buffer, min_prefix_len: int = 2):
        self.dictionary = dictionary
        self.buffer = buffer
        self.min_prefix_len = min_prefix_len
        self._last_suggestion: Optional[str] = None
        self._confusion_index: dict[str, str] = {}
        self._build_confusion_index()

    def _build_confusion_index(self) -> None:
        """Pre-compute all 1-substitution confusion variants.

        This makes fuzzy matching O(1) instead of O(n) scanning.
        "HEIIO" → "HELLO" lookup becomes instant dict access.
        """
        for word in self.dictionary._words:
            # Add canonical word
            self._confusion_index[word] = word

            # Generate 1-substitution variants via confusion pairs
            for i, ch in enumerate(word):
                for conf_ch in CONFUSION_MAP.get(ch, set()):
                    variant = word[:i] + conf_ch + word[i+1:]
                    if variant not in self._confusion_index:
                        self._confusion_index[variant] = word

    def predict(self) -> Tuple[Optional[str], PredictState]:
        """Get word prediction and state for current buffer.

        Returns:
            (word_or_none, state)
            - EXACT_WORD: auto-commit immediately
            - PREFIX_MATCH: show suggestion, don't auto-commit
            - FUZZY_FALLBACK: show "did you mean?", don't auto-commit
            - NO_MATCH: no suggestion
        """
        text = self.buffer.get_text()
        text_len = len(text)

        if text_len < self.min_prefix_len:
            return None, PredictState.NO_MATCH

        # 1. Immediate exact word match → auto-commit
        if self.dictionary.contains(text):
            self._last_suggestion = None
            return text, PredictState.EXACT_WORD

        # 2. Check if buffer is a valid prefix → get completions
        if self.dictionary.has_prefix(text):
            completions = self._get_prefix_completions(text, limit=5)
            if len(completions) == 1 and completions[0] == text:
                # Exact word but buffer not reset — shouldn't happen
                return text, PredictState.EXACT_WORD

            # Return shortest completion as suggestion (most likely first)
            self._last_suggestion = completions[0] if completions else None
            return self._last_suggestion, PredictState.PREFIX_MATCH

        # 3. Fuzzy fallback via confusion variant index (only for len >= 3)
        if text_len >= 3 and text in self._confusion_index:
            canonical = self._confusion_index[text]
            if self.dictionary.contains(canonical):
                self._last_suggestion = canonical
                return canonical, PredictState.FUZZY_FALLBACK

        # 4. No match — check if buffer is stale
        if self.buffer.age_seconds() > 3.0:
            # Show best fuzzy match as "did you mean?" even if > 1 edit
            best = self._get_best_fuzzy(text)
            if best:
                self._last_suggestion = best
                return best, PredictState.FUZZY_FALLBACK

        return None, PredictState.NO_MATCH

    def _get_prefix_completions(self, prefix: str, limit: int = 5) -> List[str]:
        """Get words from dictionary that start with prefix."""
        results = []
        node = self.dictionary._root
        for ch in prefix.upper():
            if ch not in node.children:
                return results
            node = node.children[ch]

        self._dfs_collect(node, prefix.upper(), results)
        results.sort(key=lambda w: len(w))  # shortest first
        return results[:limit]

    def _dfs_collect(self, node, prefix: str, results: List[str]) -> None:
        if node.is_word:
            results.append(prefix)
        for ch, child in node.children.items():
            self._dfs_collect(child, prefix + ch, results)

    def _get_best_fuzzy(self, text: str) -> Optional[str]:
        """Find best fuzzy match using rapidfuzz ratio."""
        if not fuzz:
            return None

        text_upper = text.upper()
        best_match = None
        best_score = 0

        for word in self.dictionary._words:
            # Length must be close (±2)
            if abs(len(word) - len(text_upper)) > 2:
                continue

            score = fuzz.ratio(text_upper, word)
            if score > best_score and score >= 70:
                best_score = score
                best_match = word

        return best_match

    @property
    def last_suggestion(self) -> Optional[str]:
        """Get the last suggested word (for UI display)."""
        return self._last_suggestion