from typing import Optional, Set

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_word: bool = False


class WordDictionary:
    def __init__(self, word_list_path: Optional[str] = None):
        self._root = TrieNode()
        self._words: Set[str] = set()
        if word_list_path:
            self.load_from_file(word_list_path)

    def load_from_file(self, path: str) -> None:
        with open(path, "r") as f:
            for line in f:
                word = line.strip().upper()
                if word:
                    self._words.add(word)
                    self._insert(word)

    def _insert(self, word: str) -> None:
        node = self._root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True

    def contains(self, word: str) -> bool:
        node = self._root
        for ch in word.upper():
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_word

    def has_prefix(self, prefix: str) -> bool:
        node = self._root
        for ch in prefix.upper():
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def fuzzy_match(self, text: str, max_distance: int = 1) -> Optional[str]:
        text_upper = text.upper()
        if not fuzz:
            return text_upper if self.contains(text_upper) else None
        best_match: Optional[str] = None
        best_score = 0
        # Only check words within ±max_distance length
        candidates = [w for w in self._words if abs(len(w) - len(text_upper)) <= max_distance]
        for word in candidates:
            score = fuzz.ratio(text_upper, word)
            if score >= 80:
                if score > best_score:
                    best_score = score
                    best_match = word
        return best_match