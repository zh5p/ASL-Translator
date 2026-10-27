import time
from typing import List, Tuple


class LetterBuffer:
    def __init__(self, max_len: int = 15):
        self._letters: List[Tuple[str, float]] = []  # (letter, timestamp)
        self._max_len = max_len

    def add(self, letter: str) -> None:
        self._letters.append((letter.upper(), time.time()))
        if len(self._letters) > self._max_len:
            self._letters.pop(0)

    def get_text(self) -> str:
        return "".join(letter for letter, _ in self._letters)

    def reset(self) -> None:
        self._letters.clear()

    def age_seconds(self) -> float:
        if not self._letters:
            return 0.0
        return time.time() - self._letters[-1][1]

    def __len__(self) -> int:
        return len(self._letters)

    def clear_last(self, count: int = 1) -> None:
        for _ in range(min(count, len(self._letters))):
            self._letters.pop()