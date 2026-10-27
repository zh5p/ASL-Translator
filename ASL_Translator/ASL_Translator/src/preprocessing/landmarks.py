"""Normalize hand landmarks and create temporal windows."""
import numpy as np
from collections import deque
from typing import List, Optional


class LandmarkPreprocessor:
    """Normalizes landmarks, smooths jitter via EMA, and manages sliding window buffer."""

    def __init__(self, window_size: int = 15, stride: int = 15, smooth_alpha: float = 0.15):
        self.window_size = window_size
        self.stride = stride
        self.smooth_alpha = smooth_alpha  # 0.15 = 15% new, 85% previous
        self._buffer: deque = deque(maxlen=window_size)
        self._ema_smoothed: Optional[np.ndarray] = None  # running EMA smoothed landmarks

    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks relative to wrist (landmark 0).

        Args:
            landmarks: ndarray of shape (21, 3) with x, y, z

        Returns:
            Normalized ndarray of shape (21, 3)
        """
        if landmarks is None or len(landmarks) == 0:
            return np.zeros((21, 3), dtype=np.float32)

        # Wrist-centered + scale-normalized (from Owo AI approach)
        wrist = landmarks[0]
        normalized = landmarks - wrist
        hand_size = np.linalg.norm(normalized[12])
        if hand_size > 1e-8:
            normalized = normalized / hand_size

        return normalized.astype(np.float32)

    def _apply_ema_smooth(self, normalized: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to reduce landmark jitter."""
        if self._ema_smoothed is None:
            self._ema_smoothed = normalized.copy()
            return self._ema_smoothed

        # EMA: new = alpha * new + (1 - alpha) * previous
        self._ema_smoothed = (
            self.smooth_alpha * normalized
            + (1 - self.smooth_alpha) * self._ema_smoothed
        )
        return self._ema_smoothed

    def add_frame(self, landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Add frame landmarks to buffer. Returns window when full.

        Args:
            landmarks: ndarray of shape (21, 3), or None to fill with last known

        Returns:
            Windowed sequence (window_size, 21, 3) if buffer full, else None
        """
        if landmarks is None:
            if self._buffer:
                normalized = self._buffer[-1]
            else:
                normalized = np.zeros((21, 3), dtype=np.float32)
        else:
            normalized = self.normalize(landmarks)
            normalized = self._apply_ema_smooth(normalized)

        self._buffer.append(normalized)

        if len(self._buffer) == self.window_size:
            return np.array(self._buffer, dtype=np.float32)
        return None

    def create_windows(self, sequence: List[np.ndarray]) -> List[np.ndarray]:
        """Create non-overlapping windows from sequence of landmarks.

        Args:
            sequence: List of ndarray, each (21, 3)

        Returns:
            List of windowed ndarrays, each (window_size, 21, 3)
        """
        windows = []
        for i in range(0, len(sequence) - self.window_size + 1, self.stride):
            window = np.array(sequence[i : i + self.window_size], dtype=np.float32)
            windows.append(window)
        return windows

    def reset(self):
        self._buffer.clear()
        self._ema_smoothed = None

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)
