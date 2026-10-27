"""MediaPipe hand landmark extraction using the new tasks API."""
import mediapipe.tasks as mp_tasks
import importlib
import numpy as np
import cv2
from typing import Optional, Tuple, List
import os
from dataclasses import dataclass

# Import Image and ImageFormat via importlib (path is non-standard)
_image_module = importlib.import_module('mediapipe.tasks.python.vision.core.image')
ImageClass = _image_module.Image
ImageFormat = _image_module.ImageFormat


@dataclass
class HandDetection:
    """Result of hand detection with landmarks and handedness."""
    landmarks: np.ndarray  # (21, 3) x, y, z normalized coords
    handedness: str  # "Left" or "Right"
    confidence: float  # detection confidence


class MediaPipeDetector:
    """Extracts 21 hand landmarks from frames using MediaPipe Tasks Vision."""

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.3,
        min_hand_presence_confidence: float = 0.5,
    ):
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence

        base_options = mp_tasks.BaseOptions(model_asset_path=self._get_model_path())
        options = mp_tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = mp_tasks.vision.HandLandmarker.create_from_options(options)

    def _get_model_path(self) -> str:
        """Return path to hand landmarker model. Uses bundled path in exe."""
        import sys
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller exe — use bundled model in _MEIPASS
            base_dir = sys._MEIPASS
        else:
            # Running as script — use project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_dir, 'models', 'hand_landmarker.task')

    def detect_landmarks(self, frame: Optional[np.ndarray]) -> Optional[HandDetection]:
        """Detect hand landmarks in a frame.

        Args:
            frame: RGB image (H, W, 3) or None

        Returns:
            HandDetection with landmarks, handedness ("Left"/"Right"), and confidence.
            Returns None if no hand found.
        """
        if frame is None:
            return None

        # frame is already RGB (from camera/NP array), pass directly as SRGB
        mp_image = ImageClass(image_format=ImageFormat.SRGB, data=frame)

        results = self._detector.detect(mp_image)

        if not results.hand_landmarks:
            return None

        # Get first hand
        hand_landmarks = results.hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        # Get handedness
        if results.handedness and len(results.handedness) > 0:
            handedness_list = results.handedness[0]
            if handedness_list:
                handedness = handedness_list[0].category_name  # "Left" or "Right"
                confidence = handedness_list[0].score
            else:
                handedness = "Unknown"
                confidence = 0.0
        else:
            handedness = "Unknown"
            confidence = 0.0

        return HandDetection(
            landmarks=np.array(landmarks, dtype=np.float32),
            handedness=handedness,
            confidence=confidence
        )

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
