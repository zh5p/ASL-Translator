"""Webcam or IP camera capture."""
import cv2
from typing import Optional, Union
import numpy as np


class CameraCapture:
    """Captures frames from webcam or IP camera.

    Usage:
        # Local webcam
        cam = CameraCapture(camera_index=0)

        # IP camera via URL
        cam = CameraCapture(ip_url="rtsp://192.168.1.100:8080/video")

    Supported IP camera URLs:
        - rtsp://<ip>:<port>/<path>
        - http://<ip>:<port>/<path>
        - https://<ip>:<port>/<path>
    """

    def __init__(
        self,
        camera_index: int = 0,
        fps: int = 30,
        ip_url: Optional[str] = None,
    ):
        self.camera_index = camera_index
        self.fps = fps
        self.ip_url = ip_url
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if self.ip_url:
            self._cap = cv2.VideoCapture(self.ip_url)
        else:
            self._cap = cv2.VideoCapture(self.camera_index)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        return self._cap is not None and self._cap.isOpened()

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Optional[np.ndarray]:
        if not self.is_opened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
