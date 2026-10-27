"""PyQt6 main window for ASL Translator with IP camera support."""
import sys
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QDialog,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import cv2
import torch
import time
import os
import json

_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".asl_translator_settings.json")


def _model_base():
    """Get base directory for model files — works in dev and packaged exe."""
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


from src.camera import CameraCapture
from src.detection import MediaPipeDetector
from src.preprocessing import LandmarkPreprocessor
import xgboost as xgb
from configs.model_config import (
    NUM_CLASSES,
    INPUT_DIM,
    CONFIDENCE_THRESHOLD,
)
from src.word_matching import WordDictionary, LetterBuffer
from src.word_matching.word_predictor import WordPredictor, PredictState


class ConnectionDialog(QDialog):
    """Dialog to enter IP camera URL before connecting."""

    def __init__(self):
        super().__init__()
        self.ip_url = None
        self.model_path = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Connect to Camera")
        self.setFixedSize(500, 180)

        layout = QVBoxLayout()

        # Camera source
        layout.addWidget(QLabel("Camera Source:"))
        self.camera_type = "local"  # or "ip"
        self.local_btn = QPushButton("Local Webcam")
        self.local_btn.clicked.connect(lambda: self._select_source("local"))
        layout.addWidget(self.local_btn)

        ip_layout = QHBoxLayout()
        self.ip_btn = QPushButton("IP Camera")
        self.ip_btn.clicked.connect(lambda: self._select_source("ip"))
        ip_layout.addWidget(self.ip_btn)
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("http://192.168.1.187:8080 or rtsp://...")
        self.ip_input.setEnabled(False)
        ip_layout.addWidget(self.ip_input, stretch=1)
        layout.addLayout(ip_layout)

        # Model file
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_input = QLineEdit("models/xgboost_model.json")
        self.model_input.setEnabled(False)
        model_layout.addWidget(self.model_input, stretch=1)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_model)
        self.browse_btn.setEnabled(False)
        model_layout.addWidget(self.browse_btn)
        layout.addLayout(model_layout)

        # Connect button
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._connect)
        layout.addWidget(self.connect_btn)

        self.setLayout(layout)

    def _select_source(self, source):
        self.camera_type = source
        if source == "ip":
            self.ip_input.setEnabled(True)
            self.ip_input.setFocus()
        else:
            self.ip_input.setEnabled(False)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "models/", "All (*.pt *.pth *.json);;PyTorch (*.pt *.pth);;XGBoost (*.json)")
        if path:
            self.model_input.setText(path)

    def _connect(self):
        if self.camera_type == "ip":
            ip = self.ip_input.text().strip()
            if not ip:
                QMessageBox.warning(self, "Error", "Enter IP camera URL")
                return
            # Auto-add scheme if missing
            if not ip.startswith(("http://", "rtsp://", "https://")):
                ip = f"http://{ip}"
            self.ip_url = ip
        else:
            self.ip_url = None

        self.model_path = self.model_input.text().strip() or None
        self.accept()


class ASLTranslatorWindow(QMainWindow):
    """Main application window for ASL fingerspelling."""

    @staticmethod
    def get_qapplication(argv):
        """Get or create QApplication instance."""
        app = QApplication.instance()
        if app is None:
            app = QApplication(argv)
        return app

    def __init__(
        self,
        ip_url: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__()
        self.setWindowTitle("ASL Fingerspelling Translator")
        self.setGeometry(100, 100, 800, 600)

        self._camera: Optional[CameraCapture] = None
        self._detector: Optional[MediaPipeDetector] = None
        self._preprocessor: Optional[LandmarkPreprocessor] = None
        self._xgb_model = None
        self._xgb_classes = None
        self._model_path = model_path
        self._ip_url = ip_url
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._current_word: str = ""
        self._last_letter: str = ""
        self._letter_start_time: float = 0.0
        self._no_hand_start_time: float = 0.0
        self._last_hand_state: bool = False
        self._letter_lock_duration: float = 1.0   # base seconds to hold letter before confirming
        self._space_duration: float = 2.0        # seconds no hand before inserting space
        self._confirmed_letter: str = ""          # last letter added to word — blocked until different letter predicted

        # Word segmenter
        word_dict_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "word_dictionary.txt")
        self._word_dict = WordDictionary(word_dict_path) if os.path.exists(word_dict_path) else None
        self._letter_buffer = LetterBuffer(max_len=15)
        self._word_predictor = WordPredictor(self._word_dict, self._letter_buffer) if self._word_dict else None

        self._lstm_model = None
        self._lstm_classes = None
        self._lstm_buffer = []  # rolling landmark buffer for LSTM
        self._lstm_buffer_max = 50  # max sequence length for LSTM
        self._lstm_min_frames = 25  # minimum frames before LSTM fires (~0.8s at 30fps)
        self._lstm_last_pred_time = 0.0  # last LSTM prediction timestamp
        self._lstm_pred_interval = 1.5  # seconds between LSTM predictions
        self._lstm_locked = False  # lock display after word prediction until hand drops

        self._init_ui()
        self._tutorial_shown = False
        self._mode = "letters"  # "letters" or "words"
        self._load_settings()
        self._init_components()

    def _init_ui(self):
        """Initialize UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self._video_label = QLabel("Camera Feed")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self._video_label)

        from src.ui.stats_panel import StatsPanel

        self._stats_panel = StatsPanel()
        self._stats_panel.start_session()
        self._stats_panel.hide()
        layout.addWidget(self._stats_panel)

        self._word_label = QLabel("Word: ")
        self._word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._word_label.setStyleSheet("font-size: 32px; font-weight: bold; padding: 10px;")
        layout.addWidget(self._word_label)

        self._prediction_label = QLabel("Waiting for input...")
        self._prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prediction_label.setStyleSheet("font-size: 24px; padding: 10px;")
        layout.addWidget(self._prediction_label)

        self._status_label = QLabel("Status: Initializing")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        # Control buttons
        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("Clear Word")
        clear_btn.clicked.connect(self._clear_word)
        btn_layout.addWidget(clear_btn)

        reconnect_btn = QPushButton("Reconnect")
        reconnect_btn.clicked.connect(self._reconnect)
        btn_layout.addWidget(reconnect_btn)

        stats_btn = QPushButton("Stats")
        stats_btn.clicked.connect(self._toggle_stats)
        btn_layout.addWidget(stats_btn)

        self._letters_btn = QPushButton("Letters")
        self._letters_btn.setCheckable(True)
        self._letters_btn.setChecked(True)
        self._letters_btn.clicked.connect(lambda: self._set_mode("letters"))
        btn_layout.addWidget(self._letters_btn)

        self._words_btn = QPushButton("Words")
        self._words_btn.setCheckable(True)
        self._words_btn.setChecked(False)
        self._words_btn.clicked.connect(lambda: self._set_mode("words"))
        btn_layout.addWidget(self._words_btn)

        layout.addLayout(btn_layout)

        # Video label gets all extra vertical space on maximize
        layout.setStretchFactor(self._video_label, 10)
        layout.setStretchFactor(self._stats_panel, 0)
        layout.setStretchFactor(self._word_label, 0)
        layout.setStretchFactor(self._prediction_label, 0)
        layout.setStretchFactor(self._status_label, 0)
        layout.setStretchFactor(btn_layout, 0)

    def _init_components(self):
        """Initialize camera, detector, preprocessor, model."""
        # Camera
        self._camera = CameraCapture(camera_index=0, fps=30, ip_url=self._ip_url)
        if not self._camera.open():
            self._status_label.setText("Status: Camera not available")
            return

        camera_mode = f"IP ({self._ip_url})" if self._ip_url else "webcam"

        # MediaPipe detector (lenient settings for better tracking during signing)
        self._detector = MediaPipeDetector(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3,
            min_hand_presence_confidence=0.4,
        )

        # Preprocessor (single frame)
        self._preprocessor = LandmarkPreprocessor(window_size=1, stride=1)

        # Model — XGBoost (.json) only for production
        self._xgb_model = None
        self._xgb_classes = None

        resolved_model_path = self._model_path
        if resolved_model_path and not os.path.exists(resolved_model_path):
            # Try relative to model base dir (dev: project root, exe: _internal/models/)
            resolved_model_path = os.path.join(_model_base(), "models", os.path.basename(resolved_model_path))
        if resolved_model_path and os.path.exists(resolved_model_path):
            try:
                if resolved_model_path.endswith('.json'):
                    self._xgb_model = xgb.XGBClassifier()
                    self._xgb_model.load_model(resolved_model_path)
                    classes_path = os.path.join(os.path.dirname(resolved_model_path), 'classes.txt')
                    if os.path.exists(classes_path):
                        with open(classes_path) as f:
                            self._xgb_classes = [line.strip() for line in f]
                    else:
                        self._xgb_classes = None
                    self._status_label.setText(f"Status: {camera_mode} + XGBoost loaded ({len(self._xgb_classes or [])} classes)")
                else:
                    raise ValueError("Only XGBoost .json models are supported")
            except Exception as e:
                self._status_label.setText(f"Status: {camera_mode} + model load failed ({e})")
        else:
            self._status_label.setText(f"Status: {camera_mode} + demo mode")

        lstm_model_path = os.path.join(_model_base(), "models", "lstm_ms_asl_25words.pt")
        lstm_classes_path = os.path.join(_model_base(), "models", "lstm_ms_asl_25words_classes.json")
        print(f"[LSTM] model path: {lstm_model_path}, exists: {os.path.exists(lstm_model_path)}", flush=True)
        print(f"[LSTM] classes path: {lstm_classes_path}, exists: {os.path.exists(lstm_classes_path)}", flush=True)
        if os.path.exists(lstm_model_path) and os.path.exists(lstm_classes_path):
            print(f"[LSTM] Loading model from {lstm_model_path}", flush=True)
            try:
                from src.model import LSTMWordClassifier
                self._lstm_model = LSTMWordClassifier(input_dim=63, hidden_dim=128, num_layers=2, dropout=0.3, num_classes=25)
                self._lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=self._device, weights_only=True))
                self._lstm_model.to(self._device)
                self._lstm_model.eval()
                with open(lstm_classes_path) as f:
                    self._lstm_classes = json.load(f)
                current_status = self._status_label.text()
                self._status_label.setText(f"{current_status} + LSTM ({len(self._lstm_classes['word_to_idx'])} words)")
                print(f"[LSTM] Loaded successfully, {len(self._lstm_classes['word_to_idx'])} words", flush=True)
            except Exception as e:
                print(f"[LSTM] Failed to load: {e}", flush=True)
                pass  # LSTM is optional, don't fail if it doesn't load

        # Start frame timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_frame)
        self._timer.start(33)  # ~30 FPS

        # Update status to include mode
        mode_str = "Letters (XGBoost)" if self._mode == "letters" else "Words (LSTM)"
        current = self._status_label.text()
        if "Mode:" not in current:
            self._status_label.setText(f"{current} | Mode: {mode_str}")

    def _load_settings(self):
        if os.path.exists(_SETTINGS_PATH):
            try:
                with open(_SETTINGS_PATH) as f:
                    settings = json.load(f)
                    self._mode = settings.get("mode", "letters")
                    self._tutorial_shown = settings.get("tutorial_shown", False)
            except Exception:
                self._mode = "letters"
                self._tutorial_shown = False
        else:
            self._mode = "letters"
            self._tutorial_shown = False

    def _save_settings(self):
        settings = {"mode": self._mode, "tutorial_shown": self._tutorial_shown}
        try:
            with open(_SETTINGS_PATH, "w") as f:
                json.dump(settings, f)
        except Exception:
            pass

    def _set_mode(self, mode: str):
        """Switch between Letters (XGBoost) and Words (LSTM) mode."""
        if mode not in ("letters", "words"):
            return
        if self._mode == mode:
            return
        self._mode = mode
        self._letters_btn.setChecked(mode == "letters")
        self._words_btn.setChecked(mode == "words")
        self._clear_word()
        self._lstm_locked = False
        self._save_settings()
        # Update status label
        mode_str = "Letters (XGBoost)" if mode == "letters" else "Words (LSTM)"
        current = self._status_label.text()
        if "Mode:" in current:
            current = current.split("Mode:")[0].strip()
        self._status_label.setText(f"{current} | Mode: {mode_str}")

    def _update_frame(self):
        """Capture and process frame every timer tick."""
        frame = self._camera.read_frame()
        if frame is None:
            return

        now = time.time()
        hand_detected = False

        # Process landmarks
        landmarks = None
        if self._detector is not None:
            detection = self._detector.detect_landmarks(frame)
            if detection is not None:
                hand_detected = True
                frame = self._draw_landmarks(frame, detection.landmarks)
                landmarks = detection.landmarks

        # Feed to preprocessor always (it fills gaps)
        if self._preprocessor is not None:
            window = self._preprocessor.add_frame(landmarks)

        # Hand state change: hand just appeared
        if hand_detected and not self._last_hand_state:
            self._no_hand_start_time = 0.0  # reset space timer

        # Hand state change: hand just disappeared
        if not hand_detected and self._last_hand_state:
            self._no_hand_start_time = now
            # If we were mid-letter-lock, cancel it
            self._letter_start_time = 0.0
            # Reset confirmed letter so same letter can be re-signed after hand returns
            self._confirmed_letter = ""

        self._last_hand_state = hand_detected

        # Space insertion: no hand for _space_duration seconds
        if not hand_detected and self._no_hand_start_time > 0:
            if (now - self._no_hand_start_time) >= self._space_duration:
                if len(self._current_word) > 0 and not self._current_word.endswith(" "):
                    self._current_word += " "
                    self._word_label.setText(f"Word: {self._current_word}")
                self._no_hand_start_time = now  # reset so we don't keep adding spaces

        # Predict when hand detected and model available
        letter = "?"
        confidence = 0.0
        confirmed = False
        word_predicted = None
        # Skip LSTM when locked (showing a word until hand drops)
        if hand_detected and window is not None and not self._lstm_locked:
            if hasattr(self, '_lstm_buffer') and self._lstm_model is not None and self._preprocessor is not None:
                latest = self._preprocessor._buffer[-1] if self._preprocessor._buffer else None
                if latest is not None:
                    self._lstm_buffer.append(torch.from_numpy(latest.ravel()).float())
                    if len(self._lstm_buffer) > self._lstm_buffer_max:
                        self._lstm_buffer.pop(0)
                    if len(self._lstm_buffer) == self._lstm_min_frames:
                        print(f"[LSTM] buffer filled to {self._lstm_min_frames}", flush=True)

            if self._mode == "letters" and self._xgb_model is not None:
                letter, confidence, confirmed = self._predict(window, now)
            elif self._mode == "words" and self._lstm_model is not None:
                word_predicted = self._try_lstm_word(now)
        else:
            # No hand — reset prediction state so meter stops
            letter = "?"
            confidence = 0.0
            confirmed = False
            self._letter_start_time = 0.0
            # Clear LSTM buffer and lock when hand lost
            if hasattr(self, '_lstm_buffer'):
                self._lstm_buffer.clear()
            self._lstm_locked = False

        if confirmed:
            self._prediction_label.setText(f"✓ {letter} ({confidence:.2f})")
            self._prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: green; padding: 10px;")
        elif hand_detected and letter != "?":
            self._prediction_label.setText(f"  {letter} ({confidence:.2f})")
            self._prediction_label.setStyleSheet("font-size: 24px; color: gray; padding: 10px;")
        else:
            self._prediction_label.setText("Waiting for input...")
            self._prediction_label.setStyleSheet("font-size: 24px; color: gray; padding: 10px;")

        # Handle LSTM word prediction result
        if word_predicted is not None:
            word, word_conf = word_predicted
            self._word_label.setText(f"Word: {word.upper()}")
            self._current_word = word.upper() + " "
            self._prediction_label.setText(f"[LSTM: {word} ({word_conf:.2f})]")
            self._prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: green; padding: 10px;")
            self._stats_panel.record_word()
            self._letter_buffer.reset()
            self._lstm_buffer.clear()
            self._lstm_locked = True
            confirmed = False

        # Display frame
        h, w, c = frame.shape
        bytes_per_line = c * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        label_w = self._video_label.width()
        label_h = self._video_label.height()
        if label_w > 0 and label_h > 0:
            scaled = q_image.scaled(label_w, label_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self._video_label.setPixmap(QPixmap.fromImage(scaled))
        else:
            self._video_label.setPixmap(QPixmap.fromImage(q_image))

    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw 21 hand landmarks and connections on frame.

        Args:
            frame: RGB image (H, W, 3)
            landmarks: ndarray of shape (21, 3) with normalized x, y, z values

        Returns:
            frame with landmarks drawn
        """
        h, w = frame.shape[:2]

        # Hand landmark connections (MediaPipe hand skeleton)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),          # index
            (0, 9), (9, 10), (10, 11), (11, 12),     # middle
            (0, 13), (13, 14), (14, 15), (15, 16),   # ring
            (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
            (5, 9), (9, 13), (13, 17),               # palm
        ]

        # Draw connections
        for start, end in connections:
            x1 = int(landmarks[start, 0] * w)
            y1 = int(landmarks[start, 1] * h)
            x2 = int(landmarks[end, 0] * w)
            y2 = int(landmarks[end, 1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw landmark points
        for i, (x, y, z) in enumerate(landmarks):
            px = int(x * w)
            py = int(y * h)
            # Wrist is larger, others smaller
            radius = 8 if i == 0 else 4
            color = (0, 255, 255) if i == 0 else (0, 200, 0)
            cv2.circle(frame, (px, py), radius, color, -1)

        return frame

    def _predict(self, window: np.ndarray, now: float) -> tuple:
        """Run inference on a landmark window with letter-lock timing.

        Confirmation time scales with confidence:
        - >0.8: 0.3s  (very fast — model is sure)
        - >0.6: 0.6s
        - >0.4: 0.9s
        - otherwise: 1.0s (base)

        Returns:
            (letter: str, confidence: float, confirmed: bool)
        """
        flat = window.squeeze(0).flatten()  # (63,)

        probs = self._xgb_model.predict_proba(flat.reshape(1, -1))[0]
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        letter = self._xgb_classes[predicted_idx] if self._xgb_classes else chr(ord('A') + predicted_idx)

        # Determine hold time from confidence
        if confidence > 0.8:
            hold_time = 0.3
        elif confidence > 0.6:
            hold_time = 0.6
        elif confidence > 0.4:
            hold_time = 0.9
        else:
            hold_time = self._letter_lock_duration

        # Block re-confirming same letter until user signs a different letter
        if letter == self._confirmed_letter:
            self._letter_start_time = 0.0
            return letter, confidence, False

        if confidence < CONFIDENCE_THRESHOLD:
            self._letter_start_time = 0.0
            return letter, confidence, False

        if letter != self._last_letter:
            self._last_letter = letter
            self._letter_start_time = now
            return letter, confidence, False

        # Same letter — check if hold time elapsed
        confirmed = False
        if self._letter_start_time > 0 and (now - self._letter_start_time) >= hold_time:
            self._current_word += letter
            self._word_label.setText(f"Word: {self._current_word}")
            self._confirmed_letter = letter
            self._letter_start_time = 0.0
            self._last_letter = ""
            confirmed = True
            self._stats_panel.record_letter()

            # Feed confirmed letter to word predictor buffer
            if self._word_predictor and self._letter_buffer:
                self._letter_buffer.add(letter)
                pred_word, pred_state = self._word_predictor.predict()
                if pred_word and pred_state == PredictState.EXACT_WORD:
                    # Exact word match — commit the word
                    word_len_before = len(pred_word)
                    if self._current_word.endswith(pred_word):
                        self._current_word = self._current_word[:-word_len_before] + pred_word + " "
                    else:
                        self._current_word += pred_word + " "
                    self._word_label.setText(f"Word: {self._current_word}")
                    self._stats_panel.record_word()

        return letter, confidence, confirmed

    def _try_lstm_word(self, now: float) -> Optional[tuple]:
        """Run LSTM inference if buffer is full enough.

        Returns:
            (word, confidence) tuple if prediction succeeds, None otherwise.
        """
        if self._lstm_model is None or self._lstm_classes is None:
            return None
        if len(self._lstm_buffer) < self._lstm_buffer_max:  # wait for full sequence (LSTM trained on 50 frames)
            return None
        # Throttle: only predict every _lstm_pred_interval seconds
        if now - self._lstm_last_pred_time < self._lstm_pred_interval:
            return None

        seq = torch.stack(self._lstm_buffer).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._lstm_model(seq)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, predicted_idx = torch.max(probs, dim=0)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()

        if confidence < 0.5:
            return None

        word = self._lstm_classes["idx_to_word"].get(str(predicted_idx), None)
        if word is None:
            print(f"[LSTM] idx {predicted_idx} not in idx_to_word", flush=True)
            return None

        print(f"[LSTM] pred={word} conf={confidence:.3f} buf_len={len(self._lstm_buffer)}", flush=True)
        self._lstm_last_pred_time = now
        return word, confidence

    def _clear_word(self):
        """Clear the current word."""
        self._current_word = ""
        self._last_letter = ""
        self._confirmed_letter = ""
        self._letter_start_time = 0.0
        self._no_hand_start_time = 0.0
        if hasattr(self, '_letter_buffer'):
            self._letter_buffer.reset()
        if hasattr(self, '_lstm_buffer'):
            self._lstm_buffer.clear()
        self._lstm_locked = False
        self._word_label.setText("Word: ")

    def _reconnect(self):
        """Reinitialize camera connection."""
        if self._camera:
            self._camera.close()
        if self._detector:
            self._detector.close()
        self._init_components()

    def _toggle_stats(self):
        if self._stats_panel.isVisible():
            self._stats_panel.hide()
        else:
            self._stats_panel.show()

    def closeEvent(self, event):
        """Clean up resources on close."""
        if self._camera:
            self._camera.close()
        if self._detector:
            self._detector.close()
        event.accept()


def main():
    app = ASLTranslatorWindow.get_qapplication(sys.argv)

    # Show connection dialog first
    dialog = ConnectionDialog()
    dialog.exec()

    # Create main window after dialog closes
    window = ASLTranslatorWindow(ip_url=dialog.ip_url, model_path=dialog.model_path)
    window.show()

    from src.ui.tutorial_overlay import TutorialOverlay

    if not window._tutorial_shown:
        tutorial = TutorialOverlay()
        tutorial.setParent(window)
        tutorial.move(
            (window.width() - tutorial.width()) // 2,
            (window.height() - tutorial.height()) // 2,
        )
        tutorial.show()
        window._tutorial_shown = True
        window._save_settings()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
