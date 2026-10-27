from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QTimer
import time


class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._session_start = None
        self._total_letters = 0
        self._total_words = 0
        self._total_predictions = 0
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("Session Stats")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")
        layout.addWidget(title)

        self._session_label = QLabel("Time: 00:00:00")
        self._session_label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(self._session_label)

        self._letters_label = QLabel("Letters: 0")
        self._letters_label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(self._letters_label)

        self._words_label = QLabel("Words: 0")
        self._words_label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(self._words_label)

        self._wpm_label = QLabel("WPM: 0")
        self._wpm_label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(self._wpm_label)

        self._accuracy_label = QLabel("Accuracy: 0%")
        self._accuracy_label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(self._accuracy_label)

        self.setStyleSheet("background-color: #2d2d2d; color: #e0e0e0;")

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_display)
        self._timer.start(1000)

    def start_session(self):
        self._session_start = time.time()

    def record_letter(self):
        self._total_letters += 1
        self._total_predictions += 1

    def record_word(self):
        self._total_words += 1

    def record_prediction(self):
        self._total_predictions += 1

    def _update_display(self):
        if self._session_start is None:
            return
        elapsed = time.time() - self._session_start
        hh = int(elapsed // 3600)
        mm = int((elapsed % 3600) // 60)
        ss = int(elapsed % 60)
        self._session_label.setText(f"Time: {hh:02d}:{mm:02d}:{ss:02d}")
        self._letters_label.setText(f"Letters: {self._total_letters}")
        self._words_label.setText(f"Words: {self._total_words}")
        minutes = elapsed / 60.0
        wpm = int((self._total_letters / 5.0) / minutes) if minutes > 0 else 0
        self._wpm_label.setText(f"WPM: {wpm}")
        acc = int(100 * self._total_letters / self._total_predictions) if self._total_predictions > 0 else 0
        self._accuracy_label.setText(f"Accuracy: {acc}%")
