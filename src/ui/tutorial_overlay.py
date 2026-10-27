from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


STEPS = [
    ("Show Your Hand", "Point your hand at the camera. The app detects ASL letters in real-time using MediaPipe hand tracking."),
    ("Hold to Confirm", "Hold a letter steady to confirm it. Higher confidence = faster confirmation (0.3s to 1.0s)."),
    ("Pause for Space", "Pause for 2 seconds to insert a space between words."),
    ("Word Auto-Confirm", "The dictionary auto-confirms common words as you spell them. Just keep signing!"),
]


class TutorialOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_step = 0
        self._init_ui()

    def _init_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: rgba(0,0,0,0.85);")
        self.setFixedSize(600, 400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        self._step_label = QLabel()
        self._step_label.setFont(QFont("Arial", 11))
        self._step_label.setStyleSheet("color: #4fc3f7;")
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._step_label)

        self._title_label = QLabel()
        self._title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self._title_label.setStyleSheet("color: white;")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title_label)

        self._desc_label = QLabel()
        self._desc_label.setFont(QFont("Arial", 14))
        self._desc_label.setStyleSheet("color: #e0e0e0;")
        self._desc_label.setWordWrap(True)
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._desc_label)

        layout.addStretch()

        nav_layout = QHBoxLayout()

        self._skip_btn = QPushButton("Skip Tutorial")
        self._skip_btn.setStyleSheet("background-color: #424242; color: #e0e0e0; padding: 8px 16px;")
        self._skip_btn.clicked.connect(self.close)
        nav_layout.addWidget(self._skip_btn)

        nav_layout.addStretch()

        self._prev_btn = QPushButton("Previous")
        self._prev_btn.setStyleSheet("background-color: #424242; color: #e0e0e0; padding: 8px 16px;")
        self._prev_btn.clicked.connect(self._prev_step)
        nav_layout.addWidget(self._prev_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.setStyleSheet("background-color: #4fc3f7; color: #1e1e1e; padding: 8px 16px; font-weight: bold;")
        self._next_btn.clicked.connect(self._next_step)
        nav_layout.addWidget(self._next_btn)

        layout.addLayout(nav_layout)

        self._update_step()

    def _update_step(self):
        step_num, (title, desc) = self._current_step + 1, STEPS[self._current_step]
        self._step_label.setText(f"Step {step_num} of {len(STEPS)}")
        self._title_label.setText(title)
        self._desc_label.setText(desc)
        self._prev_btn.setEnabled(self._current_step > 0)
        self._next_btn.setText("Finish" if self._current_step == len(STEPS) - 1 else "Next")

    def _prev_step(self):
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step()

    def _next_step(self):
        if self._current_step < len(STEPS) - 1:
            self._current_step += 1
            self._update_step()
        else:
            self.close()
