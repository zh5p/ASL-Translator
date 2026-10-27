import torch.nn as nn

# LSTM word model for word-level ASL recognition
class LSTMWordClassifier(nn.Module):
    """Bidirectional LSTM for word-level ASL recognition from landmark sequences."""

    def __init__(self, input_dim=63, hidden_dim=128, num_layers=2, dropout=0.3, num_classes=30):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)
