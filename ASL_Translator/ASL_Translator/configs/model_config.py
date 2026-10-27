"""Hyperparameters and configuration for ASL fingerspelling."""

# Dataset — 26 letters A-Z
NUM_CLASSES = 26
NUM_LANDMARKS = 21
NUM_COORDS = 3
INPUT_DIM = NUM_LANDMARKS * NUM_COORDS  # 63

# Model
HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 64
DROPOUT = 0.3

# Training
BATCH_SIZE = 256  # Larger batch for fast GPU training
EPOCHS = 30
LEARNING_RATE = 1e-3

# Inference
CONFIDENCE_THRESHOLD = 0.3

# Paths
DATA_DIR = "data/asl_fingerspelling"
LANDMARKS_DIR = "data/landmarks"           # bounding-box normalized
LANDMARKS_DIR_WRIST = "data/landmarks_wrist"  # wrist-centered normalized
MODEL_CHECKPOINT_PATH = "models/checkpoints/model.pt"
MODEL_SAVE_PATH = "models/asl_mlp_model.pt"
