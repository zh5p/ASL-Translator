# ASL Translator

Real-time American Sign Language (ASL) to text translation using MediaPipe hand tracking and machine learning.

## Requirements

- Python 3.12
- Webcam
- Windows/Linux/macOS

## Setup

### 1. Install Python 3.12

Download from https://www.python.org/downloads/python-3.12/

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate venv

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If sklearn build fails:
```bash
pip install scikit-learn --only-binary :all:
```

### 5. Run the App

```bash
python run_app.py
```

## Directory Structure

```
ASL_Translator/
├── run_app.py          # Entry point
├── requirements.txt   # Python dependencies
├── configs/            # App configuration
├── data/               # MS-ASL dataset + word dictionary
│   ├── MS-ASL/        # ASL sign data
│   └── word_dictionary.txt
├── models/            # Trained ML models
│   ├── xgboost_model.json
│   ├── lstm_word_model.pt
│   ├── lstm_ms_asl_25words.pt
│   └── lstm_ms_asl_25words_classes.json
├── src/
│   ├── camera/        # Webcam capture
│   ├── detection/     # MediaPipe hand tracking
│   ├── model/         # LSTM + XGBoost inference
│   ├── preprocessing/ # Landmark normalization
│   ├── ui/            # PyQt6 interface
│   │   └── models/    # LSTM word model files (copy from models/ if missing)
│   └── word_matching/ # Word prediction & smoothing
└── APP/               # Portable build (pre-built exe)
```

## Models

- **Letter Model**: XGBoost classifier for ASL alphabet (A-Z, space, nothing)
- **Word Model**: LSTM for word-level recognition

Pretrained models included in `models/`. LSTM word model also needed in `src/ui/models/`.

## Features

- Real-time hand tracking via MediaPipe
- Letter recognition (A-Z)
- Word prediction with sequence modeling
- EMA smoothing for stable output
- Stats panel showing confidence levels

## Troubleshooting

**Webcam not detected:**
- Check camera permissions
- Try restarting the app
- Verify no other app is using webcam

**Model load errors (LSTM not found):**
- LSTM models auto-load from `src/ui/models/`
- If missing, copy `lstm_ms_asl_25words.pt` and `lstm_ms_asl_25words_classes.json` from `models/` to `src/ui/models/`

**Import errors:**
- Verify venv is activated
- Run `pip install -r requirements.txt` again
