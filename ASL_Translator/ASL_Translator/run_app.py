r"""Launcher for ASL Translator app. Run from project root.

Usage:
    # Use bundled venv (recommended):
    venv\Scripts\python.exe run_app.py

    # Or activate venv first:
    venv\Scripts\activate
    python run_app.py
"""
import sys
import os

# Check if running from correct venv
venv_python = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "venv", "Scripts" if sys.platform == "win32" else "bin", "python.exe")
)
current_python = os.path.abspath(sys.executable)

if os.path.exists(venv_python) and current_python != venv_python:
    print(f"WARNING: Not using venv Python.")
    print(f"  Current:  {current_python}")
    print(f"  Expected: {venv_python}")
    print(f"  Continuing anyway...")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ui.window import main

if __name__ == "__main__":
    main()
