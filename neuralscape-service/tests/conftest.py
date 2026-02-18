"""Conftest for neuralscape-service tests — adds parent dir to sys.path."""

import sys
from pathlib import Path

# Ensure the service root is importable (main.py, config.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
