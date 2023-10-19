import os

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
