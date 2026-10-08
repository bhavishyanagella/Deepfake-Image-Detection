
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "4"))
MODEL_PATH = ROOT / "models" / "resnet50_lstm_deepfake.keras"
LABELS = {0: "REAL", 1: "FAKE"}
