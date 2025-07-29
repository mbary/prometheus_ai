import os
from pathlib import Path
import shutil
DATA_DIR = Path(".audio/recorded/")
POS_DIR = DATA_DIR / "positive"
NEG_DIR = DATA_DIR / "negative"

POS_DIR.mkdir(parents=True, exist_ok=True)
NEG_DIR.mkdir(parents=True, exist_ok=True)

pos_files = [f for f in os.listdir(DATA_DIR) if f.endswith("1.wav")]
neg_files = [f for f in os.listdir(DATA_DIR) if f.endswith("0.wav")]

pos_files[:10], neg_files[:10]

len(pos_files), len(neg_files)

for file in pos_files:
    shutil.move(DATA_DIR / file, POS_DIR / file)

for file in neg_files:
    shutil.move(DATA_DIR / file, NEG_DIR / file)

len(os.listdir(POS_DIR))