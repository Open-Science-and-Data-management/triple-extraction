"""ค่า config กลางของโปรเจกต์"""

import os
from pathlib import Path

MAX_TEXT_CHARS = 500_000  # เกินนี้ → HTTP 413
MODEL_NAME = "Babelscape/rebel-large"
DB_PATH = Path(os.environ.get("TRIPLE_EXTRACTION_DB", "data/jobs.db"))
