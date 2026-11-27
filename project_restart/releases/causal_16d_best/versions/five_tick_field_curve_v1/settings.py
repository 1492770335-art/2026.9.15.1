import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("TULIN_RUN_DIR", "five_tick_run")).resolve()
CACHE = Path(os.environ.get("TULIN_CACHE", ROOT / "cache")).resolve()
TAPE = Path(os.environ.get("TULIN_ROLLING", ROOT / "rolling")).resolve()
UNITS = np.array([0.160943 / 0.558942, 0.160943 / 0.272164, 1.0])
