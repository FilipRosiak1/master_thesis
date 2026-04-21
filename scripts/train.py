from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.experiments.train import main
from f1vae.utils import command_run_logger


if __name__ == "__main__":
    with command_run_logger("scripts/train.py"):
        main()
