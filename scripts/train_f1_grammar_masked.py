from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.training.loops import train_model


if __name__ == "__main__":
    train_model(
        model_name="grammar_vae_masked",
        data_path=os.path.join(ROOT, "datasets", "f1", "f1_dataset.txt"),
        output_root=os.path.join(ROOT, "models", "f1"),
    )
