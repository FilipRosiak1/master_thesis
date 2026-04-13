"""Backward-compatible re-export for the refactored grammar module."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.grammars.f1 import D, GCFG, gram, ind_of_ind, lhs_list, masks

__all__ = ["gram", "GCFG", "D", "lhs_list", "masks", "ind_of_ind"]
