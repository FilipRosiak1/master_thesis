from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ObjectiveFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class OptimizationResult:
    best_z: np.ndarray
    best_score: float
    history_best_score: list[float]
