from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


INVALID_FITNESS = -999999.0
_WINDOWS_FLOAT_EXCEPTION_MASK = 0x0008001F


def mask_floating_point_exceptions() -> None:
    """Restore Python/PyTorch-friendly floating point control flags after Framsticks calls."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.CDLL("msvcrt")._controlfp(
            _WINDOWS_FLOAT_EXCEPTION_MASK,
            _WINDOWS_FLOAT_EXCEPTION_MASK,
        )
    except Exception:
        return


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_framsticks_path() -> Path:
    return _repo_root() / "src" / "framsticks" / "Framsticks54"


def _default_sim_path() -> Path:
    return _repo_root() / "src" / "framsticks" / "framspy" / "eval-allcriteria.sim"


def _resolve_sim_settings(sim: str | Path) -> str:
    root = _repo_root()
    resolved: list[str] = []
    for item in str(sim).split(";"):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute():
            path = (root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Framsticks simulation settings file was not found. Expected: {path}.")
        resolved.append(str(path))
    if not resolved:
        raise ValueError("At least one Framsticks simulation settings file is required")
    return ";".join(resolved)


class FramsticksFitness:
    def __init__(
        self,
        frams_path: str | Path | None = None,
        *,
        lib: str | None = None,
        sim: str | Path | None = None,
        criterion: str = "vertpos",
        deterministic: bool = False,
    ) -> None:
        self.frams_path = Path(frams_path) if frams_path is not None else _default_framsticks_path()
        self.lib = lib
        self.sim = str(sim) if sim is not None else str(_default_sim_path())
        self.criterion = criterion
        self.deterministic = deterministic
        self._frams_lib = None

    def _ensure_loaded(self):
        if self._frams_lib is not None:
            return self._frams_lib

        framspy_path = _repo_root() / "src" / "framsticks" / "framspy"
        if str(framspy_path) not in sys.path:
            sys.path.insert(0, str(framspy_path))

        framsticks_lib_py = framspy_path / "FramsticksLib.py"
        if not framsticks_lib_py.exists():
            raise FileNotFoundError(
                "FramsticksLib.py was not found. Expected: "
                f"{framsticks_lib_py}. Make sure src/framsticks/framspy exists on this machine."
            )
        if not self.frams_path.exists():
            raise FileNotFoundError(
                "Framsticks runtime directory was not found. Expected: "
                f"{self.frams_path}. Make sure src/framsticks/Framsticks54 exists on this machine."
            )
        sim_settings = _resolve_sim_settings(self.sim)

        from FramsticksLib import FramsticksLib

        FramsticksLib.DETERMINISTIC = self.deterministic
        self._frams_lib = FramsticksLib(str(self.frams_path), self.lib, sim_settings)
        mask_floating_point_exceptions()
        return self._frams_lib

    def evaluate_one(self, genotype: str) -> float | None:
        values = self.evaluate_many([genotype])
        return values[0] if values else None

    def evaluate_many(self, genotypes: Iterable[str]) -> list[float | None]:
        genotypes = list(genotypes)
        if not genotypes:
            return []

        frams_lib = self._ensure_loaded()
        try:
            data = frams_lib.evaluate(genotypes)
        except Exception:
            if len(genotypes) == 1:
                return [None]
            return [self.evaluate_one(genotype) for genotype in genotypes]
        finally:
            mask_floating_point_exceptions()

        values: list[float | None] = []
        for item in data:
            try:
                value = item["evaluations"][""][self.criterion]
            except (KeyError, TypeError):
                values.append(None)
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                values.append(None)
                continue
            if value == INVALID_FITNESS:
                values.append(None)
            else:
                values.append(value)
        return values


_DEFAULT_EVALUATOR: FramsticksFitness | None = None


def _default_evaluator() -> FramsticksFitness:
    global _DEFAULT_EVALUATOR
    if _DEFAULT_EVALUATOR is None:
        _DEFAULT_EVALUATOR = FramsticksFitness(
            frams_path=os.environ.get("FRAMSTICKS_PATH"),
            lib=os.environ.get("FRAMSTICKS_LIB") or None,
            sim=os.environ.get("FRAMSTICKS_SIM"),
            criterion=os.environ.get("FRAMSTICKS_CRITERION", "vertpos"),
            deterministic=os.environ.get("FRAMSTICKS_DETERMINISTIC", "0") in {"1", "true", "True"},
        )
    return _DEFAULT_EVALUATOR


def vertpos(genotype: str) -> float:
    value = _default_evaluator().evaluate_one(genotype)
    return INVALID_FITNESS if value is None else value
