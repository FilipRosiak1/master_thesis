from __future__ import annotations

from collections import Counter
from math import log1p

import numpy as np


MODIFIERS = "RrQqCcLlWwMmIiFfAaSsEe"
CONTROL_CHARS = "()[],"
BODY_CHARS = "X"
KNOWN_CHARS = set(MODIFIERS + CONTROL_CHARS + BODY_CHARS)

FEATURE_NAMES = [
    "length",
    "log_length",
    "segments",
    "log_segments",
    "branch_open",
    "branch_close",
    "commas",
    "empty_branch_slots",
    "max_depth",
    "mean_depth",
    "final_depth",
    "min_depth",
    "unmatched_close",
    "bracket_open",
    "bracket_close",
    "bracket_unmatched_close",
    "final_bracket_depth",
    "modifier_total",
    "modifier_fraction",
    "longest_modifier_run",
    "prefix_modifier_run",
    "suffix_modifier_run",
    "unknown_chars",
    "unique_chars",
    "x_fraction",
    "comma_fraction",
    "branch_fraction",
    "control_fraction",
    "avg_chars_per_segment",
]

for _char in MODIFIERS:
    FEATURE_NAMES.append(f"count_{_char}")
for _char in MODIFIERS:
    FEATURE_NAMES.append(f"frac_{_char}")


def _normalize(genotype: str) -> str:
    return "".join(str(genotype).split())


def _depth_features(genotype: str) -> tuple[int, float, int, int]:
    depth = 0
    min_depth = 0
    max_depth = 0
    depth_sum = 0
    n = 0
    for char in genotype:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth -= 1
            min_depth = min(min_depth, depth)
        if char not in "[]":
            depth_sum += max(0, depth)
            n += 1
    return max_depth, depth_sum / max(1, n), depth, min_depth


def _bracket_features(genotype: str) -> tuple[int, int]:
    depth = 0
    unmatched_close = 0
    for char in genotype:
        if char == "[":
            depth += 1
        elif char == "]":
            if depth <= 0:
                unmatched_close += 1
            else:
                depth -= 1
    return unmatched_close, depth


def _longest_modifier_run(genotype: str) -> int:
    longest = 0
    current = 0
    for char in genotype:
        if char in MODIFIERS:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _prefix_modifier_run(genotype: str) -> int:
    count = 0
    for char in genotype:
        if char not in MODIFIERS:
            break
        count += 1
    return count


def _suffix_modifier_run(genotype: str) -> int:
    count = 0
    for char in reversed(genotype):
        if char not in MODIFIERS:
            break
        count += 1
    return count


def _empty_branch_slots(genotype: str) -> int:
    if not genotype:
        return 0
    count = 0
    for left, right in zip(genotype, genotype[1:]):
        if left == "(" and right in ",)":
            count += 1
        elif left == "," and right in ",)":
            count += 1
    return count


def extract_f1_features(genotype: str) -> np.ndarray:
    genotype = _normalize(genotype)
    length = len(genotype)
    counts = Counter(genotype)
    segments = counts.get("X", 0)
    branch_open = counts.get("(", 0)
    branch_close = counts.get(")", 0)
    commas = counts.get(",", 0)
    bracket_open = counts.get("[", 0)
    bracket_close = counts.get("]", 0)
    modifier_total = sum(counts.get(char, 0) for char in MODIFIERS)
    max_depth, mean_depth, final_depth, min_depth = _depth_features(genotype)
    bracket_unmatched_close, final_bracket_depth = _bracket_features(genotype)
    unknown_chars = sum(1 for char in genotype if char not in KNOWN_CHARS)
    branch_total = branch_open + branch_close
    control_total = branch_total + commas + bracket_open + bracket_close

    base = [
        float(length),
        float(log1p(length)),
        float(segments),
        float(log1p(segments)),
        float(branch_open),
        float(branch_close),
        float(commas),
        float(_empty_branch_slots(genotype)),
        float(max_depth),
        float(mean_depth),
        float(final_depth),
        float(min_depth),
        float(abs(final_depth) + abs(min_depth)),
        float(bracket_open),
        float(bracket_close),
        float(bracket_unmatched_close),
        float(final_bracket_depth),
        float(modifier_total),
        float(modifier_total / max(1, length)),
        float(_longest_modifier_run(genotype)),
        float(_prefix_modifier_run(genotype)),
        float(_suffix_modifier_run(genotype)),
        float(unknown_chars),
        float(len(counts)),
        float(segments / max(1, length)),
        float(commas / max(1, length)),
        float(branch_total / max(1, length)),
        float(control_total / max(1, length)),
        float(length / max(1, segments)),
    ]
    base.extend(float(counts.get(char, 0)) for char in MODIFIERS)
    base.extend(float(counts.get(char, 0) / max(1, length)) for char in MODIFIERS)
    return np.asarray(base, dtype=np.float32)


def featurize_genotypes(genotypes: list[str]) -> np.ndarray:
    if not genotypes:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([extract_f1_features(genotype) for genotype in genotypes]).astype(np.float32)
