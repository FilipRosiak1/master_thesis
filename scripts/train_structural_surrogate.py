from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from f1vae.config.io import maybe_resolve_path
from f1vae.features import FEATURE_NAMES, featurize_genotypes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a structural-feature fitness surrogate for Framsticks f1")
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--output-dir", default="models/f1_surrogate/structural_ensemble")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=600)
    parser.add_argument("--ensemble-size", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--top-fracs", default="0.01,0.05,0.10")
    parser.add_argument("--max-items", type=int, default=None)
    return parser


def _read_dataset(path: str, max_items: int | None) -> tuple[list[str], np.ndarray]:
    genotypes: list[str] = []
    fitnesses: list[float] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row or "\t" not in row:
                continue
            genotype, raw_fitness = row.split("\t", 1)
            try:
                fitness = float(raw_fitness.strip())
            except ValueError:
                continue
            if math.isnan(fitness) or math.isinf(fitness):
                continue
            genotypes.append("".join(genotype.split()))
            fitnesses.append(fitness)
            if max_items is not None and len(genotypes) >= max_items:
                break
    if not genotypes:
        raise RuntimeError("No finite fitness rows found")
    return genotypes, np.asarray(fitnesses, dtype=np.float32)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique_values, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique_values
    sums = np.zeros(len(counts), dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return sums[inverse] / counts[inverse]


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    true_rank = _rankdata(y_true)
    pred_rank = _rankdata(y_pred)
    if np.std(true_rank) < 1e-8 or np.std(pred_rank) < 1e-8:
        return 0.0
    return float(np.corrcoef(true_rank, pred_rank)[0, 1])


def _top_recall(y_true: np.ndarray, y_pred: np.ndarray, frac: float) -> float:
    count = max(1, int(round(len(y_true) * frac)))
    true_top = set(np.argsort(y_true)[-count:].tolist())
    pred_top = set(np.argsort(y_pred)[-count:].tolist())
    return len(true_top & pred_top) / max(1, len(true_top))


def _top_precision(y_true: np.ndarray, y_pred: np.ndarray, frac: float) -> float:
    return _top_recall(y_true, y_pred, frac)


def _ensemble_predict(models: list[Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.vstack([model.predict(x) for model in models])
    return predictions.mean(axis=0), predictions.std(axis=0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, top_fracs: list[float]) -> dict[str, float]:
    out: dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": _spearman(y_true, y_pred),
    }
    for frac in top_fracs:
        key = str(frac).replace(".", "p")
        out[f"top_{key}_recall"] = _top_recall(y_true, y_pred, frac)
        out[f"top_{key}_precision"] = _top_precision(y_true, y_pred, frac)
    return out


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def _write_feature_importance(path: Path, models: list[Any]) -> None:
    importances = []
    for model in models:
        if hasattr(model, "feature_importances_"):
            importances.append(np.asarray(model.feature_importances_, dtype=np.float64))
    if not importances:
        return
    values = np.vstack(importances)
    rows = []
    for name, mean, std in zip(FEATURE_NAMES, values.mean(axis=0), values.std(axis=0)):
        rows.append({"feature": name, "importance_mean": float(mean), "importance_std": float(std)})
    rows.sort(key=lambda row: row["importance_mean"], reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "importance_mean", "importance_std"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _build_parser().parse_args()
    data_path = maybe_resolve_path(args.data_path, root_dir=str(ROOT))
    output_dir = Path(maybe_resolve_path(args.output_dir, root_dir=str(ROOT)))
    output_dir.mkdir(parents=True, exist_ok=True)
    top_fracs = [float(item.strip()) for item in args.top_fracs.split(",") if item.strip()]

    genotypes, y = _read_dataset(data_path, args.max_items)
    x = featurize_genotypes(genotypes)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed)
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models: list[Any] = []
    for i in range(args.ensemble_size):
        seed = args.seed + i
        if i % 2 == 0:
            model = ExtraTreesRegressor(
                n_estimators=args.estimators,
                random_state=seed,
                max_depth=args.max_depth,
                min_samples_leaf=1,
                n_jobs=-1,
                bootstrap=False,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=args.estimators,
                random_state=seed,
                max_depth=args.max_depth,
                min_samples_leaf=1,
                n_jobs=-1,
                bootstrap=True,
            )
        print(f"Training ensemble member {i + 1}/{args.ensemble_size}: {type(model).__name__}", flush=True)
        model.fit(x_train, y_train)
        models.append(model)

    pred_test, unc_test = _ensemble_predict(models, x_test)
    pred_train, unc_train = _ensemble_predict(models, x_train)
    del unc_train
    metric_row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_total": len(y),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fitness_min": float(y.min()),
        "fitness_mean": float(y.mean()),
        "fitness_max": float(y.max()),
        "uncertainty_test_mean": float(unc_test.mean()),
        "uncertainty_test_std": float(unc_test.std()),
    }
    metric_row.update({f"train_{k}": v for k, v in _metrics(y_train, pred_train, top_fracs).items()})
    metric_row.update({f"test_{k}": v for k, v in _metrics(y_test, pred_test, top_fracs).items()})

    payload = {
        "models": models,
        "feature_names": FEATURE_NAMES,
        "data_path": data_path,
        "seed": args.seed,
        "test_size": args.test_size,
        "feature_mean": x_train.mean(axis=0),
        "feature_std": np.maximum(x_train.std(axis=0), 1e-8),
        "fitness_mean": float(y_train.mean()),
        "fitness_std": float(max(y_train.std(), 1e-8)),
        "length_mean": float(x_train[:, FEATURE_NAMES.index("length")].mean()),
        "length_std": float(max(x_train[:, FEATURE_NAMES.index("length")].std(), 1e-8)),
    }
    with (output_dir / "surrogate.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    _write_metrics(output_dir / "metrics.csv", metric_row)
    _write_feature_importance(output_dir / "feature_importance.csv", models)
    (output_dir / "metrics.json").write_text(json.dumps(metric_row, indent=2), encoding="utf-8")

    print(f"Test Spearman: {metric_row['test_spearman']:.4f}", flush=True)
    print(f"Test R2: {metric_row['test_r2']:.4f}", flush=True)
    print(f"Wrote: {output_dir / 'surrogate.pkl'}", flush=True)
    print(f"Wrote: {output_dir / 'metrics.csv'}", flush=True)
    print(f"Wrote: {output_dir / 'feature_importance.csv'}", flush=True)


if __name__ == "__main__":
    main()
