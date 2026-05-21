from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1vae.config.defaults import DEFAULTS
from f1vae.config.io import maybe_resolve_path
from f1vae.data.datasets import GrammarRuleDataset
from f1vae.fitness.framsticks import FramsticksFitness
from f1vae.inference.checkpoints import load_checkpoint
from f1vae.inference.decode import decode_char_indices, decode_grammar_indices, decode_masked_deterministic
from f1vae.inference.evaluate import reconstruction_metrics
from f1vae.models.registry import build_model, dataset_class_for_model


TREE_MODELS = {"tree_vae", "tree_vae_masked", "tree_vae_masked_lhs", "tree_vae_masked_lhs_depth", "tree_vae_masked_lhs_cond"}
MODIFIERS = set("RrQqCcLlWwMmIiFfAaSsEe")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate selected F1 VAE checkpoints")
    parser.add_argument(
        "--checkpoints-root",
        default="exports/f1_selected_10_ckpts_20260520_184738",
        help="Directory containing selected checkpoint subdirectories",
    )
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    parser.add_argument("--generation-samples", type=int, default=100)
    parser.add_argument("--latent-samples", type=int, default=1000)
    parser.add_argument("--max-recon-items", type=int, default=512, help="0 disables reconstruction pass")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--framsticks-path", default="src/framsticks/Framsticks54")
    parser.add_argument("--framsticks-lib", default=None)
    parser.add_argument("--framsticks-sim", default="src/framsticks/framspy/eval-allcriteria.sim")
    parser.add_argument("--skip-framsticks", action="store_true")
    parser.add_argument("--output", default=None, help="CSV output path; default is inside checkpoints root")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_checkpoint(run_dir: Path) -> Path:
    best = run_dir / "best_val_recon.pth"
    if best.exists():
        return best
    checkpoints = sorted(run_dir.rglob("*.pth"), key=lambda path: path.as_posix())
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoint found under {run_dir}")
    return checkpoints[0]


def _parse_recon_line(line: str, prefix: str) -> dict[str, float | int] | None:
    if not line.startswith(prefix):
        return None
    parts = line.strip().split("\t")
    if len(parts) < 4:
        return None
    try:
        epoch_text = parts[0].split()[1]
        epoch, epochs = [int(value) for value in epoch_text.split("/", 1)]
        exact_text = parts[1].split(":", 1)[1].strip()
        exact, total = [int(value) for value in exact_text.split("/", 1)]
        acc = float(parts[2].split(":", 1)[1].strip().rstrip("%"))
        sim = float(parts[3].split(":", 1)[1].strip().rstrip("%"))
    except (IndexError, ValueError):
        return None
    return {"epoch": epoch, "epochs": epochs, "exact": exact, "total": total, "acc": acc, "sim": sim}


def _parse_training_log(path: Path) -> dict[str, float | int | str | None]:
    best: dict[str, float | int] | None = None
    max_val: dict[str, float | int] | None = None
    final_progress = None
    if not path.exists():
        return {}

    for line in path.read_text(encoding="utf-8").splitlines():
        best_line = _parse_recon_line(line, "BestValRecon")
        if best_line is not None:
            best = best_line
            continue
        val_line = _parse_recon_line(line, "ValRecon")
        if val_line is not None and (
            max_val is None
            or int(val_line["exact"]) > int(max_val["exact"])
            or (int(val_line["exact"]) == int(max_val["exact"]) and float(val_line["sim"]) > float(max_val["sim"]))
        ):
            max_val = val_line
            continue
        if line.startswith("Progress "):
            final_progress = line.split("\t", 1)[0].split()[1]

    metric = best or max_val
    if metric is None:
        return {"log_progress": final_progress}
    return {
        "log_metric": "best" if best is not None else "max_val",
        "log_best_epoch": metric["epoch"],
        "log_epochs": metric["epochs"],
        "log_val_exact": metric["exact"],
        "log_val_total": metric["total"],
        "log_val_exact_acc": metric["acc"],
        "log_val_avg_sim": metric["sim"],
        "log_progress": final_progress,
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _model_config(run_config: dict[str, Any], checkpoint_meta: dict[str, Any]) -> dict[str, Any]:
    model_name = run_config.get("model") or checkpoint_meta.get("model_name")
    if model_name is None:
        raise ValueError("Missing model name in run_config.json and checkpoint metadata")
    defaults = DEFAULTS[model_name]
    return {
        "model_name": model_name,
        "latent_dim": int(run_config.get("latent_dim") or checkpoint_meta.get("latent_dim") or defaults.latent_dim),
        "hidden_dim": run_config.get("hidden_dim") or checkpoint_meta.get("hidden_dim") or defaults.hidden_dim,
        "embedding_dim": run_config.get("embedding_dim") or checkpoint_meta.get("embedding_dim") or defaults.embedding_dim,
        "max_length": int(run_config.get("max_length") or checkpoint_meta.get("max_length") or defaults.max_length),
        "seed": run_config.get("seed"),
        "learning_rate": run_config.get("learning_rate"),
    }


def _dataset_fitnesses(dataset) -> list[float | None]:
    if hasattr(dataset, "valid_fitnesses"):
        return list(dataset.valid_fitnesses)
    if hasattr(dataset, "fitnesses"):
        return list(dataset.fitnesses)
    return [None] * len(dataset)


def _decode_generated(model_name: str, model, z: torch.Tensor, dataset) -> list[str]:
    with torch.no_grad():
        if model_name == "char_vae":
            generated = model.decode_from_latent(z)
            return [decode_char_indices(generated[i], dataset.vocabulary.idx2char) for i in range(generated.size(0))]
        if model_name == "grammar_vae_masked":
            generated = decode_masked_deterministic(model, z)
            return [_safe_decode_grammar(generated[i]) for i in range(generated.size(0))]
        if model_name in TREE_MODELS:
            logits = model.decode(z, None, teacher_forcing_ratio=0.0)
        elif model_name == "vq_grammar_ae":
            logits = model.decoder(z, None, teacher_forcing_ratio=0.0)
        else:
            logits = model.decoder(z, None, teacher_forcing_ratio=0.0)
        generated = logits.argmax(dim=-1)
        return [_safe_decode_grammar(generated[i]) for i in range(generated.size(0))]


def _safe_decode_grammar(indices: torch.Tensor) -> str:
    try:
        return decode_grammar_indices(indices)
    except Exception:
        return ""


def _parse_element(genotype: str, pos: int, stops: set[str]) -> int | None:
    while pos < len(genotype) and genotype[pos] in MODIFIERS:
        pos += 1
    if pos >= len(genotype) or genotype[pos] in stops:
        return None
    if genotype[pos] == "X":
        return pos + 1
    if genotype[pos] != "(":
        return None

    pos += 1
    while pos < len(genotype) and genotype[pos] != ")":
        if genotype[pos] == ",":
            pos += 1
            continue
        pos = _parse_sequence(genotype, pos, {",", ")"})
        if pos is None:
            return None
        if pos < len(genotype) and genotype[pos] == ",":
            pos += 1
    if pos >= len(genotype) or genotype[pos] != ")":
        return None
    return pos + 1


def _parse_sequence(genotype: str, pos: int, stops: set[str]) -> int | None:
    first = _parse_element(genotype, pos, stops)
    if first is None:
        return None
    pos = first
    while pos < len(genotype) and genotype[pos] not in stops:
        pos = _parse_element(genotype, pos, stops)
        if pos is None:
            return None
    return pos


def _is_valid_f1(genotype: str, max_length: int) -> bool:
    del max_length
    genotype = "".join(genotype.split())
    if not genotype:
        return False
    end = _parse_sequence(genotype, 0, set())
    if end is None:
        return False
    return end == len(genotype)


def _generation_metrics(
    model_name: str,
    model,
    dataset,
    device: torch.device,
    *,
    latent_dim: int,
    max_length: int,
    samples: int,
    batch_size: int,
    dataset_genotypes: set[str],
    rng_seed: int,
    framsticks: FramsticksFitness | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if samples <= 0:
        return {}, []

    generator = torch.Generator(device=device)
    generator.manual_seed(rng_seed)
    generated: list[str] = []
    remaining = samples
    while remaining > 0:
        current = min(batch_size, remaining)
        z = torch.randn(current, latent_dim, generator=generator, device=device)
        generated.extend(_decode_generated(model_name, model, z, dataset))
        remaining -= current

    valid_flags = [_is_valid_f1(genotype, max_length) for genotype in generated]
    valid_genotypes = [genotype for genotype, valid in zip(generated, valid_flags) if valid]
    unique_generated = set(generated)
    unique_valid = set(valid_genotypes)
    novel_valid = [genotype for genotype in valid_genotypes if genotype not in dataset_genotypes]

    fitness_by_genotype: dict[str, float | None] = {}
    if framsticks is not None and unique_valid:
        unique_valid_list = sorted(unique_valid)
        values = framsticks.evaluate_many(unique_valid_list)
        fitness_by_genotype = dict(zip(unique_valid_list, values))

    fitness_values = [fitness_by_genotype.get(genotype) for genotype in valid_genotypes]
    numeric_fitness = [value for value in fitness_values if value is not None]
    examples = []
    for genotype, valid in zip(generated[:10], valid_flags[:10]):
        examples.append(
            {
                "genotype": genotype,
                "valid": valid,
                "novel": valid and genotype not in dataset_genotypes,
                "vertpos": fitness_by_genotype.get(genotype),
            }
        )

    metrics = {
        "gen_samples": len(generated),
        "gen_valid": len(valid_genotypes),
        "gen_valid_rate": len(valid_genotypes) / max(1, len(generated)),
        "gen_unique": len(unique_generated),
        "gen_duplicate_rate": 1.0 - (len(unique_generated) / max(1, len(generated))),
        "gen_unique_valid": len(unique_valid),
        "gen_novel_valid": len(novel_valid),
        "gen_novel_valid_rate": len(novel_valid) / max(1, len(valid_genotypes)),
        "gen_vertpos_count": len(numeric_fitness),
        "gen_vertpos_mean": float(np.mean(numeric_fitness)) if numeric_fitness else None,
        "gen_vertpos_max": float(np.max(numeric_fitness)) if numeric_fitness else None,
    }
    return metrics, examples


def _encode_mu(model_name: str, model, batch: torch.Tensor) -> torch.Tensor:
    if model_name == "char_vae":
        mu, _ = model.encoder(batch)
        return mu
    if model_name in TREE_MODELS:
        mu, _ = model.encode(batch)
        return mu
    if model_name == "vq_grammar_ae":
        return model.encoder(batch)
    mu, _ = model.encoder(batch)
    return mu


def _latent_fitness_metrics(
    model_name: str,
    model,
    dataset,
    device: torch.device,
    *,
    samples: int,
    batch_size: int,
    rng: random.Random,
) -> dict[str, Any]:
    fitnesses = _dataset_fitnesses(dataset)
    indices = [idx for idx, fitness in enumerate(fitnesses) if _safe_float(fitness) is not None]
    if samples <= 0 or len(indices) < 3:
        return {"latent_fitness_samples": 0}
    rng.shuffle(indices)
    indices = indices[: min(samples, len(indices))]

    z_values: list[np.ndarray] = []
    y_values: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            batch = torch.stack([dataset[idx] for idx in chunk]).to(device)
            mu = _encode_mu(model_name, model, batch).detach().cpu().numpy()
            z_values.append(mu)
            y_values.extend(float(fitnesses[idx]) for idx in chunk)

    x = np.concatenate(z_values, axis=0)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape[0] < 3:
        return {"latent_fitness_samples": int(x.shape[0])}

    x_std = x.astype(np.float64)
    x_std = (x_std - x_std.mean(axis=0, keepdims=True)) / np.maximum(x_std.std(axis=0, keepdims=True), 1e-8)
    y_centered = y - y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        return {"latent_fitness_samples": int(x.shape[0]), "latent_fitness_error": "zero fitness variance"}

    corrs = (x_std * (y_centered[:, None] / y_std)).mean(axis=0)
    design = np.concatenate([x_std, np.ones((x_std.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, y_centered, rcond=None)
    pred = design @ coef
    sst = float(np.sum(y_centered**2))
    sse = float(np.sum((y_centered - pred) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else None

    return {
        "latent_fitness_samples": int(x.shape[0]),
        "latent_dim": int(x.shape[1]),
        "latent_abs_corr_mean": float(np.mean(np.abs(corrs))),
        "latent_abs_corr_max": float(np.max(np.abs(corrs))),
        "latent_linear_r2_in_sample": r2,
    }


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6g}"
    if value is None:
        return ""
    return value


def main() -> None:
    args = _build_parser().parse_args()
    checkpoints_root = maybe_resolve_path(args.checkpoints_root, root_dir=str(ROOT))
    data_path = maybe_resolve_path(args.data_path, root_dir=str(ROOT))
    checkpoints_root_path = Path(checkpoints_root)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    reference_dataset = GrammarRuleDataset(data_path, max_length=500)
    dataset_genotypes = set(reference_dataset.valid_lines)

    framsticks: FramsticksFitness | None = None
    framsticks_error = ""
    if not args.skip_framsticks:
        try:
            framsticks = FramsticksFitness(
                maybe_resolve_path(args.framsticks_path, root_dir=str(ROOT)),
                lib=args.framsticks_lib,
                sim=maybe_resolve_path(args.framsticks_sim, root_dir=str(ROOT)),
            )
            framsticks.evaluate_one("X")
        except Exception as exc:
            framsticks = None
            framsticks_error = f"{type(exc).__name__}: {exc}"
            print(f"Framsticks unavailable, continuing without generated vertpos: {framsticks_error}")

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"framsticks_error": framsticks_error, "models": {}}

    for run_dir in sorted(path for path in checkpoints_root_path.iterdir() if path.is_dir()):
        print(f"Evaluating {run_dir.name}")
        checkpoint_path = _find_checkpoint(run_dir)
        run_config = _load_json(run_dir / "run_config.json")
        state_dict, checkpoint_meta = load_checkpoint(str(checkpoint_path), device)
        cfg = _model_config(run_config, checkpoint_meta)
        model_name = cfg["model_name"]

        dataset_cls = dataset_class_for_model(model_name)
        dataset = dataset_cls(data_path, cfg["max_length"])
        model = build_model(
            model_name=model_name,
            dataset=dataset,
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            embedding_dim=cfg["embedding_dim"],
            max_length=cfg["max_length"],
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        row: dict[str, Any] = {
            "label": run_dir.name,
            "model": model_name,
            "checkpoint": checkpoint_path.relative_to(run_dir).as_posix(),
            "seed": cfg["seed"],
            "learning_rate": cfg["learning_rate"],
            "latent_dim": cfg["latent_dim"],
            "hidden_dim": cfg["hidden_dim"],
            "embedding_dim": cfg["embedding_dim"],
        }
        row.update(_parse_training_log(run_dir / "training.log"))

        if args.max_recon_items != 0:
            max_items = None if args.max_recon_items < 0 else args.max_recon_items
            try:
                recon = reconstruction_metrics(
                    model_name=model_name,
                    model=model,
                    dataset=dataset,
                    device=device,
                    max_items=max_items,
                    batch_size=args.batch_size,
                )
                row.update({f"recon_{key}": value for key, value in recon.items()})
            except Exception as exc:
                row["recon_error"] = f"{type(exc).__name__}: {exc}"

        gen_metrics, examples = _generation_metrics(
            model_name,
            model,
            dataset,
            device,
            latent_dim=cfg["latent_dim"],
            max_length=cfg["max_length"],
            samples=args.generation_samples,
            batch_size=args.batch_size,
            dataset_genotypes=dataset_genotypes,
            rng_seed=args.seed + len(rows),
            framsticks=framsticks,
        )
        row.update(gen_metrics)

        try:
            row.update(
                _latent_fitness_metrics(
                    model_name,
                    model,
                    dataset,
                    device,
                    samples=args.latent_samples,
                    batch_size=args.batch_size,
                    rng=rng,
                )
            )
        except Exception as exc:
            row["latent_fitness_error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)
        details["models"][run_dir.name] = {"examples": examples}

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = checkpoints_root_path / f"evaluation_{stamp}.csv"
    else:
        output_path = Path(maybe_resolve_path(args.output, root_dir=str(ROOT)))

    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in fieldnames})

    details_path = output_path.with_suffix(".details.json")
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    print(f"Wrote: {output_path}")
    print(f"Wrote: {details_path}")
    for row in rows:
        print(
            f"{row['label']}: valid={_format_value(row.get('gen_valid_rate'))} "
            f"novel={_format_value(row.get('gen_novel_valid_rate'))} "
            f"vertpos_max={_format_value(row.get('gen_vertpos_max'))} "
            f"latent_r2={_format_value(row.get('latent_linear_r2_in_sample'))}"
        )


if __name__ == "__main__":
    main()
