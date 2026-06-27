#!/usr/bin/env python3
"""Build a self-contained meeting summary pack.

The script inventories scripts, models, exports, and final benchmark results,
then writes Markdown/CSV summaries and plots under `summary_for_meeting/`.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "summary_for_meeting"
DATA = OUT / "data"
PLOTS = OUT / "plots"
EXISTING_PLOTS = OUT / "existing_final_plots"

CPU_RUN = ROOT / "exports/final_results_cpu/exports/final_seed_bucket_benchmark_baseline_20260613_011516"
GPU_RUN = ROOT / "exports/final_results_gpu/exports/final_seed_bucket_benchmark_20260607_194941"
FINAL_PLOTS = ROOT / "exports/final_benchmark_plots"


MODEL_DESCRIPTIONS = {
    "char_vae": "Character-level VAE: decodes genotype text as a character sequence. Useful as a simple baseline, but validity is harder to enforce than with grammar-rule decoders.",
    "grammar_vae": "Grammar VAE: represents F1 genotypes as grammar-production sequences and decodes through grammar rules.",
    "grammar_vae_masked": "Masked Grammar VAE: grammar-rule decoder with masks that restrict choices to syntactically valid productions for the current nonterminal.",
    "tree_vae": "TreeVAE: tree/derivation-oriented VAE for F1 grammar structures rather than plain character strings.",
    "tree_vae_masked": "Masked TreeVAE: tree-oriented decoder with grammar masks to reduce invalid production choices.",
    "tree_vae_masked_lhs": "TreeVAE LHS variants: tree/masked decoder conditioned on left-hand-side grammar context; some variants also add depth or conditioning signals.",
    "transformer_vae": "TransformerVAE: sequence VAE using attention-based encoder/decoder components for grammar-rule sequences.",
    "vq_grammar_ae": "VQ Grammar AE: vector-quantized grammar autoencoder variant explored during model development.",
    "surrogate": "Surrogate/auxiliary models: predictors or auxiliary search components used in exploratory experiments, not the final thesis comparison core.",
}

ARCHITECTURE_DETAILS = [
    {
        "name": "CharVAE",
        "what": "Character-level variational autoencoder for raw F1 genotype strings.",
        "description": "This is the simplest neural representation: the encoder reads the genotype as a sequence of characters and the decoder emits characters back. It is useful as a baseline because it makes the fewest assumptions about the grammar. Its weakness is that syntactic validity is not enforced during decoding, so small decoding errors can create invalid or semantically poor F1 genotypes. In the experimental process it mainly served as an early reference point for why grammar-aware representations are needed.",
    },
    {
        "name": "GrammarVAE",
        "what": "VAE over grammar-production sequences instead of raw characters.",
        "description": "GrammarVAE converts an F1 genotype into a sequence of grammar production choices and trains the autoencoder to reconstruct that sequence. This moves the representation closer to the formal F1 syntax and makes the latent space more structured than a pure character model. It still has to learn which production is valid at each derivation step, so invalid choices can remain possible without masking. It is an important intermediate step between text-level decoding and stricter grammar-constrained decoding.",
    },
    {
        "name": "Masked GrammarVAE",
        "what": "Grammar-production VAE with masks over valid productions.",
        "description": "The masked grammar decoder uses the current grammar context to restrict which productions can be chosen. This directly targets the syntactic-validity problem: the decoder is not asked to choose among productions that do not fit the current nonterminal. It keeps the sequential grammar-rule representation but adds hard structural information during decoding. In practice this family was used to test whether validity constraints alone improve reconstruction and downstream usefulness.",
    },
    {
        "name": "TreeVAE",
        "what": "Tree/derivation-aware VAE for F1 grammar structures.",
        "description": "TreeVAE treats the genotype as a derivation structure rather than only a flat sequence. The motivation is that F1 genotypes are variable-length and hierarchical, so parent-child and stack-like derivation context matters. This family attempts to represent the structural organization of the genotype more directly than plain sequence VAEs. It later became one of the strongest final benchmark families when combined with latent-space evolutionary search.",
    },
    {
        "name": "Masked TreeVAE",
        "what": "Tree-oriented VAE with grammar masks.",
        "description": "Masked TreeVAE combines tree-oriented decoding with grammar-validity masking. The decoder receives the benefit of structural organization while masks prevent choices that are incompatible with the current grammar expansion. This is a natural bridge between derivation-aware representation and syntactic validity enforcement. It was part of the validation sweeps that led to selecting stronger grammar-aware checkpoints.",
    },
    {
        "name": "TreeVAE LHS",
        "what": "Masked TreeVAE conditioned on the left-hand-side nonterminal.",
        "description": "The LHS variant explicitly conditions decoding on the grammar nonterminal currently being expanded. This makes the decoder aware of the local grammar context, rather than requiring it to infer that context only from the latent vector and previous decisions. It is especially relevant for F1 because the same local-looking production choice can have different meaning depending on where it occurs in the derivation. Several final candidate checkpoints came from this family.",
    },
    {
        "name": "TreeVAE LHS-depth",
        "what": "LHS-conditioned TreeVAE augmented with derivation-depth information.",
        "description": "The depth variant adds information about how deep the current expansion is in the derivation process. This is meant to help the decoder distinguish shallow structural decisions from deeply nested details. Depth is not an optimization result by itself; it is an additional conditioning signal intended to make reconstruction and valid decoding easier. It was included among final benchmark candidates to test whether this structural signal improves downstream search.",
    },
    {
        "name": "Fitness-aware LHS",
        "what": "LHS-conditioned TreeVAE with an additional scalar fitness-conditioning signal.",
        "description": "Fitness-aware decoding conditions the autoencoder on a scalar related to target fitness. The motivation is to give the representation some information about the quality dimension that will later matter during optimization. This does not replace true Framsticks evaluation; candidates still need to be decoded and evaluated by the simulator. The final benchmark includes a fitness-aware LHS model as one of the five selected model labels.",
    },
    {
        "name": "Conditional/structural LHS variants",
        "what": "LHS-conditioned variants with auxiliary structural conditioning.",
        "description": "These runs explored whether additional structural signals can help the decoder preserve useful genotype organization. They are part of the broader search over representation design, not the central final result by themselves. Their role is to test whether conditioning beyond syntax can produce better latent spaces for optimization. The meeting pack lists their checkpoints and training runs so that the development path is visible.",
    },
    {
        "name": "TransformerVAE",
        "what": "Attention-based VAE over grammar-rule sequences.",
        "description": "TransformerVAE uses attention-based sequence modeling to encode and decode grammar-rule sequences. The motivation is that attention can model longer-range dependencies in the derivation sequence better than purely recurrent or local mechanisms. In the final paired benchmark, the `10_transformer_vae_epoch90` checkpoint combined with latent-space EA achieved the best mean score among the latent configurations. This makes it one of the clearest models to explain during the supervisor meeting.",
    },
    {
        "name": "VQ Grammar AE",
        "what": "Vector-quantized grammar autoencoder variant.",
        "description": "The VQ Grammar AE explores a discrete latent-code alternative to the continuous VAE latent vector. It was part of the architecture exploration phase rather than the final best-performing benchmark set. Its purpose was to test whether quantized latent representations can organize grammar-rule genotypes in a useful way. It should be presented as an explored branch, not as the main final result.",
    },
    {
        "name": "Structural surrogate models",
        "what": "Auxiliary predictors/search helpers outside the core autoencoder benchmark.",
        "description": "Surrogate models were used in auxiliary experiments to estimate or filter candidate quality from structural features. They are useful for explaining the broader experimental exploration, but they are not the central CoSO-vs-LSO thesis result. In this meeting pack they are listed separately to avoid mixing auxiliary search experiments with the final latent-space benchmark. Any surrogate conclusions should be framed cautiously unless tied to a specific result file.",
    },
]

FINAL_LABEL_DESCRIPTIONS = {
    "09_tree_vae_epoch80": "TreeVAE checkpoint from epoch 80; one of the two best average final latent-space EA configurations.",
    "10_transformer_vae_epoch90": "TransformerVAE checkpoint from epoch 90; best average final latent-space EA configuration in the paired benchmark.",
    "02_lhs_latent256_seed42_best": "LHS grammar-aware TreeVAE variant with 256-dimensional latent space selected from validation/development results.",
    "06_lhs_depth_seed42_best": "LHS-depth variant that augments grammar context with derivation-depth information.",
    "fitness_aware_lhs_seed42": "Fitness-aware LHS variant conditioning decoding on a scalar fitness signal.",
}

COPY_ROOT_NOTES = {
    "exports/cluster_export": "cluster export copy; overlaps with timestamped/nested export snapshots",
    "exports/cluster_export_20260524_222538": "timestamped cluster export copy; overlaps with cluster_export",
    "exports/exports": "nested export snapshot; treat as copied evidence unless a file is unique",
}


def display_label(label: object) -> str:
    text = str(label)
    if "/" in text or "\\" in text:
        return Path(text.replace("\\", "/")).name
    return text


def ensure_dirs() -> None:
    for path in (OUT, DATA, PLOTS, EXISTING_PLOTS):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def evidence_path_priority(path_text: str) -> tuple[int, str]:
    lower = path_text.lower().replace("\\", "/")
    if "exports/f1_val_logs" in lower:
        return (0, lower)
    if "/models/" in lower and "cluster_export" not in lower:
        return (1, lower)
    if "exports/f1_selected_10_ckpts" in lower:
        return (2, lower)
    if "exports/cluster_export_20260524_222538" in lower:
        return (3, lower)
    if "exports/cluster_export" in lower:
        return (4, lower)
    if "exports/latest_results" in lower:
        return (5, lower)
    if "exports/exports" in lower:
        return (6, lower)
    return (10, lower)


def annotate_duplicate_groups(
    rows: list[dict[str, object]],
    *,
    key_col: str,
    path_col: str,
    group_col: str = "duplicate_group",
    count_col: str = "duplicate_count",
    canonical_col: str = "canonical_file",
    duplicate_col: str = "is_duplicate_copy",
) -> None:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(key_col, ""))
        if key:
            groups[key].append(row)

    for key, group in groups.items():
        canonical = min(group, key=lambda row: evidence_path_priority(str(row.get(path_col, ""))))
        canonical_path = str(canonical.get(path_col, ""))
        group_id = f"sha1:{key[:12]}" if len(group) > 1 else ""
        for row in group:
            path_text = str(row.get(path_col, ""))
            row[group_col] = group_id
            row[count_col] = len(group)
            row[canonical_col] = canonical_path
            row[duplicate_col] = len(group) > 1 and path_text != canonical_path


def unique_evidence_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if not bool(row.get("is_duplicate_copy"))]


def duplicate_group_rows(rows: list[dict[str, object]], *, file_col: str = "file") -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get("duplicate_group", ""))
        if group_id:
            groups[group_id].append(row)

    summaries = []
    for group_id, group in sorted(groups.items()):
        if len(group) <= 1:
            continue
        canonical = str(group[0].get("canonical_file", ""))
        categories = sorted({str(row.get("category", "")) for row in group if row.get("category")})
        files = sorted(str(row.get(file_col, "")) for row in group)
        first_rows = group[0].get("rows", "")
        summaries.append(
            {
                "duplicate_group": group_id,
                "copies": len(group),
                "canonical_file": canonical,
                "categories": ", ".join(categories),
                "rows_per_copy": first_rows,
                "raw_rows_across_copies": numeric_row_sum(row.get("rows", 0) for row in group),
                "duplicate_files_sample": " | ".join(files[:6]),
            }
        )
    return summaries


def numeric_row_sum(values: object) -> int:
    total = 0
    for value in values:  # type: ignore[operator]
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def export_evidence_note(path: Path) -> str:
    path_text = rel(path)
    if path_text in COPY_ROOT_NOTES:
        return COPY_ROOT_NOTES[path_text]
    if path_text.endswith(".tar.gz"):
        return "archive copy; not separate experimental evidence"
    if path_text.startswith("exports/latest_results_"):
        return "latest cluster result snapshot; may duplicate nested CSVs from copied exports"
    return ""


def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for col in columns:
            raw_value = row.get(col, "")
            if isinstance(raw_value, float) and math.isnan(raw_value):
                value = ""
            else:
                value = str(raw_value)
            value = value.replace("\n", " ").replace("|", "\\|")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def script_category(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("slurm_final") or name == "submit_final_benchmark_jobs.sh" or "final_benchmark" in name:
        return "final benchmark"
    if name.startswith("slurm_train") or name.startswith("train"):
        return "training"
    if "latent" in name or "cem" in name or "mutation_search" in name or "optimize" in name:
        return "optimization/search"
    if "plot" in name or "presentation" in name:
        return "plotting/presentation"
    if "export" in name or "prepare" in name or "merge" in name:
        return "export/prepare/merge"
    if "eval" in name or "benchmark" in name or "compare" in name:
        return "evaluation/comparison"
    if "workbench" in name:
        return "interactive/workbench"
    return "utility"


def first_description(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    lines = text.splitlines()
    for line in lines[:60]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#!") or stripped in {'"""', "'''"}:
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            return stripped.strip('"\' ')[:160]
        if stripped.startswith("#") and not stripped.startswith("#SBATCH"):
            return stripped.lstrip("# ")[:160]
        if "ArgumentParser" in stripped and "description=" in stripped:
            return stripped[:160]
    return ""


def inventory_scripts() -> list[dict[str, object]]:
    rows = []
    for path in sorted((ROOT / "scripts").glob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "script": rel(path),
                "category": script_category(path),
                "description": first_description(path),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "scripts_inventory.csv", index=False)
    return rows


def inventory_source_models() -> list[dict[str, object]]:
    rows = []
    for path in sorted((ROOT / "src/f1vae/models").glob("*.py")):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        classes = re.findall(r"^class\s+(\w+)", text, flags=re.MULTILINE)
        key = path.stem
        rows.append(
            {
                "module": rel(path),
                "classes": ", ".join(classes),
                "description": MODEL_DESCRIPTIONS.get(key, "Model implementation module."),
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "source_model_modules.csv", index=False)
    return rows


def inventory_checkpoints() -> list[dict[str, object]]:
    rows = []
    for base_name in ("models", "exports"):
        base = ROOT / base_name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"}:
                rows.append(
                    {
                        "checkpoint": rel(path),
                        "root": base_name,
                        "size_mb": round(file_size_mb(path), 2),
                        "modified": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                    }
                )
    rows = sorted(rows, key=lambda row: row["checkpoint"])
    pd.DataFrame(rows).to_csv(DATA / "checkpoint_inventory.csv", index=False)
    return rows


def model_run_category(path: Path) -> str:
    text = rel(path).lower()
    if "fitness_aware" in text:
        return "fitness-aware LHS"
    if "struct_conditional" in text or "structural" in text:
        return "structural/conditional"
    if "conditional" in text:
        return "conditional LHS"
    if "transformer" in text:
        return "TransformerVAE"
    if "tree_vae_masked_lhs" in text or "lhs" in text:
        return "LHS/depth TreeVAE"
    if "tree_vae_masked" in text:
        return "masked TreeVAE"
    if "tree_vae" in text:
        return "TreeVAE"
    if "grammar_vae_masked" in text:
        return "masked GrammarVAE"
    if "grammar_vae" in text:
        return "GrammarVAE"
    if "char_vae" in text:
        return "CharVAE"
    if "vq" in text:
        return "VQ Grammar AE"
    if "surrogate" in text:
        return "surrogate"
    return "other"


def inventory_training_runs() -> list[dict[str, object]]:
    roots = [ROOT / "models", ROOT / "exports"]
    run_dirs: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            direct_ckpts = list(path.glob("*.pth")) + list(path.glob("*.pt")) + list(path.glob("*.ckpt"))
            checkpoint_child = path / "checkpoints"
            if direct_ckpts or checkpoint_child.is_dir():
                if path.name == "checkpoints":
                    run_dirs.add(path.parent)
                else:
                    run_dirs.add(path)

    rows = []
    for run_dir in sorted(run_dirs):
        ckpts = sorted(
            child
            for child in run_dir.rglob("*")
            if child.is_file() and child.suffix.lower() in {".pth", ".pt", ".ckpt"}
        )
        if not ckpts:
            continue
        best = [path for path in ckpts if "best" in path.name.lower()]
        finals = [path for path in ckpts if "final" in path.name.lower()]
        configs = [path for path in run_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json"}]
        total_mb = sum(path.stat().st_size for path in ckpts) / (1024 * 1024)
        rows.append(
            {
                "run_dir": rel(run_dir),
                "category": model_run_category(run_dir),
                "checkpoints": len(ckpts),
                "best_like": len(best),
                "final_like": len(finals),
                "config_files": len(configs),
                "size_mb": round(total_mb, 2),
                "example_checkpoint": rel(best[0] if best else ckpts[-1]),
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "training_run_inventory.csv", index=False)
    return rows


VAL_RECON_RE = re.compile(
    r"ValRecon\s+(\d+)/(\d+)\s+Exact:\s+(\d+)/(\d+)\s+ExactAcc:\s+([0-9.]+)%\s+AvgSim:\s+([0-9.]+)%"
)
TRAIN_RECON_RE = re.compile(
    r"TrainRecon\s+(\d+)/(\d+)\s+Exact:\s+(\d+)/(\d+)\s+ExactAcc:\s+([0-9.]+)%\s+AvgSim:\s+([0-9.]+)%"
)
BEST_RECON_RE = re.compile(
    r"BestValRecon\s+(\d+)/(\d+)\s+Exact:\s+(\d+)/(\d+)\s+ExactAcc:\s+([0-9.]+)%\s+AvgSim:\s+([0-9.]+)%"
)
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+Loss:\s+([0-9.]+)\s+CE:\s+([0-9.]+)\s+KLD:\s+([0-9.]+)\s+Beta:\s+([0-9.]+)\s+TF:\s+([0-9.]+)"
)


def load_run_config(run_dir: Path) -> dict[str, object]:
    config = run_dir / "run_config.json"
    if not config.exists():
        return {}
    try:
        return json.loads(config.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_training_logs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    roots = [ROOT / "models", ROOT / "exports", ROOT / "legacy/models"]
    logs: list[Path] = []
    for root in roots:
        if root.exists():
            logs.extend(sorted(root.rglob("training.log")))

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for log_path in sorted(set(logs)):
        run_dir = log_path.parent
        config = load_run_config(run_dir)
        model = str(config.get("model") or model_run_category(run_dir))
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        log_hash = sha1_text(log_text)
        val_rows: list[dict[str, object]] = []
        train_by_epoch: dict[int, dict[str, object]] = {}
        epoch_loss_by_epoch: dict[int, dict[str, object]] = {}
        best_markers: list[dict[str, object]] = []
        for line in log_text.splitlines():
            if match := EPOCH_RE.search(line):
                epoch = int(match.group(1))
                epoch_loss_by_epoch[epoch] = {
                    "train_loss": float(match.group(3)),
                    "train_ce": float(match.group(4)),
                    "train_kld": float(match.group(5)),
                    "beta": float(match.group(6)),
                    "teacher_forcing": float(match.group(7)),
                }
            if match := TRAIN_RECON_RE.search(line):
                epoch = int(match.group(1))
                train_by_epoch[epoch] = {
                    "train_exact": int(match.group(3)),
                    "train_total": int(match.group(4)),
                    "train_exact_acc": float(match.group(5)),
                    "train_avg_sim": float(match.group(6)),
                }
            if match := VAL_RECON_RE.search(line):
                epoch = int(match.group(1))
                row = {
                    "run_dir": rel(run_dir),
                    "log_path": rel(log_path),
                    "training_log_sha1": log_hash,
                    "model": model,
                    "category": model_run_category(run_dir),
                    "epoch": epoch,
                    "total_epochs": int(match.group(2)),
                    "val_exact": int(match.group(3)),
                    "val_total": int(match.group(4)),
                    "val_exact_acc": float(match.group(5)),
                    "val_avg_sim": float(match.group(6)),
                }
                row.update(train_by_epoch.get(epoch, {}))
                row.update(epoch_loss_by_epoch.get(epoch, {}))
                val_rows.append(row)
                curve_rows.append(row)
            if match := BEST_RECON_RE.search(line):
                best_markers.append(
                    {
                        "epoch": int(match.group(1)),
                        "total_epochs": int(match.group(2)),
                        "val_exact": int(match.group(3)),
                        "val_total": int(match.group(4)),
                        "val_exact_acc": float(match.group(5)),
                        "val_avg_sim": float(match.group(6)),
                    }
                )
        if not val_rows:
            continue
        best = max(val_rows, key=lambda row: (float(row["val_exact_acc"]), float(row["val_avg_sim"])))
        last = max(val_rows, key=lambda row: int(row["epoch"]))
        marker_best = best_markers[-1] if best_markers else best
        summary_rows.append(
            {
                "run_dir": rel(run_dir),
                "log_path": rel(log_path),
                "training_log_sha1": log_hash,
                "model": model,
                "category": model_run_category(run_dir),
                "latent_dim": config.get("latent_dim", ""),
                "hidden_dim": config.get("hidden_dim", ""),
                "embedding_dim": config.get("embedding_dim", ""),
                "seed": config.get("seed", ""),
                "epochs_config": config.get("epochs", ""),
                "best_epoch": best["epoch"],
                "best_val_exact_acc": best["val_exact_acc"],
                "best_val_avg_sim": best["val_avg_sim"],
                "best_val_exact": best["val_exact"],
                "val_total": best["val_total"],
                "last_epoch": last["epoch"],
                "last_val_exact_acc": last["val_exact_acc"],
                "last_val_avg_sim": last["val_avg_sim"],
                "best_marker_epoch": marker_best["epoch"],
                "best_marker_val_exact_acc": marker_best["val_exact_acc"],
                "best_marker_val_avg_sim": marker_best["val_avg_sim"],
            }
        )
    annotate_duplicate_groups(
        summary_rows,
        key_col="training_log_sha1",
        path_col="run_dir",
        group_col="training_log_duplicate_group",
        count_col="training_log_copies",
        canonical_col="canonical_run_dir",
    )
    pd.DataFrame(summary_rows).to_csv(DATA / "training_log_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(DATA / "training_curves.csv", index=False)
    return summary_rows, curve_rows


def selected_checkpoint_evaluation() -> pd.DataFrame:
    base = ROOT / "exports/f1_selected_10_ckpts_20260520_184738"
    preferred = base / "evaluation_framsticks.csv"
    fallback = base / "evaluation_local.csv"
    if preferred.exists():
        df = pd.read_csv(preferred)
        df["evaluation_source"] = rel(preferred)
    elif fallback.exists():
        df = pd.read_csv(fallback)
        df["evaluation_source"] = rel(fallback)
    else:
        df = pd.DataFrame()
    if not df.empty:
        df["display_label"] = df["label"].map(display_label)
    df.to_csv(DATA / "selected_checkpoint_evaluation.csv", index=False)
    return df


def inventory_exports() -> list[dict[str, object]]:
    rows = []
    exports = ROOT / "exports"
    if not exports.exists():
        return rows
    for path in sorted(exports.iterdir()):
        if path.is_dir():
            file_count = sum(1 for child in path.rglob("*") if child.is_file())
            total_mb = sum(child.stat().st_size for child in path.rglob("*") if child.is_file()) / (1024 * 1024)
            kind = "directory"
        else:
            file_count = 1
            total_mb = file_size_mb(path)
            kind = path.suffix.lower() or "file"
        rows.append(
            {
                "entry": rel(path),
                "kind": kind,
                "files": file_count,
                "size_mb": round(total_mb, 2),
                "evidence_note": export_evidence_note(path),
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "exports_inventory.csv", index=False)
    return rows


def load_job_csvs(run_dir: Path, job_kind: str) -> pd.DataFrame:
    jobs = run_dir / "jobs" / job_kind
    files = sorted(path for path in jobs.glob("*.csv") if not path.name.endswith(".history.csv"))
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_csv"] = rel(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_history_csvs(run_dir: Path, job_kind: str) -> pd.DataFrame:
    jobs = run_dir / "jobs" / job_kind
    files = sorted(jobs.glob("*.history.csv"))
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_csv"] = rel(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_final_benchmark() -> dict[str, object]:
    cpu = load_job_csvs(CPU_RUN, "cpu")
    gpu = load_job_csvs(GPU_RUN, "gpu")
    cpu_hist = load_history_csvs(CPU_RUN, "cpu")
    gpu_hist = load_history_csvs(GPU_RUN, "gpu")

    cpu.to_csv(DATA / "final_cpu_rows.csv", index=False)
    gpu.to_csv(DATA / "final_gpu_rows.csv", index=False)

    base_cols = ["bucket_id", "seed_rank", "seed_dataset_idx", "best_score"]
    baseline = cpu[base_cols].rename(columns={"best_score": "baseline_best_score"})
    paired = gpu.merge(baseline, on=["bucket_id", "seed_rank", "seed_dataset_idx"], how="left")
    paired["diff_vs_baseline"] = paired["best_score"] - paired["baseline_best_score"]
    paired["win_vs_baseline"] = paired["diff_vs_baseline"] > 0
    paired["improvement_vs_seed"] = paired["best_score"] - paired["seed_source_fitness"]
    paired.to_csv(DATA / "final_paired_gpu_vs_baseline.csv", index=False)

    group_cols = ["label", "model", "method"]
    summary = (
        paired.groupby(group_cols)
        .agg(
            n=("best_score", "size"),
            mean_best=("best_score", "mean"),
            median_best=("best_score", "median"),
            max_best=("best_score", "max"),
            std_best=("best_score", "std"),
            mean_diff_vs_baseline=("diff_vs_baseline", "mean"),
            median_diff_vs_baseline=("diff_vs_baseline", "median"),
            wins_vs_baseline=("win_vs_baseline", "sum"),
            mean_improvement_vs_seed=("improvement_vs_seed", "mean"),
            mean_evaluated_unique=("evaluated_unique", "mean"),
            mean_invalid_count=("invalid_count", "mean"),
            mean_duplicate_count=("duplicate_count", "mean"),
        )
        .reset_index()
    )
    summary["winrate_vs_baseline"] = summary["wins_vs_baseline"] / summary["n"]
    summary = summary.sort_values(["mean_best", "mean_diff_vs_baseline"], ascending=False)
    summary.to_csv(DATA / "final_config_summary.csv", index=False)

    method_summary = (
        paired.groupby("method")
        .agg(
            n=("best_score", "size"),
            mean_best=("best_score", "mean"),
            mean_diff_vs_baseline=("diff_vs_baseline", "mean"),
            wins_vs_baseline=("win_vs_baseline", "sum"),
            max_best=("best_score", "max"),
        )
        .reset_index()
    )
    method_summary["winrate_vs_baseline"] = method_summary["wins_vs_baseline"] / method_summary["n"]
    method_summary.to_csv(DATA / "final_method_summary.csv", index=False)

    bucket_summary = (
        paired.groupby(["bucket_id", "label", "method"])
        .agg(
            n=("best_score", "size"),
            mean_best=("best_score", "mean"),
            mean_diff_vs_baseline=("diff_vs_baseline", "mean"),
            winrate_vs_baseline=("win_vs_baseline", "mean"),
        )
        .reset_index()
    )
    bucket_summary.to_csv(DATA / "final_bucket_summary.csv", index=False)

    baseline_summary = pd.DataFrame(
        [
            {
                "n": len(cpu),
                "mean_best": cpu["best_score"].mean(),
                "median_best": cpu["best_score"].median(),
                "max_best": cpu["best_score"].max(),
                "min_best": cpu["best_score"].min(),
                "mean_improvement_vs_seed": (cpu["best_score"] - cpu["seed_source_fitness"]).mean(),
            }
        ]
    )
    baseline_summary.to_csv(DATA / "final_baseline_summary.csv", index=False)

    oracle = paired.groupby(["bucket_id", "seed_rank", "seed_dataset_idx"]).agg(
        oracle_best=("best_score", "max"), baseline_best_score=("baseline_best_score", "first")
    )
    oracle = oracle.reset_index()
    oracle["oracle_diff_vs_baseline"] = oracle["oracle_best"] - oracle["baseline_best_score"]
    oracle["oracle_win_vs_baseline"] = oracle["oracle_diff_vs_baseline"] > 0
    oracle.to_csv(DATA / "final_oracle_portfolio.csv", index=False)
    oracle_summary = pd.DataFrame(
        [
            {
                "n": len(oracle),
                "mean_oracle_best": oracle["oracle_best"].mean(),
                "mean_diff_vs_baseline": oracle["oracle_diff_vs_baseline"].mean(),
                "winrate_vs_baseline": oracle["oracle_win_vs_baseline"].mean(),
                "max_oracle_best": oracle["oracle_best"].max(),
            }
        ]
    )
    oracle_summary.to_csv(DATA / "final_oracle_summary.csv", index=False)

    return {
        "cpu": cpu,
        "gpu": gpu,
        "cpu_history_rows": len(cpu_hist),
        "gpu_history_rows": len(gpu_hist),
        "paired": paired,
        "summary": summary,
        "method_summary": method_summary,
        "bucket_summary": bucket_summary,
        "baseline_summary": baseline_summary,
        "oracle_summary": oracle_summary,
    }


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, output: Path, limit: int = 15) -> None:
    plot_df = df.head(limit).copy()
    plt.figure(figsize=(11, 6))
    plt.barh(plot_df[x][::-1], plot_df[y][::-1])
    plt.title(title)
    plt.xlabel(y)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_heatmap(summary: pd.DataFrame, value: str, output: Path, title: str) -> None:
    plot_summary = summary.copy()
    plot_summary["display_label"] = plot_summary["label"].map(display_label)
    pivot = plot_summary.pivot(index="display_label", columns="method", values=value).fillna(0)
    plt.figure(figsize=(9, max(4, 0.55 * len(pivot))))
    plt.imshow(pivot.values, aspect="auto", cmap="coolwarm")
    plt.colorbar(label=value)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_bucket_lines(bucket_summary: pd.DataFrame, top_keys: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(10, 6))
    for _, row in top_keys.iterrows():
        label = row["label"]
        method = row["method"]
        subset = bucket_summary[(bucket_summary["label"] == label) & (bucket_summary["method"] == method)]
        subset = subset.sort_values("bucket_id")
        plt.plot(subset["bucket_id"], subset["mean_diff_vs_baseline"], marker="o", label=f"{display_label(label)} / {method}")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Mean difference vs Framsticks EA by starting-fitness bucket")
    plt.xlabel("Bucket")
    plt.ylabel("Mean best score difference")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def copy_existing_plots() -> list[str]:
    copied = []
    if not FINAL_PLOTS.exists():
        return copied
    for path in sorted(FINAL_PLOTS.glob("*.png")):
        target = EXISTING_PLOTS / path.name
        shutil.copy2(path, target)
        copied.append(rel(target))
    return copied


def make_plots(final: dict[str, object]) -> list[str]:
    summary: pd.DataFrame = final["summary"]  # type: ignore[assignment]
    method_summary: pd.DataFrame = final["method_summary"]  # type: ignore[assignment]
    bucket_summary: pd.DataFrame = final["bucket_summary"]  # type: ignore[assignment]
    outputs = []

    top = summary.sort_values("mean_best", ascending=False).copy()
    top["display_label"] = top["label"].map(display_label)
    top["config"] = top["display_label"] + " / " + top["method"]
    plot_bar(top, "config", "mean_best", "Top configurations by mean best score", PLOTS / "top_configs_mean_best.png")
    outputs.append(rel(PLOTS / "top_configs_mean_best.png"))
    top_diff = summary.sort_values("mean_diff_vs_baseline", ascending=False).copy()
    top_diff["display_label"] = top_diff["label"].map(display_label)
    top_diff["config"] = top_diff["display_label"] + " / " + top_diff["method"]
    plot_bar(top_diff, "config", "mean_diff_vs_baseline", "Top configurations by mean difference vs Framsticks EA", PLOTS / "top_configs_diff_vs_baseline.png")
    outputs.append(rel(PLOTS / "top_configs_diff_vs_baseline.png"))
    win = summary.sort_values("winrate_vs_baseline", ascending=False).copy()
    win["display_label"] = win["label"].map(display_label)
    win["config"] = win["display_label"] + " / " + win["method"]
    plot_bar(win, "config", "winrate_vs_baseline", "Top configurations by win rate vs Framsticks EA", PLOTS / "top_configs_winrate.png")
    outputs.append(rel(PLOTS / "top_configs_winrate.png"))
    plot_heatmap(summary, "mean_diff_vs_baseline", PLOTS / "heatmap_mean_diff_vs_baseline.png", "Mean difference vs baseline")
    outputs.append(rel(PLOTS / "heatmap_mean_diff_vs_baseline.png"))
    plot_heatmap(summary, "winrate_vs_baseline", PLOTS / "heatmap_winrate_vs_baseline.png", "Win rate vs baseline")
    outputs.append(rel(PLOTS / "heatmap_winrate_vs_baseline.png"))
    plot_bucket_lines(bucket_summary, top_diff.head(5), PLOTS / "bucket_diff_top5.png")
    outputs.append(rel(PLOTS / "bucket_diff_top5.png"))

    method_plot = method_summary.sort_values("mean_diff_vs_baseline", ascending=False).copy()
    plot_bar(method_plot, "method", "mean_diff_vs_baseline", "Method-level mean difference vs baseline", PLOTS / "method_mean_diff.png", limit=10)
    outputs.append(rel(PLOTS / "method_mean_diff.png"))
    return outputs


def make_training_plots(training_summary: list[dict[str, object]], training_curves: list[dict[str, object]], selected_eval: pd.DataFrame) -> list[str]:
    outputs: list[str] = []
    training_summary = unique_evidence_rows(training_summary)
    if training_summary:
        df = pd.DataFrame(training_summary).sort_values("best_val_exact_acc", ascending=False)
        plot_df = df.head(20).copy()
        plot_df["run_short"] = plot_df["run_dir"].map(lambda text: "/".join(str(text).split("/")[-3:]))
        plt.figure(figsize=(12, 8))
        plt.barh(plot_df["run_short"][::-1], plot_df["best_val_exact_acc"][::-1])
        plt.xlabel("Best validation exact reconstruction accuracy (%)")
        plt.title("Top training runs by validation exact reconstruction")
        plt.tight_layout()
        out = PLOTS / "training_top_val_exact_acc.png"
        plt.savefig(out, dpi=180)
        plt.close()
        outputs.append(rel(out))

        family = (
            df.groupby("category")
            .agg(runs=("run_dir", "size"), best_val_exact_acc=("best_val_exact_acc", "max"), best_val_avg_sim=("best_val_avg_sim", "max"))
            .reset_index()
            .sort_values("best_val_exact_acc", ascending=False)
        )
        plt.figure(figsize=(10, 5))
        plt.barh(family["category"][::-1], family["best_val_exact_acc"][::-1])
        plt.xlabel("Best validation exact reconstruction accuracy (%)")
        plt.title("Best validation reconstruction by model family")
        plt.tight_layout()
        out = PLOTS / "training_family_best_val_exact_acc.png"
        plt.savefig(out, dpi=180)
        plt.close()
        outputs.append(rel(out))

    if training_curves and training_summary:
        curves = pd.DataFrame(training_curves)
        top_runs = pd.DataFrame(training_summary).sort_values("best_val_exact_acc", ascending=False).head(8)["run_dir"].tolist()
        plt.figure(figsize=(11, 6))
        for run_dir in top_runs:
            subset = curves[curves["run_dir"] == run_dir].sort_values("epoch")
            if subset.empty:
                continue
            short = "/".join(str(run_dir).split("/")[-3:])
            plt.plot(subset["epoch"], subset["val_exact_acc"], label=short)
        plt.xlabel("Epoch")
        plt.ylabel("Validation exact accuracy (%)")
        plt.title("Validation reconstruction curves for top runs")
        plt.legend(fontsize=7)
        plt.tight_layout()
        out = PLOTS / "training_val_exact_curves_top_runs.png"
        plt.savefig(out, dpi=180)
        plt.close()
        outputs.append(rel(out))

    if not selected_eval.empty:
        eval_df = selected_eval.copy().sort_values("log_val_exact_acc", ascending=False)
        metrics = [
            ("log_val_exact_acc", "Selected checkpoints: log validation exact accuracy (%)", "selected_log_val_exact_acc.png"),
            ("recon_exact_accuracy", "Selected checkpoints: local reconstruction exact accuracy (%)", "selected_recon_exact_accuracy.png"),
            ("gen_valid_rate", "Selected checkpoints: random generation valid rate", "selected_gen_valid_rate.png"),
            ("latent_linear_r2_in_sample", "Selected checkpoints: latent linear fitness R2", "selected_latent_linear_r2.png"),
        ]
        for metric, title, filename in metrics:
            if metric not in eval_df.columns:
                continue
            plot_df = eval_df[["display_label", metric]].dropna().copy()
            if plot_df.empty:
                continue
            plt.figure(figsize=(11, 6))
            plt.barh(plot_df["display_label"][::-1], plot_df[metric][::-1])
            plt.xlabel(metric)
            plt.title(title)
            plt.tight_layout()
            out = PLOTS / filename
            plt.savefig(out, dpi=180)
            plt.close()
            outputs.append(rel(out))
    return outputs


def inventory_misc_results() -> list[dict[str, object]]:
    rows = []
    for path in sorted((ROOT / "exports").glob("*.csv")):
        try:
            df = pd.read_csv(path)
            rows.append(
                {
                    "file": rel(path),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "has_best_score": "best_score" in df.columns,
                    "max_best_score": round(float(df["best_score"].max()), 6) if "best_score" in df.columns else "",
                    "mean_best_score": round(float(df["best_score"].mean()), 6) if "best_score" in df.columns else "",
                }
            )
        except Exception as exc:
            rows.append({"file": rel(path), "rows": "error", "columns": str(exc), "has_best_score": "", "max_best_score": "", "mean_best_score": ""})
    pd.DataFrame(rows).to_csv(DATA / "misc_export_csv_summary.csv", index=False)
    return rows


def csv_result_category(path: Path) -> str:
    text = rel(path).lower()
    if "final_results_cpu" in text or "final_results_gpu" in text:
        if text.endswith(".history.csv"):
            return "final benchmark history"
        return "final benchmark summary/job"
    if "f1_selected_10_ckpts" in text:
        if "evaluation" in text:
            return "selected checkpoint evaluation"
        if "latent_optimization" in text or "seeded_method" in text:
            return "selected checkpoint optimization prototype"
        return "selected checkpoint export"
    if "latest_results" in text:
        return "latest cluster result"
    if "surrogate" in text:
        return "surrogate experiment"
    if "local_30min" in text:
        return "local smoke/prototype"
    if "grammar_cem" in text:
        return "grammar CEM prototype"
    return "other export CSV"


def inventory_all_export_csv_results() -> list[dict[str, object]]:
    rows = []
    exports = ROOT / "exports"
    for path in sorted(exports.rglob("*.csv")) if exports.exists() else []:
        try:
            content_sha1 = sha1_file(path)
            df = pd.read_csv(path)
            row: dict[str, object] = {
                "file": rel(path),
                "category": csv_result_category(path),
                "content_sha1": content_sha1,
                "rows": len(df),
                "columns": len(df.columns),
                "labels": ", ".join(map(str, sorted(df["label"].dropna().unique())[:8])) if "label" in df.columns else "",
                "methods": ", ".join(map(str, sorted(df["method"].dropna().unique())[:8])) if "method" in df.columns else ", ".join(map(str, sorted(df["algorithm"].dropna().unique())[:8])) if "algorithm" in df.columns else "",
                "models": ", ".join(map(str, sorted(df["model"].dropna().unique())[:8])) if "model" in df.columns else "",
                "has_best_score": "best_score" in df.columns,
                "mean_best_score": round(float(df["best_score"].mean()), 6) if "best_score" in df.columns else "",
                "max_best_score": round(float(df["best_score"].max()), 6) if "best_score" in df.columns else "",
                "has_history": path.name.endswith(".history.csv"),
            }
            rows.append(row)
        except Exception as exc:
            rows.append({"file": rel(path), "category": csv_result_category(path), "rows": "error", "columns": str(exc)})
    annotate_duplicate_groups(rows, key_col="content_sha1", path_col="file")
    pd.DataFrame(rows).to_csv(DATA / "all_export_csv_summary.csv", index=False)
    pd.DataFrame(duplicate_group_rows(rows)).to_csv(DATA / "all_export_csv_duplicate_groups.csv", index=False)
    return rows


def write_overview(
    scripts: list[dict[str, object]],
    model_modules: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
    training_runs: list[dict[str, object]],
    training_summary: list[dict[str, object]],
    training_curves: list[dict[str, object]],
    selected_eval: pd.DataFrame,
    exports: list[dict[str, object]],
    misc_results: list[dict[str, object]],
    all_csv_results: list[dict[str, object]],
    final: dict[str, object],
    plot_outputs: list[str],
    training_plot_outputs: list[str],
    copied_plots: list[str],
) -> None:
    cpu: pd.DataFrame = final["cpu"]  # type: ignore[assignment]
    gpu: pd.DataFrame = final["gpu"]  # type: ignore[assignment]
    summary: pd.DataFrame = final["summary"]  # type: ignore[assignment]
    baseline_summary: pd.DataFrame = final["baseline_summary"]  # type: ignore[assignment]
    oracle_summary: pd.DataFrame = final["oracle_summary"]  # type: ignore[assignment]
    method_summary: pd.DataFrame = final["method_summary"]  # type: ignore[assignment]
    unique_training_summary = unique_evidence_rows(training_summary)
    unique_csv_results = unique_evidence_rows(all_csv_results)
    duplicate_csv_groups = duplicate_group_rows(all_csv_results)

    generated = dt.datetime.now().isoformat(timespec="seconds")
    top_rows = summary.head(10).copy()
    top_rows["display_label"] = top_rows["label"].map(display_label)
    top_rows["mean_best"] = top_rows["mean_best"].map(lambda x: f"{x:.4f}")
    top_rows["mean_diff_vs_baseline"] = top_rows["mean_diff_vs_baseline"].map(lambda x: f"{x:.4f}")
    top_rows["winrate_vs_baseline"] = top_rows["winrate_vs_baseline"].map(lambda x: f"{x:.1%}")
    top_rows["max_best"] = top_rows["max_best"].map(lambda x: f"{x:.4f}")

    method_rows = method_summary.copy()
    for col in ("mean_best", "mean_diff_vs_baseline", "max_best"):
        method_rows[col] = method_rows[col].map(lambda x: f"{x:.4f}")
    method_rows["winrate_vs_baseline"] = method_rows["winrate_vs_baseline"].map(lambda x: f"{x:.1%}")

    base = baseline_summary.iloc[0].to_dict()
    oracle = oracle_summary.iloc[0].to_dict()

    readme = []
    readme.append("# Summary For Meeting")
    readme.append("")
    readme.append(f"Generated: {generated}")
    readme.append("")
    readme.append("## Executive Summary")
    readme.append("")
    readme.append("This folder summarizes the current thesis/codebase state for a supervisor meeting. It focuses on the implemented autoencoder/Latent Space Optimization path, final paired benchmark evidence, scripts, exported results, and model/checkpoint inventory. CoSO is listed only as thesis scope/pending work where relevant; no CoSO results are invented here.")
    readme.append("")
    readme.append("Main points:")
    readme.append(f"- Dataset: `datasets/f1/f1_dataset.txt`, 10,893 F1 genotype/`vertpos` rows according to the thesis audit.")
    readme.append(f"- Final benchmark evidence: {len(cpu)} Framsticks EA baseline rows and {len(gpu)} latent-method rows.")
    readme.append(f"- Final latent benchmark design represented in data: {gpu['label'].nunique()} model labels, {gpu['method'].nunique()} methods, {gpu['bucket_id'].nunique()} fitness buckets.")
    readme.append(f"- Export CSV evidence: {len(unique_csv_results)} unique CSV contents from {len(all_csv_results)} local CSV files; copied export snapshots are kept for provenance but not counted as independent results.")
    readme.append(f"- Baseline Framsticks EA mean best score: {base['mean_best']:.4f}; max: {base['max_best']:.4f}.")
    readme.append(f"- Best average configuration: `{display_label(summary.iloc[0]['label'])} / {summary.iloc[0]['method']}` with mean best score {summary.iloc[0]['mean_best']:.4f} and mean difference vs baseline {summary.iloc[0]['mean_diff_vs_baseline']:.4f}.")
    readme.append(f"- Oracle/portfolio analysis across latent configurations gives mean best score {oracle['mean_oracle_best']:.4f}; this is an upper-bound diagnostic, not a deployable method.")
    readme.append("")
    readme.append("## What Has Been Built")
    readme.append("")
    readme.append("- A full F1 genotype data pipeline: loading genotype/fitness pairs, grammar-rule encodings, one-hot encodings, and train/validation handling.")
    readme.append("- Multiple autoencoder families for F1 genotypes: character-level, grammar-rule, masked grammar, tree-oriented, LHS/depth/fitness-aware variants, TransformerVAE, and VQ Grammar AE experiments.")
    readme.append(f"- Training evidence extracted from {len(unique_training_summary)} unique `training.log` contents ({len(training_summary)} local copies) and {len(training_runs)} model/checkpoint run directories or copies.")
    readme.append("- Latent-space optimization methods: latent evolutionary search, CEM, and CMA-ES operating on latent vectors decoded back to F1 genotypes and evaluated by true Framsticks `vertpos`.")
    readme.append("- A paired final benchmark using shared starting genotypes and buckets, with a separately run Framsticks EA baseline.")
    readme.append("- Plotting/export infrastructure, seminar presentation assets, and current thesis chapters through selected sections of Chapter 3.")
    readme.append("")
    readme.append("## Final Benchmark Top Configurations")
    readme.append("")
    readme.append(md_table(top_rows.to_dict("records"), ["display_label", "model", "method", "n", "mean_best", "mean_diff_vs_baseline", "winrate_vs_baseline", "max_best"]))
    readme.append("")
    readme.append("## Method-Level Summary")
    readme.append("")
    readme.append(md_table(method_rows.to_dict("records"), ["method", "n", "mean_best", "mean_diff_vs_baseline", "winrate_vs_baseline", "max_best"]))
    readme.append("")
    readme.append("## Final Model Labels")
    readme.append("")
    label_rows = []
    for label in sorted(gpu["label"].dropna().unique()):
        shown = display_label(label)
        label_rows.append({"label": shown, "source_label": label, "description": FINAL_LABEL_DESCRIPTIONS.get(shown, FINAL_LABEL_DESCRIPTIONS.get(label, "Final benchmark model label."))})
    readme.append(md_table(label_rows, ["label", "description"]))
    readme.append("")
    readme.append("## Generated Plots")
    readme.append("")
    for path in plot_outputs:
        readme.append(f"- `{path}`")
    readme.append("")
    readme.append("## Existing Final Benchmark Plots Copied Here")
    readme.append("")
    for path in copied_plots:
        readme.append(f"- `{path}`")
    readme.append("")
    readme.append("## Detailed Files")
    readme.append("")
    readme.append("- `RESULTS_FINAL_BENCHMARK.md`: detailed final benchmark explanation and interpretation.")
    readme.append("- `MEETING_TALK_TRACK.md`: recommended order for presenting the evidence to the supervisor.")
    readme.append("- `PODSUMOWANIE_PL.md`: Polish meeting summary with cautious interpretation of models, validation, SLP, and final benchmark.")
    readme.append("- `SLP_WYJASNIENIE_PL.md`: Polish explanation of the SLP/decode-encode projection prototype.")
    readme.append("- `EXPERIMENTAL_PROCESS.md`: narrative of the experimental process from dataset to final benchmark.")
    readme.append("- `ARCHITECTURES.md`: several-sentence explanation of every tested architecture family.")
    readme.append("- `TRAINING_RESULTS.md`: long-training validation logs and selected-checkpoint evaluation results.")
    readme.append("- `EXPERIMENTAL_RUN_RESULTS.md`: recursive summary of result CSVs under `exports/`, including prototype and cluster runs.")
    readme.append("- `PLOTS_GALLERY.md`: embedded plot gallery for quick browsing during the meeting.")
    readme.append("- `SCRIPTS_INVENTORY.md`: every script in `scripts/`, categorized.")
    readme.append("- `MODELS_AND_CHECKPOINTS.md`: model families, training-run inventory, checkpoint inventory, and final selected labels.")
    readme.append("- `EXPORTS_INVENTORY.md`: exported artifacts and result directories.")
    readme.append("- `THESIS_PROGRESS.md`: current thesis-writing status.")
    readme.append("- `data/*.csv`: machine-readable tables used to generate this summary.")
    readme.append("")
    readme.append("## Caveats")
    readme.append("")
    readme.append("- This pack summarizes local artifacts only. If cluster artifacts are missing locally, they cannot be summarized here.")
    readme.append("- CoSO implementation/results are intentionally not expanded because current thesis workflow treats them as pending source material.")
    readme.append("- Oracle/portfolio plots are diagnostic upper bounds, not methods to claim as final algorithms.")
    readme.append("- Several export directories are copied snapshots of the same cluster artifacts; use the unique-content counts and duplicate-group table when discussing how much independent evidence exists.")
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    talk_md = ["# Meeting Talk Track", ""]
    talk_md.append("Use this as the order for explaining the work. The goal is to show an experimental process, not only a list of files.")
    talk_md.append("")
    talk_rows = [
        {
            "step": "1",
            "topic": "Research setup",
            "say": "The work studies whether learned latent representations of F1 genotypes can support optimization of true Framsticks `vertpos` fitness.",
            "show": "`README.md`, `EXPERIMENTAL_PROCESS.md`",
        },
        {
            "step": "2",
            "topic": "Dataset and representation",
            "say": "The input is a fixed F1 genotype/fitness dataset; representation moved from raw characters toward grammar-rule and derivation-aware encodings because validity matters.",
            "show": "`EXPERIMENTAL_PROCESS.md`, `ARCHITECTURES.md`",
        },
        {
            "step": "3",
            "topic": "Architecture progression",
            "say": "Explain CharVAE, GrammarVAE, masked grammar, TreeVAE, LHS/depth/fitness-aware variants, TransformerVAE, and VQ Grammar AE as successive attempts to improve validity and useful latent structure.",
            "show": "`ARCHITECTURES.md`",
        },
        {
            "step": "4",
            "topic": "Training evidence",
            "say": "Use validation exact reconstruction and similarity to show which autoencoder families became credible candidates. Mention unique logs separately from copied export folders.",
            "show": "`TRAINING_RESULTS.md`, `plots/training_*.png`",
        },
        {
            "step": "5",
            "topic": "Checkpoint selection",
            "say": "The selected-checkpoint stage checked reconstruction, generated-validity, uniqueness, and simple latent-fitness diagnostics before expensive optimization runs.",
            "show": "`TRAINING_RESULTS.md`, `data/selected_checkpoint_evaluation.csv`",
        },
        {
            "step": "6",
            "topic": "Optimization prototypes",
            "say": "Prototype/local/cluster runs tested latent EA, CEM, CMA-ES, direct grammar CEM, and auxiliary surrogate ideas to decide what deserved a paired final benchmark.",
            "show": "`EXPERIMENTAL_RUN_RESULTS.md`, `data/all_export_csv_summary.csv`",
        },
        {
            "step": "7",
            "topic": "Final benchmark",
            "say": "The final claim should be paired and cautious: the best TreeVAE/TransformerVAE latent-EA configurations beat the baseline on mean score, but Framsticks EA remains a strong baseline and keeps the best single record.",
            "show": "`RESULTS_FINAL_BENCHMARK.md`, `existing_final_plots/*.png`",
        },
        {
            "step": "8",
            "topic": "Thesis status and ask",
            "say": "Explain what is already written, what remains TODO, and ask whether the final results should be framed as a competitive LSO result, a representation study, or both.",
            "show": "`THESIS_PROGRESS.md`",
        },
    ]
    talk_md.append(md_table(talk_rows, ["step", "topic", "say", "show"]))
    talk_md.append("")
    talk_md.append("## Claims To Keep Cautious")
    talk_md.append("")
    talk_md.append("- Do not present oracle/portfolio values as an implemented method; they are upper-bound diagnostics.")
    talk_md.append("- Do not count copied export directories as independent experiments; use unique-content counts for evidence volume.")
    talk_md.append("- Do not claim that every latent method beats Framsticks EA; the strongest result is specific to selected model/method configurations, especially latent EA.")
    talk_md.append("- Do not expand CoSO results unless source material is supplied; keep it as pending scope.")
    talk_md.append("")
    talk_md.append("## Questions For The Supervisor")
    talk_md.append("")
    talk_md.append("- Is the final framing acceptable as a cautious comparison against a strong Framsticks EA baseline?")
    talk_md.append("- Should Chapter 4 emphasize the representation-development process, the final paired benchmark, or both equally?")
    talk_md.append("- Which remaining thesis sections should be prioritized before returning to CoSO material?")
    (OUT / "MEETING_TALK_TRACK.md").write_text("\n".join(talk_md) + "\n", encoding="utf-8")

    gallery_md = ["# Plots Gallery", "", "## Newly Generated Meeting Plots", ""]
    for path in [*plot_outputs, *training_plot_outputs]:
        gallery_md.append(f"### {Path(path).stem}")
        gallery_md.append("")
        gallery_md.append(f"![{Path(path).stem}]({Path(path).as_posix().replace('summary_for_meeting/', '')})")
        gallery_md.append("")
    gallery_md.extend(["", "## Existing Final Benchmark Plots", ""])
    for path in copied_plots:
        gallery_md.append(f"### {Path(path).stem}")
        gallery_md.append("")
        gallery_md.append(f"![{Path(path).stem}]({Path(path).as_posix().replace('summary_for_meeting/', '')})")
        gallery_md.append("")
    (OUT / "PLOTS_GALLERY.md").write_text("\n".join(gallery_md) + "\n", encoding="utf-8")

    arch_md = ["# Architecture Explanations", ""]
    arch_md.append("This file is meant for explaining the experimental process verbally. Each architecture is described in a few sentences, including what it changes and why it was tested.")
    arch_md.append("")
    for item in ARCHITECTURE_DETAILS:
        arch_md.append(f"## {item['name']}")
        arch_md.append("")
        arch_md.append(f"**Role:** {item['what']}")
        arch_md.append("")
        arch_md.append(str(item["description"]))
        arch_md.append("")
    (OUT / "ARCHITECTURES.md").write_text("\n".join(arch_md), encoding="utf-8")

    training_md = ["# Training And Checkpoint Results", ""]
    training_md.append("This file summarizes long training logs and the selected-checkpoint evaluation stage. It is the main meeting file for explaining how the autoencoder experiments evolved before the final latent-space benchmark.")
    training_md.append("")
    if unique_training_summary:
        ts = pd.DataFrame(unique_training_summary).sort_values("best_val_exact_acc", ascending=False)
        family = (
            ts.groupby("category")
            .agg(runs=("run_dir", "size"), best_val_exact_acc=("best_val_exact_acc", "max"), best_val_avg_sim=("best_val_avg_sim", "max"))
            .reset_index()
            .sort_values("best_val_exact_acc", ascending=False)
        )
        for col in ("best_val_exact_acc", "best_val_avg_sim"):
            ts[col] = ts[col].map(lambda x: f"{x:.2f}")
            family[col] = family[col].map(lambda x: f"{x:.2f}")
        training_md.append("## Best Validation Reconstruction By Family")
        training_md.append("")
        training_md.append(f"Duplicate copied logs are omitted here: {len(unique_training_summary)} unique log contents from {len(training_summary)} local `training.log` copies. Full copy-level provenance is in `data/training_log_summary.csv`.")
        training_md.append("")
        training_md.append(md_table(family.to_dict("records"), ["category", "runs", "best_val_exact_acc", "best_val_avg_sim"]))
        training_md.append("")
        training_md.append("## Top Training Runs")
        training_md.append("")
        training_md.append(md_table(ts.head(20).to_dict("records"), ["run_dir", "model", "category", "latent_dim", "seed", "best_epoch", "best_val_exact_acc", "best_val_avg_sim", "last_epoch"]))
        training_md.append("")
    else:
        training_md.append("No `training.log` files were parsed.")
        training_md.append("")
    if not selected_eval.empty:
        se = selected_eval.copy().sort_values("log_val_exact_acc", ascending=False)
        keep = [
            "display_label",
            "model",
            "latent_dim",
            "log_best_epoch",
            "log_val_exact_acc",
            "log_val_avg_sim",
            "recon_exact_accuracy",
            "gen_valid_rate",
            "gen_unique_valid",
            "latent_linear_r2_in_sample",
        ]
        for col in keep:
            if col not in se.columns:
                se[col] = ""
        for col in ["log_val_exact_acc", "log_val_avg_sim", "recon_exact_accuracy", "gen_valid_rate", "latent_linear_r2_in_sample"]:
            se[col] = se[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not math.isnan(x) else x)
        training_md.append("## Selected 10 Checkpoint Evaluation")
        training_md.append("")
        training_md.append("These rows summarize the selected checkpoints before final optimization. They include validation reconstruction from logs, local reconstruction checks, validity of generated samples, and a simple in-sample latent-fitness linear relationship diagnostic.")
        training_md.append("")
        training_md.append(md_table(se.to_dict("records"), keep))
        training_md.append("")
    training_md.append("## Training Plots")
    training_md.append("")
    for path in training_plot_outputs:
        training_md.append(f"- `{path}`")
    (OUT / "TRAINING_RESULTS.md").write_text("\n".join(training_md) + "\n", encoding="utf-8")

    process_md = ["# Experimental Process", ""]
    process_md.append("This is the narrative version of the repository state: what was tried, why it was tried, and how the final benchmark emerged.")
    process_md.append("")
    process_md.append("## 1. Dataset And Representation")
    process_md.append("")
    process_md.append("The starting point is the F1 dataset of genotype strings paired with `vertpos` fitness values. The code supports both raw character representations and grammar-production representations. The grammar representation became central because F1 genotypes are structured and validity is easier to control when decoding production rules rather than arbitrary characters.")
    process_md.append("")
    process_md.append("## 2. Baseline Autoencoder Families")
    process_md.append("")
    process_md.append("Early experiments trained character-level, grammar-level, masked grammar, tree-oriented, Transformer, and VQ variants. These runs established that reconstruction quality alone is not enough: a model also needs to decode valid and optimizable genotypes. This motivated increasingly grammar-aware and context-aware decoders.")
    process_md.append("")
    process_md.append("## 3. Long Validation Sweeps")
    process_md.append("")
    process_md.append("Longer sweeps in `models/` and `exports/f1_val_logs_*` tracked validation exact reconstruction and average similarity. These logs were used to select promising checkpoints, especially LHS/depth variants and later TreeVAE/Transformer checkpoints. The meeting pack extracts the best validation metrics from those logs into `TRAINING_RESULTS.md` and `data/training_log_summary.csv`.")
    process_md.append("")
    process_md.append("## 4. Selected Checkpoint Evaluation")
    process_md.append("")
    process_md.append("The `f1_selected_10_ckpts_20260520_184738` export collects ten candidate checkpoints and evaluation CSVs. These evaluations measure reconstruction, validity of generated samples, duplicate rates, and simple latent-fitness diagnostics. This stage narrowed the model set before more expensive latent optimization experiments.")
    process_md.append("")
    process_md.append("## 5. Latent Optimization Prototypes")
    process_md.append("")
    process_md.append("Before the final benchmark, several local and cluster runs tested latent evolution, CEM, CMA-ES, direct grammar CEM, seeded comparisons, and surrogate-guided mutation. These runs helped identify which optimizer/model combinations were promising and which were unstable or weak. They are summarized in `EXPORTS_INVENTORY.md` and the root export CSV summaries.")
    process_md.append("")
    process_md.append("## 6. Final Paired Benchmark")
    process_md.append("")
    process_md.append("The final benchmark compares five selected model labels and three latent optimization methods against a paired Framsticks EA baseline on the same starting genotypes. It contains 150 baseline rows and 2250 latent-method rows. The strongest average results come from TreeVAE/TransformerVAE with latent-space EA, while the Framsticks EA baseline remains strong and has the best single known record.")
    process_md.append("")
    process_md.append("## 7. Current Thesis Status")
    process_md.append("")
    process_md.append("The thesis text currently documents the dataset, grammar representation, autoencoder representation, and latent optimization methods. Results and experimental design chapters still need to be written from the provenance summarized here. CoSO remains outside this meeting pack except as pending thesis scope.")
    (OUT / "EXPERIMENTAL_PROCESS.md").write_text("\n".join(process_md) + "\n", encoding="utf-8")

    scripts_md = ["# Scripts Inventory", "", f"Total scripts: {len(scripts)}", ""]
    for category in sorted(set(row["category"] for row in scripts)):
        scripts_md.extend([f"## {category.title()}", ""])
        rows = [row for row in scripts if row["category"] == category]
        scripts_md.append(md_table(rows, ["script", "description", "size_kb"]))
        scripts_md.append("")
    (OUT / "SCRIPTS_INVENTORY.md").write_text("\n".join(scripts_md), encoding="utf-8")

    models_md = ["# Models And Checkpoints", "", "## Source Model Modules", ""]
    models_md.append(md_table(model_modules, ["module", "classes", "description"]))
    models_md.extend(["", "## Training Run Inventory", "", f"Model/checkpoint run directories found: {len(training_runs)}", ""])
    models_md.append("Full run list: `data/training_run_inventory.csv`. Meeting-oriented sample:")
    training_sample = sorted(training_runs, key=lambda row: (str(row["category"]), str(row["run_dir"])))[:80]
    models_md.append(md_table(training_sample, ["run_dir", "category", "checkpoints", "best_like", "final_like", "config_files", "size_mb", "example_checkpoint"]))
    models_md.extend(["", "## Checkpoint Inventory", "", f"Total checkpoint-like files found: {len(checkpoints)}", ""])
    models_md.append("Full checkpoint list: `data/checkpoint_inventory.csv`. Top-level sample:")
    models_md.append(md_table(checkpoints[:40], ["checkpoint", "root", "size_mb", "modified"]))
    models_md.extend(["", "## Final Selected Labels", ""])
    models_md.append(md_table(label_rows, ["label", "source_label", "description"]))
    (OUT / "MODELS_AND_CHECKPOINTS.md").write_text("\n".join(models_md), encoding="utf-8")

    exports_md = ["# Exports Inventory", "", f"Top-level export entries: {len(exports)}", ""]
    exports_md.append("Entries marked as copied snapshots should not be treated as independent experimental runs.")
    exports_md.append("")
    exports_md.append(md_table(exports, ["entry", "kind", "files", "size_mb", "evidence_note"]))
    exports_md.extend(["", "## Miscellaneous Root CSV Results", ""])
    exports_md.append(md_table(misc_results, ["file", "rows", "columns", "has_best_score", "mean_best_score", "max_best_score"]))
    (OUT / "EXPORTS_INVENTORY.md").write_text("\n".join(exports_md), encoding="utf-8")

    csv_md = ["# Experimental Run Results", ""]
    csv_md.append("This file recursively summarizes CSV result artifacts under `exports/`. It is meant to show the experimental trail before the final benchmark: selected checkpoint evaluations, local prototypes, latest cluster runs, surrogate experiments, and final benchmark job files.")
    csv_md.append("")
    csv_df = pd.DataFrame(all_csv_results)
    if not csv_df.empty:
        unique_csv_df = pd.DataFrame(unique_csv_results)
        raw_counts = csv_df.groupby("category").agg(raw_files=("file", "size"), raw_rows=("rows", numeric_row_sum)).reset_index()
        unique_counts = unique_csv_df.groupby("category").agg(unique_files=("file", "size"), unique_rows=("rows", numeric_row_sum)).reset_index()
        category_counts = raw_counts.merge(unique_counts, on="category", how="left").fillna(0)
        csv_md.append("## Result CSV Categories")
        csv_md.append("")
        csv_md.append("Raw counts include copied export folders. Unique counts collapse exact duplicate CSV contents and are safer for discussing independent evidence volume.")
        csv_md.append("If the same CSV content appears under multiple categories, its unique rows are assigned to the canonical file category shown in the duplicate table.")
        csv_md.append("")
        csv_md.append(md_table(category_counts.to_dict("records"), ["category", "raw_files", "unique_files", "raw_rows", "unique_rows"]))
        csv_md.append("")
        if duplicate_csv_groups:
            csv_md.append("## Duplicate CSV Content Groups")
            csv_md.append("")
            csv_md.append("Exact duplicate CSV contents are usually copied cluster/export snapshots or alias files. Full duplicate groups are stored in `data/all_export_csv_duplicate_groups.csv`.")
            csv_md.append("")
            csv_md.append(md_table(duplicate_csv_groups[:30], ["duplicate_group", "copies", "canonical_file", "categories", "rows_per_copy", "raw_rows_across_copies", "duplicate_files_sample"]))
            csv_md.append("")
        csv_md.append("")
        csv_md.append("## Non-Final Prototype/Exploratory Result Files")
        csv_md.append("")
        non_final = unique_csv_df[~unique_csv_df["category"].astype(str).str.startswith("final benchmark")].copy()
        non_final = non_final.sort_values(["category", "file"])
        csv_md.append("This table shows canonical unique CSV contents only; copied duplicates remain in `data/all_export_csv_summary.csv`.")
        csv_md.append("")
        csv_md.append(md_table(non_final.head(120).to_dict("records"), ["file", "category", "rows", "labels", "methods", "models", "mean_best_score", "max_best_score", "duplicate_count"]))
        csv_md.append("")
        csv_md.append("## Final Benchmark CSV Groups")
        csv_md.append("")
        final_rows = unique_csv_df[unique_csv_df["category"].astype(str).str.startswith("final benchmark")].copy()
        csv_md.append(md_table(final_rows.head(80).to_dict("records"), ["file", "category", "rows", "labels", "methods", "models", "mean_best_score", "max_best_score", "duplicate_count"]))
    else:
        csv_md.append("No CSV files were found under `exports/`.")
    (OUT / "EXPERIMENTAL_RUN_RESULTS.md").write_text("\n".join(csv_md) + "\n", encoding="utf-8")

    results_md = ["# Final Benchmark Results", ""]
    results_md.append("The final benchmark compares latent-space optimization configurations against a paired Framsticks EA baseline on the same starting genotypes.")
    results_md.append("")
    results_md.append("## Completeness")
    results_md.append("")
    results_md.append(f"- CPU baseline rows: {len(cpu)}")
    results_md.append(f"- GPU/latent rows: {len(gpu)}")
    results_md.append(f"- CPU history rows: {final['cpu_history_rows']}")
    results_md.append(f"- GPU history rows: {final['gpu_history_rows']}")
    results_md.append("")
    results_md.append("## Baseline")
    results_md.append("")
    base_rows = [{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in base.items()}]
    results_md.append(md_table(base_rows, list(base_rows[0].keys())))
    results_md.append("")
    results_md.append("## Top Configurations")
    results_md.append("")
    results_md.append(md_table(top_rows.to_dict("records"), ["display_label", "model", "method", "n", "mean_best", "mean_diff_vs_baseline", "winrate_vs_baseline", "max_best"]))
    results_md.append("")
    results_md.append("## Interpretation")
    results_md.append("")
    results_md.append("- Latent-space EA is the strongest latent optimization method on average in the final paired benchmark.")
    results_md.append("- CEM is generally weaker on average, while CMA-ES can find high maxima but is less stable.")
    results_md.append("- Framsticks EA remains a strong baseline and keeps the best single observed record in the known thesis state.")
    results_md.append("- Improvements should be discussed as paired comparisons against the same starting seeds, not as independent aggregate runs.")
    (OUT / "RESULTS_FINAL_BENCHMARK.md").write_text("\n".join(results_md) + "\n", encoding="utf-8")

    thesis_md = ["# Thesis Progress", ""]
    thesis_md.append("Written and reviewed/partly reviewed so far:")
    thesis_md.append("- Abstract and Polish `Streszczenie` skeleton/content present.")
    thesis_md.append("- Chapter 1 complete and reviewed.")
    thesis_md.append("- Chapter 2 sections 2.1, 2.2, 2.5, and 2.6 written; 2.5/2.6 now have real citations for autoencoders/LSO.")
    thesis_md.append("- Chapter 3 sections 3.1, 3.2, 3.4, and 3.5 written and reviewed; CoSO section 3.3 remains intentionally pending.")
    thesis_md.append("- Chapters 4--6 and appendix remain TODO skeletons.")
    thesis_md.append("")
    thesis_md.append("Current workflow safeguards:")
    thesis_md.append("- Experimental provenance protocol for claims about runs/results.")
    thesis_md.append("- Local similarity audit script for literature-heavy sections.")
    thesis_md.append("- Writer/reviewer agents record state in `THESIS_STATE.md` and `THESIS_REVIEW_STATE.md`.")
    (OUT / "THESIS_PROGRESS.md").write_text("\n".join(thesis_md) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    scripts = inventory_scripts()
    model_modules = inventory_source_models()
    checkpoints = inventory_checkpoints()
    training_runs = inventory_training_runs()
    training_summary, training_curves = parse_training_logs()
    selected_eval = selected_checkpoint_evaluation()
    exports = inventory_exports()
    misc_results = inventory_misc_results()
    all_csv_results = inventory_all_export_csv_results()
    final = summarize_final_benchmark()
    plot_outputs = make_plots(final)
    training_plot_outputs = make_training_plots(training_summary, training_curves, selected_eval)
    copied_plots = copy_existing_plots()
    write_overview(scripts, model_modules, checkpoints, training_runs, training_summary, training_curves, selected_eval, exports, misc_results, all_csv_results, final, plot_outputs, training_plot_outputs, copied_plots)
    print(f"Wrote {OUT}")
    print(f"Scripts: {len(scripts)}")
    print(f"Training runs: {len(training_runs)}")
    print(f"Training logs parsed: {len(training_summary)}")
    print(f"Unique training log contents: {len(unique_evidence_rows(training_summary))}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Export CSV files: {len(all_csv_results)}")
    print(f"Unique export CSV contents: {len(unique_evidence_rows(all_csv_results))}")
    print(f"Final CPU rows: {len(final['cpu'])}")
    print(f"Final GPU rows: {len(final['gpu'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
