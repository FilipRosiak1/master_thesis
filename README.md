# F1 VAE Refactor

This repository now uses a modular layout to make model-architecture experiments easier and reproducible.

## 1) Environment setup

### Requirements

- Python `3.10+` recommended
- `pip` available in your environment

### Install dependencies

From project root:

```bash
python -m pip install -r requirements.txt
```

### (First run) NLTK note

The code uses `nltk` CFG parsing. If your environment is fresh and you get NLTK resource errors, run once:

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 2) Project structure

- `src/f1vae/data` - dataset parsing/encoding classes
- `src/f1vae/grammars` - grammar definitions and grammar masks
- `src/f1vae/models` - model architectures
- `src/f1vae/training` - shared losses and training loop
- `src/f1vae/inference` - decode and evaluation utilities
- `src/f1vae/optimization` - latent-space optimization algorithms (CMA-ES, CEM)
- `src/f1vae/experiments` - CLI-backed experiment entrypoints
- `scripts/train.py` - unified training CLI wrapper
- `scripts/eval.py` - unified evaluation CLI wrapper
- `scripts/infer.py` - single-sample inference (no full-dataset initialization)
- `scripts/optimize_latent.py` - optimize latent `z` with pluggable fitness function
- `scripts/workbench.py` - one entrypoint for train/eval/infer/optimize and full pipeline
- `scripts/workbench_ui.py` - Streamlit UI for running the same workflows visually
- `scripts/train_f1_*.py` - quick architecture-specific training scripts
- `tests/test_smoke_forward.py` - smoke tests for dataset+forward+loss

### 2.1 Top-level folders (what each one is for)

- `src/` - main Python package code used by training/evaluation CLIs
- `scripts/` - runnable command-line entry scripts
- `configs/` - YAML config snapshots/presets for data/model/train settings
- `datasets/` - input datasets (currently F1 genotype text data)
- `tests/` - automated tests for the refactored code
- `legacy/` - archived older scripts and older external code kept for reference/compatibility
- `models/` - default output location for newly trained weights/checkpoints/logs
- `papers/`, `master_thesis/` - documentation/research material
- `temp/` - temporary/vendor experiments not used by default pipeline

### 2.2 Detailed file map

#### `src/f1vae/`

- `src/f1vae/__init__.py`
  - package marker

- `src/f1vae/config/defaults.py`
  - per-architecture default hyperparameters
  - defines defaults for `char_vae`, `grammar_vae`, `grammar_vae_masked`, `tree_vae`, `tree_vae_masked`, `transformer_vae`, `vq_grammar_ae`

- `src/f1vae/grammars/f1.py`
  - F1 grammar definition string
  - builds `nltk.CFG`
  - precomputes grammar masks (`masks`, `lhs_list`, `ind_of_ind`) used in masked decoding/loss

- `src/f1vae/data/datasets.py`
  - `CharGenotypeDataset`: character-level tokenized/padded dataset
  - `GrammarRuleDataset`: production-rule index dataset
  - `GrammarOneHotDataset`: one-hot production tensor dataset for masked grammar VAE
  - grammar datasets use on-disk cache in `datasets/.../.cache/` by default to avoid reparsing each run

- `src/f1vae/models/char_vae.py`
  - character sequence VAE architecture (GRU encoder/decoder)

- `src/f1vae/models/grammar_vae.py`
  - rule-index grammar VAE architecture (GRU encoder/decoder)

- `src/f1vae/models/grammar_vae_masked.py`
  - masked grammar VAE architecture (Conv encoder + GRU decoder)
  - includes grammar-constrained masked sampling (`decode_masked`)

- `src/f1vae/models/tree_vae.py`
  - tree-aware grammar VAE with context-enriched rule encoding

- `src/f1vae/models/tree_vae_masked.py`
  - tree-aware grammar VAE with stack-constrained masked grammar decoding

- `src/f1vae/models/transformer_vae.py`
  - transformer-based grammar VAE

- `src/f1vae/models/vq_grammar_ae.py`
  - VQ latent grammar autoencoder with codebook quantization

- `src/f1vae/models/registry.py`
  - maps model name -> dataset class and model constructor
  - central place enabling architecture swapping with one CLI

- `src/f1vae/training/losses.py`
  - shared loss functions:
    - sequence CE + KL loss (`sequence_vae_loss`)
    - grammar-masked CE + KL loss (`grammar_masked_vae_loss`)

- `src/f1vae/training/loops.py`
  - shared training loop used by all architectures
  - handles dataloaders, scheduler logic (beta/tf), logging, checkpoints, run directory creation

- `src/f1vae/experiments/train.py`
  - argument parser and entrypoint for unified train CLI

- `src/f1vae/inference/decode.py`
  - decoding helpers:
    - char index -> string
    - grammar production indices -> string
    - deterministic masked grammar decode helper

- `src/f1vae/inference/evaluate.py`
  - reusable eval routines:
    - reconstruction metrics
    - mutation sampling examples

- `src/f1vae/optimization/cmaes.py`
  - CMA-ES optimizer (uses `python-cma` backend if installed, otherwise internal fallback)

- `src/f1vae/optimization/cem.py`
  - CEM optimizer

- `src/f1vae/optimization/types.py`
  - shared optimization result/objective types

#### `scripts/`

- `scripts/train.py`
  - unified training wrapper (`python scripts/train.py ...`)

- `scripts/train_f1_char.py`
  - quick train entrypoint for `char_vae`

- `scripts/train_f1_grammar.py`
  - quick train entrypoint for `grammar_vae`

- `scripts/train_f1_grammar_masked.py`
  - quick train entrypoint for `grammar_vae_masked`

- `scripts/eval.py`
  - unified eval CLI for reconstruction and mutation modes

- `scripts/infer.py`
  - inference on one genotype string without loading/parsing the full dataset

- `scripts/optimize_latent.py`
  - latent search script for maximizing user-provided genotype fitness

- `scripts/workbench.py`
  - unified launcher with subcommands: `train`, `eval`, `infer`, `optimize`, `pipeline`, `menu`

- `scripts/workbench_ui.py`
  - Streamlit frontend for train/eval/infer/optimize/pipeline with command output panels

#### `configs/`

- `configs/data/f1.yaml`
  - dataset path preset

- `configs/model/char_vae.yaml`
  - char model preset values

- `configs/model/grammar_vae.yaml`
  - grammar rule model preset values

- `configs/model/grammar_vae_masked.yaml`
  - masked grammar model preset values

- `configs/model/tree_vae.yaml`
  - tree-aware grammar VAE preset values

- `configs/model/tree_vae_masked.yaml`
  - masked tree-aware grammar VAE preset values

- `configs/model/transformer_vae.yaml`
  - transformer grammar VAE preset values

- `configs/model/vq_grammar_ae.yaml`
  - VQ grammar autoencoder preset values

- `configs/train/base.yaml`
  - training preset values (batch, epochs, lr, checkpoint interval)

These files are now actively used by CLI options `--config-model`, `--config-data`, and `--config-train`.
CLI flags always override YAML values.

#### `datasets/`

- `datasets/f1/f1_dataset.txt`
  - main training/evaluation dataset (one genotype per line)

#### `tests/`

- `tests/test_smoke_forward.py`
  - small synthetic-data smoke tests for all model families
  - validates dataset parsing, forward pass, and finite loss

#### `legacy/`

- `legacy/train_f1_vae.py`, `legacy/train_grammar_vae.py`, `legacy/train_grammar_vae_better.py`
  - archived wrappers kept for backward compatibility patterns

- `legacy/test_*.py`
  - archived standalone evaluation scripts

- `legacy/f1_grammar.py`
  - compatibility re-export path for old imports

- `legacy/grammarVAE/`, `legacy/temp/`
  - archived original/reference project trees not used by default refactored pipeline

## 3) Training

You can train in two ways:

1. quick architecture script
2. unified CLI with all parameters

### 3.1 Quick scripts

```bash
python scripts/train_f1_char.py
python scripts/train_f1_grammar.py
python scripts/train_f1_grammar_masked.py
```

These use defaults from `src/f1vae/config/defaults.py` and dataset `datasets/f1/f1_dataset.txt`.

### 3.2 Unified training CLI

```bash
python scripts/train.py --model char_vae --data-path datasets/f1/f1_dataset.txt --output-root models/f1
```

Config-driven example:

```bash
python scripts/train.py --config-model configs/model/char_vae.yaml --config-data configs/data/f1.yaml --config-train configs/train/base.yaml --output-root models/f1
```

Supported models:

- `char_vae`
- `grammar_vae`
- `grammar_vae_masked`
- `tree_vae`
- `transformer_vae`
- `vq_grammar_ae`

#### All training parameters

- `--config-model`
  - type: string (path)
  - default: `None`
  - description: YAML with model family + model hyperparameters
- `--config-data`
  - type: string (path)
  - default: `None`
  - description: YAML with dataset path
- `--config-train`
  - type: string (path)
  - default: `None`
  - description: YAML with training settings
- `--model` (required unless set in `--config-model`)
  - type: choice
  - values: `char_vae`, `grammar_vae`, `grammar_vae_masked`, `tree_vae`, `transformer_vae`, `vq_grammar_ae`
  - description: architecture to train
- `--data-path` (required unless set in `--config-data`)
  - type: string (path)
  - description: input genotype dataset file
- `--output-root`
  - type: string (path)
  - default: `models/f1`
  - description: base output directory
- `--latent-dim`
  - type: int
  - default: per-model default
  - description: latent vector size
- `--hidden-dim`
  - type: int
  - default: per-model default
  - description: hidden size (used in recurrent/convolutional heads where applicable)
- `--embedding-dim`
  - type: int
  - default: per-model default
  - description: token/rule embedding size (for sequence models)
- `--batch-size`
  - type: int
  - default: per-model default
  - description: training batch size
- `--epochs`
  - type: int
  - default: per-model default
  - description: number of epochs
- `--learning-rate`
  - type: float
  - default: per-model default
  - description: optimizer learning rate
- `--max-length`
  - type: int
  - default: per-model default
  - description: max sequence/rule length used by dataset/model
- `--checkpoint-every`
  - type: int
  - default: `50`
  - description: checkpoint interval in epochs
- `--seed`
  - type: int
  - default: `None`
  - description: random seed for torch reproducibility

### 3.3 Training outputs

Each run creates:

`<output-root>/<model>/<YYYY-MM-DD_HH-MM-SS>/`

with:

- `run_config.json` - exact run parameters
- `training.log` - periodic loss logs
- `checkpoints/epoch_<N>.pth` - periodic checkpoints
- `<model>.pth` - final model weights

Checkpoint files now store both `state_dict` and `meta` (model settings and extra info like char vocab), so inference can run without rebuilding dataset objects.

For newer checkpoints, metadata also includes the training `data_path`, which is used as a fallback in evaluation.

## 4) Evaluation

Use the shared evaluator for either reconstruction metrics or mutation sampling.

### 4.1 Reconstruction mode

```bash
python scripts/eval.py --weights <path_to_weights.pth> --mode reconstruct
```

Config-driven example:

```bash
python scripts/eval.py --config-model configs/model/char_vae.yaml --config-data configs/data/f1.yaml --weights <path_to_weights.pth> --mode reconstruct
```

### 4.2 Mutation mode

```bash
python scripts/eval.py --weights <path_to_weights.pth> --mode mutate --num-samples 10 --noise-scale 1.0
```

### 4.3 All evaluation parameters

- `--config-model`
  - type: string (path)
  - default: `None`
  - description: YAML with model family + model hyperparameters
- `--config-data`
  - type: string (path)
  - default: `None`
  - description: YAML with dataset path
- `--model` (optional)
  - type: choice
  - values: `char_vae`, `grammar_vae`, `grammar_vae_masked`, `tree_vae`, `transformer_vae`, `vq_grammar_ae`
  - description: architecture used to load/build model (fallback order: `--model` -> `--config-model` -> checkpoint metadata)
- `--weights` (required)
  - type: string (path)
  - description: `.pth` state dict path
- `--data-path`
  - type: string (path)
  - default: from `--config-data`, then checkpoint metadata (`data_path`), then `datasets/f1/f1_dataset.txt`
  - description: dataset used for evaluation
- `--mode`
  - type: choice
  - values: `reconstruct`, `mutate`
  - default: `reconstruct`
- `--device`
  - type: string
  - default: auto (`cuda` if available, else `cpu`)
  - description: force a specific device
- `--max-items`
  - type: int
  - default: `None`
  - description: optional cap for reconstruction sample count
- `--num-samples`
  - type: int
  - default: `10`
  - description: number of mutation examples (used in `mutate` mode)
- `--noise-scale`
  - type: float
  - default: `1.0`
  - description: latent mutation scale
- `--latent-dim`
  - type: int
  - default: from `--config-model`, then checkpoint metadata, then per-model default
  - description: override latent size when loading model
- `--hidden-dim`
  - type: int
  - default: from `--config-model`, then checkpoint metadata, then per-model default
  - description: override hidden size when loading model
- `--embedding-dim`
  - type: int
  - default: from `--config-model`, then checkpoint metadata, then per-model default
  - description: override embedding size when loading model
- `--max-length`
  - type: int
  - default: from `--config-model`, then checkpoint metadata, then per-model default
  - description: override sequence length when loading model

## 5) Latent optimization with external fitness

Use `scripts/optimize_latent.py` to search latent vector `z` directly and decode candidate genotypes.

Two built-in algorithms are available:

- `cmaes` - covariance matrix adaptation evolution strategy
- `cem` - cross-entropy method

For `cmaes`, the script can use the external `python-cma` package automatically when available.
Install it optionally with:

```bash
python -m pip install cma
```

The fitness function is passed as a parameter and loaded dynamically.

- format: `module:function`
- or file path format: `path/to/file.py:function`

Fitness function signature:

```python
def fitness(genotype: str) -> float:
    ...
```

Example:

```bash
python scripts/optimize_latent.py --weights <path_to_weights.pth> --fitness-fn my_fitness:fitness --algorithm cmaes --iterations 100
```

`scripts/optimize_latent.py` and `scripts/infer.py` use the same metadata fallback order for model hyperparameters (`CLI -> config -> checkpoint meta -> defaults`).

Main parameters:

- `--algorithm`: `cmaes` or `cem`
- `--cma-backend`: `auto` (default, prefer `python-cma` if installed) or `internal`
- `--iterations`: optimization iterations/generations
- `--population-size`: number of candidates per iteration
- `--fitness-fn`: required callable target (`module:function` or `file.py:function`)
- `--seed`: optional RNG seed

Algorithm-specific parameters:

- CMA-ES: `--initial-sigma`
- CEM: `--elite-fraction`, `--initial-std`, `--smoothing`, `--min-std`

## 6) Legacy compatibility scripts

Old entrypoints are archived in `legacy/` as wrappers to the new modules:

- `legacy/train_f1_vae.py`
- `legacy/train_grammar_vae.py`
- `legacy/train_grammar_vae_better.py`
- `legacy/f1_grammar.py`

These remain to avoid breaking older workflows and imports.

## 7) Tests

Smoke tests:

```bash
python -m pytest tests/test_smoke_forward.py
```

What they check:

- tiny synthetic dataset parsing
- model forward pass for all supported architecture families
- loss computation stability (`torch.isfinite`)

## 8) Helpful commands

Show all training args:

```bash
python scripts/train.py --help
```

Run a full project pipeline from one command:

```bash
python scripts/workbench.py pipeline --model grammar_vae_masked --data-path datasets/f1/f1_dataset.txt
```

Forward arguments to any existing script:

```bash
python scripts/workbench.py train -- --model tree_vae --data-path datasets/f1/f1_dataset.txt --output-root models/f1
python scripts/workbench.py optimize -- --model grammar_vae_masked --weights <path_to_weights.pth> --fitness-fn my_fitness:fitness --algorithm cmaes
```

Run the web UI:

```bash
streamlit run scripts/workbench_ui.py
```

Show all evaluation args:

```bash
python scripts/eval.py --help
```

Single-sample inference args:

```bash
python scripts/infer.py --help
```
