# HOWTO: Script Usage Examples

This file shows practical command examples for every script in `scripts/`.

Assumptions:

- You run commands from project root (`Magisterka/`)
- Dependencies are installed: `python -m pip install -r requirements.txt`
- Dataset exists at `datasets/f1/f1_dataset.txt`

## 1) `scripts/train.py` (unified trainer)

### Example 1: Minimal char-VAE training

```bash
python scripts/train.py --model char_vae --data-path datasets/f1/f1_dataset.txt --output-root models/f1
```

### Example 2: Full config-driven run

```bash
python scripts/train.py --config-model configs/model/grammar_vae.yaml --config-data configs/data/f1.yaml --config-train configs/train/base.yaml --output-root models/f1
```

### Example 3: Override YAML with CLI flags

```bash
python scripts/train.py --config-model configs/model/grammar_vae_masked.yaml --config-data configs/data/f1.yaml --config-train configs/train/base.yaml --epochs 300 --learning-rate 0.0005 --checkpoint-every 25 --output-root models/f1
```

### Example 4: Deterministic-ish run with seed

```bash
python scripts/train.py --model char_vae --data-path datasets/f1/f1_dataset.txt --output-root models/f1 --seed 42 --epochs 100
```

## 2) `scripts/eval.py` (unified evaluator)

Replace `<WEIGHTS_PATH>` with your `.pth` file, e.g. `models/f1/char_vae/<RUN_ID>/char_vae.pth`.

### Example 1: Reconstruction metrics (char model)

```bash
python scripts/eval.py --model char_vae --weights <WEIGHTS_PATH> --mode reconstruct
```

### Example 2: Reconstruction on subset only

```bash
python scripts/eval.py --model grammar_vae --weights <WEIGHTS_PATH> --mode reconstruct --max-items 500
```

### Example 3: Mutation sampling (masked grammar)

```bash
python scripts/eval.py --model grammar_vae_masked --weights <WEIGHTS_PATH> --mode mutate --num-samples 20 --noise-scale 0.8
```

### Example 4: Config-driven evaluation + forced CPU

```bash
python scripts/eval.py --config-model configs/model/char_vae.yaml --config-data configs/data/f1.yaml --weights <WEIGHTS_PATH> --mode reconstruct --device cpu
```

## 2.1 Cache behavior (grammar datasets)

For grammar-based datasets, parsed tensors are cached automatically in `datasets/.../.cache/`.

- First run: parses full dataset and writes cache
- Next runs: loads cache directly (no full parse)

If you edit the dataset file, cache is automatically invalidated using file metadata and rebuilt.

## 3) `scripts/infer.py` (single-sample inference without full dataset)

Replace `<WEIGHTS_PATH>` and input strings with your own values.

### Example 1: Char model reconstruction

```bash
python scripts/infer.py --model char_vae --weights <WEIGHTS_PATH> --mode reconstruct --input-string "XRRX"
```

### Example 2: Char model latent mutation

```bash
python scripts/infer.py --model char_vae --weights <WEIGHTS_PATH> --mode mutate --noise-scale 0.7 --input-string "XRRX"
```

### Example 3: Grammar model reconstruction

```bash
python scripts/infer.py --model grammar_vae --weights <WEIGHTS_PATH> --mode reconstruct --input-string "RX"
```

### Example 4: Config-driven masked grammar mutation

```bash
python scripts/infer.py --config-model configs/model/grammar_vae_masked.yaml --weights <WEIGHTS_PATH> --mode mutate --noise-scale 1.0 --input-string "R(X,X)"
```

## 4) `scripts/train_f1_char.py` (quick char-VAE training)

### Example 1: Standard run

```bash
python scripts/train_f1_char.py
```

### Example 2: Unbuffered logs (better live monitoring)

```bash
python -u scripts/train_f1_char.py
```

### Example 3: Save terminal output to file

```bash
python scripts/train_f1_char.py > char_train.log
```

### Example 4: Explicit interpreter path

```bash
py -3.10 scripts/train_f1_char.py
```

## 5) `scripts/train_f1_grammar.py` (quick grammar-rule VAE training)

### Example 1: Standard run

```bash
python scripts/train_f1_grammar.py
```

### Example 2: Unbuffered logs

```bash
python -u scripts/train_f1_grammar.py
```

### Example 3: Save output to file

```bash
python scripts/train_f1_grammar.py > grammar_train.log
```

### Example 4: Explicit interpreter path

```bash
py -3.10 scripts/train_f1_grammar.py
```

## 6) `scripts/train_f1_grammar_masked.py` (quick masked grammar VAE training)

### Example 1: Standard run

```bash
python scripts/train_f1_grammar_masked.py
```

### Example 2: Unbuffered logs

```bash
python -u scripts/train_f1_grammar_masked.py
```

### Example 3: Save output to file

```bash
python scripts/train_f1_grammar_masked.py > grammar_masked_train.log
```

### Example 4: Explicit interpreter path

```bash
py -3.10 scripts/train_f1_grammar_masked.py
```

## 7) Typical end-to-end flows

### Flow A: Train char model and evaluate reconstruction

```bash
python scripts/train.py --model char_vae --data-path datasets/f1/f1_dataset.txt --output-root models/f1
python scripts/eval.py --model char_vae --weights models/f1/char_vae/<RUN_ID>/char_vae.pth --mode reconstruct
```

### Flow B: Train masked grammar model and sample mutations

```bash
python scripts/train.py --model grammar_vae_masked --data-path datasets/f1/f1_dataset.txt --output-root models/f1
python scripts/eval.py --model grammar_vae_masked --weights models/f1/grammar_vae_masked/<RUN_ID>/grammar_vae_masked.pth --mode mutate --num-samples 15 --noise-scale 1.0
```
