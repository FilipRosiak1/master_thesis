from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from pathlib import Path

import nltk
import numpy as np
import torch
from torch.utils.data import Dataset

from f1vae.grammars import f1 as G


def _file_meta(filepath: str) -> dict:
    path = Path(filepath).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _default_cache_path(filepath: str, dataset_kind: str, max_length: int) -> str:
    source = str(Path(filepath).resolve())
    digest = md5(source.encode("utf-8")).hexdigest()[:8]
    cache_dir = Path(filepath).resolve().parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{dataset_kind}_max{max_length}_{digest}.pt")


def _meta_matches(
    cache_meta: dict, *, filepath: str, max_length: int, dataset_kind: str
) -> bool:
    if not isinstance(cache_meta, dict):
        return False
    current = _file_meta(filepath)
    return (
        cache_meta.get("dataset_kind") == dataset_kind
        and cache_meta.get("max_length") == max_length
        and cache_meta.get("source") == current
        and cache_meta.get("grammar_hash") == md5(G.gram.encode("utf-8")).hexdigest()
        and cache_meta.get("dataset_format_version") == 4
    )


def _parse_dataset_row(line: str) -> tuple[str, float | None]:
    row = line.strip()
    if not row:
        return "", None

    # New format: genotype<TAB>fitness
    if "\t" in row:
        genotype, fitness_raw = row.split("\t", 1)
        genotype = genotype.strip()
        fitness_raw = fitness_raw.strip()
        if not genotype:
            return "", None
        if not fitness_raw:
            return genotype, None
        try:
            return genotype, float(fitness_raw)
        except ValueError:
            return genotype, None

    # Old format: genotype only
    return row, None


@dataclass
class Vocabulary:
    vocab: list[str]
    char2idx: dict[str, int]
    idx2char: dict[int, str]


class CharGenotypeDataset(Dataset):
    def __init__(self, filepath: str, max_length: int) -> None:
        with open(filepath, "r", encoding="utf-8") as handle:
            raw_rows = [line.strip() for line in handle if line.strip()]

        lines: list[str] = []
        fitnesses: list[float | None] = []
        for row in raw_rows:
            genotype, fitness = _parse_dataset_row(row)
            if genotype:
                lines.append(genotype)
                fitnesses.append(fitness)

        chars = set("".join(lines))
        vocab = ["<PAD>", "<SOS>", "<EOS>"] + sorted(list(chars))
        char2idx = {ch: i for i, ch in enumerate(vocab)}
        idx2char = {i: ch for i, ch in enumerate(vocab)}

        encoded_data = []
        for line in lines:
            encoded = (
                [char2idx["<SOS>"]] + [char2idx[c] for c in line] + [char2idx["<EOS>"]]
            )
            if len(encoded) < max_length:
                encoded.extend([char2idx["<PAD>"]] * (max_length - len(encoded)))
            else:
                encoded = encoded[:max_length]
            encoded_data.append(encoded)

        self.lines = lines
        self.fitnesses = fitnesses
        self.max_length = max_length
        self.vocabulary = Vocabulary(vocab=vocab, char2idx=char2idx, idx2char=idx2char)
        self.data = torch.tensor(encoded_data, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary.vocab)

    @property
    def pad_idx(self) -> int:
        return self.vocabulary.char2idx["<PAD>"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class GrammarRuleDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        max_length: int,
        *,
        use_cache: bool = True,
        cache_path: str | None = None,
        rebuild_cache: bool = False,
    ) -> None:
        self.max_length = max_length
        self.grammar = G.GCFG
        self.parser = nltk.BottomUpChartParser(self.grammar)
        self.productions = self.grammar.productions()
        self.rule2idx = {prod: i for i, prod in enumerate(self.productions)}
        self.idx2rule = {i: prod for i, prod in enumerate(self.productions)}

        self.pad_rule_idx = len(self.productions)
        self.num_classes = len(self.productions) + 1

        cache_path = cache_path or _default_cache_path(
            filepath, "grammar_rule", max_length
        )
        if use_cache and not rebuild_cache and Path(cache_path).exists():
            payload = torch.load(cache_path, map_location="cpu")
            if _meta_matches(
                payload.get("meta", {}),
                filepath=filepath,
                max_length=max_length,
                dataset_kind="grammar_rule",
            ):
                self.valid_lines = payload["valid_lines"]
                self.valid_fitnesses = payload.get(
                    "valid_fitnesses", [None] * len(self.valid_lines)
                )
                self.data = payload["data"]
                return

        with open(filepath, "r", encoding="utf-8") as handle:
            raw_rows = [line.strip() for line in handle if line.strip()]

        lines: list[str] = []
        fitnesses: list[float | None] = []
        for row in raw_rows:
            genotype, fitness = _parse_dataset_row(row)
            if genotype:
                lines.append(genotype)
                fitnesses.append(fitness)

        encoded_rules = []
        valid_lines = []
        valid_fitnesses: list[float | None] = []
        for line, fitness in zip(lines, fitnesses):
            tokens = list(line)
            try:
                trees = list(self.parser.parse(tokens))
                if not trees:
                    continue
                productions_seq = trees[0].productions()
                if len(productions_seq) > self.max_length:
                    continue
                encoded = [self.rule2idx[prod] for prod in productions_seq]
                encoded.extend([self.pad_rule_idx] * (self.max_length - len(encoded)))
                encoded_rules.append(encoded)
                valid_lines.append(line)
                valid_fitnesses.append(fitness)
            except Exception:
                continue

        self.valid_lines = valid_lines
        self.valid_fitnesses = valid_fitnesses
        self.data = torch.tensor(encoded_rules, dtype=torch.long)

        if use_cache:
            torch.save(
                {
                    "meta": {
                        "dataset_kind": "grammar_rule",
                        "source": _file_meta(filepath),
                        "max_length": max_length,
                        "grammar_hash": md5(G.gram.encode("utf-8")).hexdigest(),
                        "dataset_format_version": 4,
                    },
                    "valid_lines": self.valid_lines,
                    "valid_fitnesses": self.valid_fitnesses,
                    "data": self.data,
                },
                cache_path,
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class GrammarOneHotDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        max_length: int,
        *,
        use_cache: bool = True,
        cache_path: str | None = None,
        rebuild_cache: bool = False,
    ) -> None:
        self.max_length = max_length
        self.parser = nltk.BottomUpChartParser(G.GCFG)
        self.productions = G.GCFG.productions()
        self.prod_map = {prod: i for i, prod in enumerate(self.productions)}
        self.n_chars = len(self.productions)

        cache_path = cache_path or _default_cache_path(
            filepath, "grammar_onehot", max_length
        )
        if use_cache and not rebuild_cache and Path(cache_path).exists():
            payload = torch.load(cache_path, map_location="cpu")
            if _meta_matches(
                payload.get("meta", {}),
                filepath=filepath,
                max_length=max_length,
                dataset_kind="grammar_onehot",
            ):
                self.valid_lines = payload["valid_lines"]
                self.valid_fitnesses = payload.get(
                    "valid_fitnesses", [None] * len(self.valid_lines)
                )
                self.data = payload["data"]
                return

        with open(filepath, "r", encoding="utf-8") as handle:
            raw_rows = [line.strip() for line in handle if line.strip()]

        lines: list[str] = []
        fitnesses: list[float | None] = []
        for row in raw_rows:
            genotype, fitness = _parse_dataset_row(row)
            if genotype:
                lines.append(genotype)
                fitnesses.append(fitness)

        one_hot_data = []
        valid_lines = []
        valid_fitnesses: list[float | None] = []
        for line, fitness in zip(lines, fitnesses):
            tokens = list(line)
            try:
                trees = list(self.parser.parse(tokens))
                if not trees:
                    continue
                productions_seq = trees[0].productions()
                if len(productions_seq) > self.max_length:
                    continue

                indices = [self.prod_map[prod] for prod in productions_seq]
                one_hot = np.zeros((self.max_length, self.n_chars), dtype=np.float32)
                for t, idx in enumerate(indices):
                    one_hot[t, idx] = 1.0
                if len(indices) < self.max_length:
                    one_hot[len(indices) :, -1] = 1.0

                one_hot_data.append(one_hot)
                valid_lines.append(line)
                valid_fitnesses.append(fitness)
            except Exception:
                continue

        self.valid_lines = valid_lines
        self.valid_fitnesses = valid_fitnesses
        self.data = torch.tensor(np.array(one_hot_data), dtype=torch.float32)

        if use_cache:
            torch.save(
                {
                    "meta": {
                        "dataset_kind": "grammar_onehot",
                        "source": _file_meta(filepath),
                        "max_length": max_length,
                        "grammar_hash": md5(G.gram.encode("utf-8")).hexdigest(),
                        "dataset_format_version": 4,
                    },
                    "valid_lines": self.valid_lines,
                    "valid_fitnesses": self.valid_fitnesses,
                    "data": self.data,
                },
                cache_path,
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
