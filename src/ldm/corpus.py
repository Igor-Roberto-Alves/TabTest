"""Corpus de pré-treino: várias tabelas codificadas e um amostrador de lotes.

Cada lote vem de uma única tabela (todas as linhas têm as mesmas colunas), sorteada
com prob. ∝ n_linhas^temperature: temperature < 1 dá mais peso às tabelas pequenas.
Tabelas largas são cortadas num subconjunto aleatório de colunas, o que também serve
de aumento de dados (o modelo aprende a prever com contextos parciais).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .table import TableEncoder


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["?"], low_memory=False)  # OpenML usa "?" como nulo


def find_tables(dirs, exclude=()) -> list[Path]:
    paths = [p for d in dirs for p in sorted(Path(d).glob("*.csv"))]
    return [p for p in paths if p.stem not in set(exclude)]


def to_torch(arrays: dict, device) -> dict:
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


def subset_columns(arrays: dict, cols) -> dict:
    return {k: v[:, cols] if k in ("local", "z") else v[cols] for k, v in arrays.items()}


@dataclass
class EncodedTable:
    name: str
    encoder: TableEncoder
    local: np.ndarray
    z: np.ndarray

    @classmethod
    def from_frame(cls, name, df, encoder_kwargs=None, kinds=None, exact=()):
        encoder = TableEncoder(**(encoder_kwargs or {})).fit(df, kinds, exact)
        data = encoder.encode(df)
        return cls(name, encoder, data["local"], data["z"])

    @property
    def meta(self):
        return self.encoder.meta()

    def __len__(self):
        return len(self.local)


class TableCorpus:
    def __init__(self, paths, max_rows=50_000, max_cols=64, temperature=0.5, seed=0, encoder_kwargs=None):
        self.rng = np.random.default_rng(seed)
        self.max_cols = max_cols
        self.tables: list[EncodedTable] = []
        for path in paths:
            df = read_csv(path).dropna(axis=1, how="all")
            if df.shape[1] < 2:
                continue
            if len(df) > max_rows:
                df = df.sample(max_rows, random_state=seed)
            self.tables.append(EncodedTable.from_frame(Path(path).stem, df, encoder_kwargs))
        if not self.tables:
            raise ValueError("nenhuma tabela no corpus")
        sizes = np.array([len(t) for t in self.tables], dtype=float)
        self.probs = sizes ** temperature / (sizes ** temperature).sum()

    @property
    def names(self):
        return [t.name for t in self.tables]

    def sample(self, batch_size, device):
        """Retorna (nome, local, z, meta) de um lote de uma tabela sorteada."""
        table = self.tables[self.rng.choice(len(self.tables), p=self.probs)]
        rows = self.rng.integers(0, len(table), batch_size)
        C = table.local.shape[1]
        cols = np.sort(self.rng.choice(C, self.max_cols, replace=False)) if C > self.max_cols else np.arange(C)
        data = subset_columns({"local": table.local[rows], "z": table.z[rows]}, cols)
        meta = subset_columns(table.meta, cols)
        data, meta = to_torch(data, device), to_torch(meta, device)
        return table.name, data["local"], data["z"], meta
