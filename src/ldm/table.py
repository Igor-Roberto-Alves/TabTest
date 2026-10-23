"""Codificador por tabela do Large Data Model.

Nada aqui gera ids específicos de uma tabela que o modelo precise memorizar. Cada
coluna é descrita pelo texto do seu nome, e cada opção de valor por uma descrição
que o modelo sabe embutir em qualquer tabela:

    categórica -> texto da categoria (ids de hash, ver text.py)
    numérica   -> bin de quantil, descrito pela posição no CDF do treino (q)
                  e pela mediana do bin padronizada (z)

Índice local de uma célula: 0 = [NULL]; 1..n_options = categoria ou bin.
Numéricas guardam também o valor contínuo padronizado (z), usado na entrada
e como alvo da head de regressão.
"""

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .text import MAX_PIECES, encode_texts

NUM, CAT = 0, 1
LOCAL_NULL = 0
OTHER = "<other>"  # categoria rara ou cortada por max_categories
Z_CLIP = 5.0


@dataclass
class ColumnSpec:
    name: str
    kind: int
    categories: list = field(default_factory=list)  # CAT: em ordem de índice (1..)
    edges: list = field(default_factory=list)       # NUM: fronteiras internas dos bins
    bin_values: list = field(default_factory=list)  # NUM: mediana de cada bin (escala original)
    bin_q: list = field(default_factory=list)       # NUM: posição de cada bin no CDF do treino
    mean: float = 0.0
    std: float = 1.0

    @property
    def n_options(self):
        return len(self.categories) if self.kind == CAT else len(self.edges) + 1

    def standardize(self, x):
        return np.clip((np.asarray(x, dtype=float) - self.mean) / self.std, -Z_CLIP, Z_CLIP)


def _key(v):
    """Texto estável de uma categoria (1.0 e 1 viram "1")."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


class TableEncoder:
    """Ajusta numa tabela e converte linhas <-> (índices locais, z).

    n_bins:         máximo de bins de quantil por coluna numérica
    max_cat_levels: numérica com até esse número de valores distintos vira categórica
    max_categories: categorias mais frequentes mantidas; o resto vira OTHER
    min_frequency:  categoria com menos ocorrências que isso vira OTHER
    """

    def __init__(self, n_bins=32, max_cat_levels=10, max_categories=64, min_frequency=2):
        self.n_bins = n_bins
        self.max_cat_levels = max_cat_levels
        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self.columns: list[ColumnSpec] = []
        self._meta = None

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, kinds=None, exact=()):
        """kinds: força o tipo de colunas ({nome: NUM|CAT});
        exact: colunas cujas categorias não são cortadas (ex.: o alvo de uma classificação)."""
        kinds = kinds or {}
        self.columns = []
        for name in df.columns:
            kind = kinds.get(name, self._detect_kind(df[name]))
            spec = self._fit_num(name, df[name]) if kind == NUM else self._fit_cat(name, df[name], name in exact)
            self.columns.append(spec)
        self._meta = None
        return self

    def _detect_kind(self, s):
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            return CAT
        return NUM if s.nunique(dropna=True) > self.max_cat_levels else CAT

    def _fit_cat(self, name, s, exact):
        counts = s.dropna().map(_key).value_counts()  # já ordenado por frequência
        if exact:
            cats = counts.index.tolist()
        else:
            keep = counts[counts >= self.min_frequency].index[: self.max_categories - 1].tolist()
            cats = keep + ([OTHER] if len(keep) < len(counts) else [])
        return ColumnSpec(name, CAT, categories=cats or [OTHER])

    def _fit_num(self, name, s):
        x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(x) == 0:  # coluna toda nula: um único bin
            return ColumnSpec(name, NUM, bin_values=[0.0], bin_q=[0.5])
        # quantis repetidos colapsam (np.unique), então colunas discretas têm menos bins
        edges = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        bins = np.searchsorted(edges, x, side="right")
        counts = np.bincount(bins, minlength=len(edges) + 1)
        bounds = np.concatenate([[x.min()], edges, [x.max()]])
        medians = [float(np.median(x[bins == b])) if counts[b] else float((bounds[b] + bounds[b + 1]) / 2)
                   for b in range(len(edges) + 1)]
        q = (np.cumsum(counts) - counts / 2) / len(x)
        return ColumnSpec(name, NUM, edges=edges.tolist(), bin_values=medians, bin_q=q.tolist(),
                          mean=float(x.mean()), std=float(x.std()) or 1.0)

    # --------------------------------------------------------- propriedades
    @property
    def column_names(self):
        return [c.name for c in self.columns]

    def column_index(self, name):
        return self.column_names.index(name)

    def meta(self) -> dict[str, np.ndarray]:
        """Descrição da tabela que o modelo consome (todas as chaves têm C na 1ª dimensão).

        kind      (C,)        NUM/CAT
        n_options (C,)        opções válidas por coluna (sem contar [NULL])
        name_ids  (C, L)      ids de hash do nome da coluna
        cand_ids  (C, K, L)   ids de hash do texto de cada categoria (CAT)
        opt_q     (C, K)      posição no CDF de cada bin (NUM)
        opt_z     (C, K)      mediana padronizada de cada bin (NUM)
        """
        if self._meta is None:
            C, K = len(self.columns), max(c.n_options for c in self.columns)
            cand_ids = np.zeros((C, K, MAX_PIECES), dtype=np.int64)
            opt_q = np.zeros((C, K), dtype=np.float32)
            opt_z = np.zeros((C, K), dtype=np.float32)
            for j, spec in enumerate(self.columns):
                n = spec.n_options
                if spec.kind == CAT:
                    cand_ids[j, :n] = encode_texts(spec.categories)
                else:
                    opt_q[j, :n] = spec.bin_q
                    opt_z[j, :n] = spec.standardize(spec.bin_values)
            self._meta = {
                "kind": np.array([c.kind for c in self.columns], dtype=np.int64),
                "n_options": np.array([c.n_options for c in self.columns], dtype=np.int64),
                "name_ids": encode_texts(self.column_names),
                "cand_ids": cand_ids, "opt_q": opt_q, "opt_z": opt_z,
            }
        return self._meta

    # --------------------------------------------------------------- encode
    def encode(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """DataFrame -> {"local": (N, C) int64, "z": (N, C) float32}. Coluna ausente vira nula."""
        n = len(df)
        local = np.zeros((n, len(self.columns)), dtype=np.int64)
        z = np.zeros((n, len(self.columns)), dtype=np.float32)
        for j, spec in enumerate(self.columns):
            s = df[spec.name] if spec.name in df.columns else pd.Series(np.nan, index=df.index)
            if spec.kind == CAT:
                index = {c: i + 1 for i, c in enumerate(spec.categories)}
                # categoria nunca vista: OTHER se existir, senão [NULL]
                fallback = index.get(OTHER, LOCAL_NULL)
                codes = np.array([index.get(k, fallback) for k in s.map(_key)], dtype=np.int64)
                local[:, j] = np.where(s.isna().to_numpy(), LOCAL_NULL, codes)
            else:
                x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
                null = np.isnan(x)
                local[:, j] = np.where(null, LOCAL_NULL, np.searchsorted(spec.edges, np.nan_to_num(x), side="right") + 1)
                z[:, j] = np.where(null, 0.0, spec.standardize(np.nan_to_num(x)))
        return {"local": local, "z": z}

    # --------------------------------------------------------------- decode
    def decode(self, local, z=None) -> pd.DataFrame:
        """Índices locais (N, C) -> DataFrame. Numéricas viram a mediana do bin,
        ou o valor contínuo se z for dado; [NULL] vira NaN."""
        local = np.asarray(local)
        data = {}
        for j, spec in enumerate(self.columns):
            col = local[:, j]
            ok = col != LOCAL_NULL
            if spec.kind == CAT:
                data[spec.name] = [spec.categories[i - 1] if i != LOCAL_NULL else np.nan for i in col]
            else:
                out = np.full(len(col), np.nan)
                out[ok] = np.asarray(spec.bin_values)[col[ok] - 1]
                if z is not None:
                    out[ok] = np.asarray(z)[ok, j] * spec.std + spec.mean
                data[spec.name] = out
        return pd.DataFrame(data)

    # --------------------------------------------------------- serialização
    def to_dict(self):
        params = {k: getattr(self, k) for k in ("n_bins", "max_cat_levels", "max_categories", "min_frequency")}
        return {"params": params, "columns": [asdict(c) for c in self.columns]}

    @classmethod
    def from_dict(cls, d):
        enc = cls(**d["params"])
        enc.columns = [ColumnSpec(**c) for c in d["columns"]]
        return enc

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
