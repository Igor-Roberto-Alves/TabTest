"""Carrega os arquivos de uma Task (protocolo em src/tasks.py) e monta o split treino/teste.

Ordem: lê arquivos -> filter -> cria alvo (expr) -> remove alvo nulo -> max_rows ->
split -> monta X (drop/features, categorical, datetime).
"""

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

from src.tasks import ROOT, Task

DATA_DIR = ROOT / "data"


@dataclass
class Split:
    task: Task
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray  # classificação: inteiros 0..k-1; regressão: float
    y_test: np.ndarray
    classes: np.ndarray | None  # rótulos originais, na ordem dos inteiros

    @property
    def is_regression(self):
        return self.task.kind == "regression"

    @property
    def cat_cols(self):
        return [c for c in self.X_train.columns if not pd.api.types.is_numeric_dtype(self.X_train[c])]

    @property
    def num_cols(self):
        return [c for c in self.X_train.columns if pd.api.types.is_numeric_dtype(self.X_train[c])]


def resolve_files(task: Task) -> list[Path]:
    paths = []
    for pattern in task.files:
        full = pattern if Path(pattern).is_absolute() else str(ROOT / pattern)
        matches = sorted(glob.glob(full))
        if not matches:
            hint = f"; rode: python download_datasets.py {task.dataset}" if pattern.startswith("data/") else ""
            raise FileNotFoundError(f"{task.source}: nenhum arquivo para '{pattern}'{hint}")
        paths += [Path(m) for m in matches]
    return paths


def _read(path: Path, read_kwargs: dict) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, **{"low_memory": False, **read_kwargs})


def load_frame(task: Task) -> pd.DataFrame:
    """DataFrame completo da tarefa (após filter e com a coluna alvo criada)."""
    df = pd.concat([_read(p, task.read) for p in resolve_files(task)], ignore_index=True)
    if task.filter:
        df = df.query(task.filter, engine="python").reset_index(drop=True)
    if task.expr:
        df[task.target] = df.eval(task.expr, engine="python")
    if task.target not in df.columns:
        raise KeyError(f"{task.name}: coluna alvo '{task.target}' não existe (defina 'expr' para derivá-la)")
    return df


def _split_index(task: Task, df: pd.DataFrame, y, seed: int):
    s = task.split
    idx = np.arange(len(df))
    if s.method == "temporal":
        order = pd.to_datetime(df[s.column]) if not pd.api.types.is_numeric_dtype(df[s.column]) else df[s.column]
        if s.cutoff is not None:
            cutoff = pd.Timestamp(s.cutoff) if not pd.api.types.is_numeric_dtype(order) else float(s.cutoff)
            test = (order >= cutoff).to_numpy()
            return idx[~test], idx[test]
        ranked = idx[np.argsort(order.to_numpy(), kind="stable")]
        n_test = int(round(len(df) * s.test_size))
        return ranked[:-n_test], ranked[-n_test:]
    if s.method == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=s.test_size, random_state=seed)
        return next(gss.split(idx, groups=df[s.column].to_numpy()))
    stratify = None
    if task.kind != "regression" and np.bincount(y).min() >= 2:
        stratify = y
    return train_test_split(idx, test_size=s.test_size, random_state=seed, stratify=stratify)


def _expand_datetime(X: pd.DataFrame, cols) -> pd.DataFrame:
    for col in cols:
        if col not in X.columns:
            continue
        t = pd.to_datetime(X[col], errors="coerce")
        parts = {"year": t.dt.year, "month": t.dt.month, "day": t.dt.day, "dow": t.dt.dayofweek, "hour": t.dt.hour}
        for name, values in parts.items():
            if values.nunique() > 1:  # parte constante não informa nada
                X[f"{col}_{name}"] = values.astype("float")
        X = X.drop(columns=col)
    return X


def build_features(task: Task, df: pd.DataFrame) -> pd.DataFrame:
    cols = list(task.features) if task.features is not None else list(df.columns)
    drop = set(task.drop) | {task.target}
    X = df[[c for c in cols if c not in drop]].copy()
    for col in task.categorical:
        if col in X.columns:
            X[col] = X[col].astype("string").astype(object).where(X[col].notna())
    return _expand_datetime(X, task.datetime)


def encode_target(task: Task, y: pd.Series):
    if task.kind == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise TypeError(f"{task.name}: kind=regression exige alvo numérico, veio {y.dtype}")
        return y.to_numpy(dtype=float), None
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.astype(str))
    n = len(encoder.classes_)
    if task.kind == "binary" and n != 2:
        raise ValueError(f"{task.name}: kind=binary mas o alvo tem {n} classes: {list(encoder.classes_)[:10]}")
    if task.kind == "multiclass" and n < 3:
        raise ValueError(f"{task.name}: kind=multiclass mas o alvo tem {n} classes (use binary)")
    return encoded, encoder.classes_


def prepare(task: Task, seed=0) -> pd.DataFrame:
    """Frame após filter, alvo derivado, remoção de alvo nulo e max_rows."""
    df = load_frame(task)
    df = df[df[task.target].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{task.name}: nenhuma linha com alvo não nulo após 'filter'; revise filter/expr")
    if task.max_rows and len(df) > task.max_rows:
        df = df.sample(task.max_rows, random_state=seed).reset_index(drop=True)
    return df


def load_task(task: Task, seed=0) -> Split:
    df = prepare(task, seed)
    y, classes = encode_target(task, df[task.target])
    train_idx, test_idx = _split_index(task, df, y, seed)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"{task.name}: split vazio (treino={len(train_idx)}, teste={len(test_idx)}); revise 'split'")
    X = build_features(task, df)
    return Split(task, X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx], classes)


def describe(task: Task, seed=0) -> str:
    """Resumo da tarefa e alertas de protocolo (ids, constantes, alvo desbalanceado)."""
    split = load_task(task, seed)
    X = pd.concat([split.X_train, split.X_test])
    lines = [f"{task.name} ({task.kind}) — {task.source}",
             f"  arquivos: {', '.join(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in resolve_files(task))}",
             f"  linhas: treino={len(split.X_train)} teste={len(split.X_test)}  split={task.split.method}",
             f"  features: {X.shape[1]} ({len(split.num_cols)} num, {len(split.cat_cols)} cat)"]
    if task.kind == "regression":
        y = pd.Series(np.concatenate([split.y_train, split.y_test]))
        lines.append(f"  alvo '{task.target}': média={y.mean():.4g} dp={y.std():.4g} min={y.min():.4g} max={y.max():.4g}")
    else:
        for name, part in (("treino", split.y_train), ("teste", split.y_test)):
            dist = pd.Series(split.classes[part]).value_counts(normalize=True).round(4).to_dict()
            lines.append(f"  alvo '{task.target}' {name}: {dist}")

    alerts = []
    for col in X.columns:
        n_unique = X[col].nunique()
        if n_unique <= 1:
            alerts.append(f"'{col}' é constante")
        elif col in split.cat_cols and n_unique > 0.5 * len(X):
            alerts.append(f"'{col}' parece identificador ({n_unique} valores únicos em {len(X)}); considere drop")
        if X[col].isna().mean() > 0.5:
            alerts.append(f"'{col}' tem {X[col].isna().mean():.0%} de nulos")
    if task.kind != "regression" and np.bincount(split.y_train).min() / len(split.y_train) < 0.01:
        alerts.append("classe minoritária < 1% no treino")
    lines += [f"  ALERTA: {a}" for a in alerts] or ["  sem alertas"]
    lines.append("  colunas: " + ", ".join(X.columns))
    return "\n".join(lines)


def subsample(X, y, n: int | None, seed=0):
    """Limita o treino a n linhas (TabPFN e TabSTAR escalam mal com N)."""
    if n is None or len(X) <= n:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    X = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
    return X, y[idx]
