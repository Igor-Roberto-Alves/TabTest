"""Carrega os CSVs de data/ e monta os splits de treino e teste de cada Task."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.tasks import Task

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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


def load_frame(task: Task, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"{task.dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe; rode: python download_datasets.py {task.dataset}")
    df = pd.read_csv(path)
    if task.derive:
        df = task.derive(df)
    return df.drop(columns=list(task.drop), errors="ignore")


def load_task(task: Task, data_dir: Path = DATA_DIR, test_size=0.2, seed=0) -> Split:
    df = load_frame(task, data_dir).dropna(subset=[task.target])
    if task.max_rows and len(df) > task.max_rows:
        df = df.sample(task.max_rows, random_state=seed)

    X = df.drop(columns=[task.target])
    y = df[task.target]
    classes = None
    if task.kind == "regression":
        y = y.to_numpy(dtype=float)
    else:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y.astype(str))
        classes = encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=None if task.kind == "regression" else y,
    )
    return Split(task, X_train, X_test, y_train, y_test, classes)


def subsample(X, y, n: int | None, seed=0):
    """Limita o treino a n linhas (TabPFN e TabSTAR escalam mal com N)."""
    if n is None or len(X) <= n:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    X = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
    return X, y[idx]
