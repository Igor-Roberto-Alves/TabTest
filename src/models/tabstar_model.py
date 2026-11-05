"""TabSTAR: modelo fundacional tabular que verbaliza as células como texto.

Pesos em `alana89/TabSTAR` no Hugging Face, baixados automaticamente no fit().
Diferente do TabPFN, o fit faz fine-tuning LoRA (até `max_epochs` épocas, early stopping),
então GPU é fortemente recomendada. Recebe o DataFrame bruto: nomes de colunas
e rótulos de classe são usados semanticamente, por isso passamos os originais.

Sem limite de linhas por padrão (`max_train=None`): o custo cresce linearmente com N
por época. A validação do early stopping é separada aqui (estratificada, `val_size`
linhas) em vez da interna do TabSTAR, que usa no máximo 1000 linhas e fica quase sem
positivos em classes raras.

Curvas no TensorBoard: curves/train_loss, curves/val_loss e curves/val_metric por época.
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import Split, subsample
from src.models.base import Prediction

KINDS = {"binary", "multiclass", "regression"}


@contextmanager
def _log_epochs(writer):
    """Intercepta os métodos de época do treinador do TabSTAR para mandar ao TensorBoard."""
    if writer is None:
        yield
        return
    from tabstar.training.trainer import TabStarTrainer

    train_epoch, evaluate_epoch = TabStarTrainer._train_epoch, TabStarTrainer._evaluate_epoch
    state = {"epoch": 0}

    def patched_train(self, loader):
        loss = train_epoch(self, loader)
        state["epoch"] += 1
        writer.add_scalar("curves/train_loss", loss, state["epoch"])
        return loss

    def patched_eval(self, loader):
        val_loss, val_metric = evaluate_epoch(self, loader)
        writer.add_scalar("curves/val_loss", val_loss, state["epoch"])
        writer.add_scalar("curves/val_metric", val_metric, state["epoch"])
        return val_loss, val_metric

    TabStarTrainer._train_epoch, TabStarTrainer._evaluate_epoch = patched_train, patched_eval
    try:
        yield
    finally:
        TabStarTrainer._train_epoch, TabStarTrainer._evaluate_epoch = train_epoch, evaluate_epoch


def fit_predict(split: Split, seed=0, max_train: int | None = None, time_limit: int | None = None,
                device: str | None = None, val_size: int | float = 0.1, max_val: int = 20_000,
                max_epochs: int = 50, patience: int = 5, lora_batch: int = 64, global_batch: int = 128,
                lora_lr: float | None = None, writer=None) -> Prediction:
    import torch
    from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

    # Passa um torch.device explícito: a detecção do TabSTAR recusa qualquer GPU que tenha
    # algum processo (ex.: o desktop/VS Code), mesmo com memória sobrando.
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    X_train, y_train = subsample(split.X_train, split.y_train, max_train, seed)
    # o TabSTAR não reconhece o dtype `str` do pandas 3: texto vai como object
    as_object = {c: object for c in split.cat_cols}
    X_train = X_train.reset_index(drop=True).astype(as_object)
    X_test = split.X_test.reset_index(drop=True).astype(as_object)

    n_val = min(int(val_size * len(X_train)) if isinstance(val_size, float) else val_size, max_val)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=max(n_val, 1), random_state=seed,
        stratify=None if split.is_regression else y_train)
    X_fit, X_val = X_fit.reset_index(drop=True), X_val.reset_index(drop=True)
    print(f"    tabstar: fit={len(X_fit)} val={len(X_val)} device={device}")

    params = dict(random_state=seed, time_limit=time_limit, device=device, max_epochs=max_epochs,
                  patience=patience, lora_batch=lora_batch, global_batch=global_batch)
    # O OneCycleLR do TabSTAR tem total_steps=max_epochs e aquece em 10% das épocas (pico de
    # 1e-3 na época 5 com max_epochs=50). Com ~850k linhas são milhares de updates por época
    # e o treino diverge perto do pico. Em dataset grande: menos épocas ou lora_lr menor.
    if lora_lr is not None:
        params["lora_lr"] = lora_lr
    target = split.task.target
    if split.is_regression:
        model = TabSTARRegressor(**params)
        with _log_epochs(writer):
            model.fit(X_fit, pd.Series(y_fit, name=target), X_val, pd.Series(y_val, name=target))
        return Prediction(np.asarray(model.predict(X_test)).ravel(), model=model)

    model = TabSTARClassifier(**params)
    with _log_epochs(writer):
        model.fit(X_fit, pd.Series(split.classes[y_fit], name=target),
                  X_val, pd.Series(split.classes[y_val], name=target))
    labels = np.asarray(model.predict(X_test)).astype(str)
    y_pred = np.searchsorted(split.classes, labels)  # rótulo original -> inteiro

    # assume colunas na ordem ordenada dos rótulos (mesma do LabelEncoder);
    # se o formato não bater, avalia só as predições duras
    proba = np.asarray(model.predict_proba(X_test))
    if proba.ndim == 1 and len(split.classes) == 2:
        proba = np.column_stack([1 - proba, proba])
    if proba.ndim != 2 or proba.shape[1] != len(split.classes):
        proba = None
    return Prediction(y_pred, proba, model)
