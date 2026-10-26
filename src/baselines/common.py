"""Peças comuns dos baselines: rótulos do store, subamostra de rótulos e treino com registro.

As linhas de rótulo, o split e a subamostra da curva de rótulos são os mesmos do fine-tuning
(src/pragma/finetune.py), para a comparação ser linha a linha. Os rótulos são lidos direto dos
.npy do store, sem abrir as fontes de eventos.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "banking"
PREPARED = RAW / "prepared"
FEATURES = RAW / "features"
SPLITS = {"train": 0, "val": 1, "test": 2}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_f32(df):
    """float64 -> float32 e inteiros -> float32: as features não precisam de mais precisão."""
    for c in df.columns:
        if df[c].dtype != np.float32 and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
    return df


def clean_names(cols):
    """O XGBoost recusa '[', ']' e '<' nos nomes."""
    return [re.sub(r"[\[\]<>,]", "_", str(c)) for c in cols]


# -------------------------------------------------------------------- rótulos
class Labels:
    """Linhas de rótulo do store: id original do cliente, corte, y (R, k) e split de cada linha."""

    def __init__(self, base):
        d = PREPARED / base
        with open(d / "meta.json") as f:
            self.task = json.load(f)["task"]
        clients = np.load(d / "clients.npy")
        split = np.load(d / "split.npy")
        cidx = np.load(d / "labels" / "client.npy")
        self.client = clients[cidx]
        self.cutoff = np.load(d / "labels" / "cutoff.npy")
        self.y = np.load(d / "labels" / "y.npy")
        self.split = split[cidx]

    def rows(self, split):
        return np.nonzero(self.split == SPLITS[split])[0]


def subsample(rows, n, seed):
    """Igual a src/pragma/finetune.subsample: mesma semente, mesmas linhas de treino."""
    if n is None or len(rows) <= n:
        return rows
    return np.sort(np.random.default_rng(seed).choice(rows, n, replace=False))


# ---------------------------------------------------------------------- treino
class Recorder:
    """Grava métricas no status.json (RunStatus) e numa linha de metrics.jsonl por registro."""

    def __init__(self, run_dir, run):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.path.write_text("")
        self.run = run

    def __call__(self, step, metrics):
        metrics = {k: float(v) for k, v in metrics.items()}
        self.run.update(step, metrics)
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": int(step), **metrics}) + "\n")


class LogCallback(xgb.callback.TrainingCallback):
    """A cada `every` rodadas manda train/logloss, val/logloss, val/auc (e val/<extra>) ao Recorder.
    `offset` torna o passo contínuo entre combinações da grade e rótulos."""

    def __init__(self, record, every, offset, extra=None):
        self.record, self.every, self.offset, self.extra = record, every, offset, extra

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.every == 0:
            m = {"train/logloss": evals_log["train"]["logloss"][-1], "val/logloss": evals_log["val"]["logloss"][-1],
                 "val/auc": evals_log["val"]["auc"][-1]}
            if self.extra is not None:
                m.update(self.extra(model, epoch))
            self.record(self.offset + epoch + 1, m)
        return False


def fit(params, Xtr, ytr, Xva, yva, cfg, record, offset, names, extra=None):
    """Um XGBoost com early stopping na AUC de validação. Retorna o booster (best_iteration definido)."""
    dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=params.get("max_bin", 256), feature_names=names)
    dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr, feature_names=names)
    cb_extra = None if extra is None else (lambda model, epoch: extra(model, dva, epoch))
    callbacks = [xgb.callback.EarlyStopping(rounds=cfg["early_stopping"], metric_name="auc", data_name="val",
                                            maximize=True, save_best=False),
                 LogCallback(record, cfg["log_every"], offset, cb_extra)]
    booster = xgb.train(params, dtr, num_boost_round=cfg["max_rounds"], evals=[(dtr, "train"), (dva, "val")],
                        callbacks=callbacks, verbose_eval=False)
    return booster


def predict(booster, X, names):
    return booster.predict(xgb.DMatrix(X, feature_names=names), iteration_range=(0, booster.best_iteration + 1))


def importance(boosters, names, top=50):
    """Ganho médio (normalizado por modelo) de cada feature, nas top `top`."""
    total = pd.Series(0.0, index=names)
    for b in boosters:
        g = pd.Series(b.get_score(importance_type="gain"), dtype=float).reindex(names).fillna(0.0)
        if g.sum() > 0:
            total += g / g.sum()
    total /= max(len(boosters), 1)
    return total.sort_values(ascending=False).head(top).rename("gain").rename_axis("feature").reset_index()
