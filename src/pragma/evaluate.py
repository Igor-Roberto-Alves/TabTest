"""Embeddings de clientes, linear probe e métricas das tarefas.

Métricas binárias: ROC-AUC, PR-AUC, KS, Brier, log-loss e, na AmEx, a métrica oficial
(média do Gini normalizado e da taxa de default capturada nos 4% de maior risco, com os
negativos pesando 20 para compensar a subamostragem). Em multirrótulo (MBD), cada rótulo é
avaliado à parte e "auc" é a média entre os rótulos com as duas classes presentes.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import make_loader, to_device


@torch.no_grad()
def embed(model, metas, dataset, batch_size, device, num_workers=4, amp=True):
    """Embedding do cliente ([CLS]) de cada amostra do dataset -> (emb (M, d), y (M, k) ou None)."""
    model.eval()
    embs, ys = [], []
    for batch in make_loader(dataset, batch_size, shuffle=False, num_workers=num_workers):
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.startswith("cuda")):
            embs.append(model(batch, metas)["client"].float().cpu())
        if "y" in batch:
            ys.append(batch["y"].cpu())
    return torch.cat(embs).numpy(), (torch.cat(ys).numpy() if ys else None)


def amex_metric(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    w = np.where(y == 0, 20.0, 1.0)

    def weighted_gini(order):
        ys, ws = y[order], w[order]
        random = np.cumsum(ws / ws.sum())
        lorentz = np.cumsum(ys * ws) / (ys * ws).sum()
        return ((lorentz - random) * ws).sum()

    by_pred = np.argsort(-p, kind="stable")
    top = np.cumsum(w[by_pred]) <= 0.04 * w.sum()
    top4 = y[by_pred][top].sum() / y.sum()
    gini = weighted_gini(by_pred) / weighted_gini(np.argsort(-y, kind="stable"))
    return 0.5 * (gini + top4)


def binary_metrics(y, p, amex=False):
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    if len(np.unique(y)) < 2:
        return {}
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    grid = np.unique(p)
    ks = np.max(np.abs(np.searchsorted(pos, grid, side="right") / len(pos)
                       - np.searchsorted(neg, grid, side="right") / len(neg)))
    out = {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "ks": ks,
           "brier": brier_score_loss(y, p), "logloss": log_loss(y, p, labels=[0, 1])}
    if amex:
        out["amex"] = amex_metric(y, p)
    return out


def task_metrics(task, Y, P):
    """Y, P (M, k). Retorna métricas planas: {"auc", ...} ou, em multirrótulo, também {"<rótulo>/auc", ...}."""
    Y, P = np.asarray(Y).reshape(len(Y), -1), np.asarray(P).reshape(len(P), -1)
    amex = task.get("metric") == "amex"
    if Y.shape[1] == 1:
        return binary_metrics(Y[:, 0], P[:, 0], amex)
    out, per = {}, []
    for j, name in enumerate(task["labels"]):
        m = binary_metrics(Y[:, j], P[:, j])
        per.append(m)
        out.update({f"{name}/{k}": v for k, v in m.items()})
    for k in ("auc", "pr_auc", "ks", "brier", "logloss"):
        vals = [m[k] for m in per if k in m]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def primary_metric(task):
    return "amex" if task.get("metric") == "amex" else "auc"


def linear_probe(train_emb, train_y, eval_emb, eval_y, task, C=1.0, max_iter=2000):
    """Logística (uma por rótulo) sobre embeddings padronizados. Retorna (métricas, probabilidades)."""
    scaler = StandardScaler().fit(train_emb)
    Xtr, Xev = scaler.transform(train_emb), scaler.transform(eval_emb)
    train_y = train_y.reshape(len(train_y), -1)
    P = np.zeros((len(eval_emb), train_y.shape[1]))
    for j in range(train_y.shape[1]):
        if len(np.unique(train_y[:, j])) < 2:
            P[:, j] = train_y[:, j].mean()
            continue
        clf = LogisticRegression(C=C, max_iter=max_iter).fit(Xtr, train_y[:, j])
        P[:, j] = clf.predict_proba(Xev)[:, 1]
    return task_metrics(task, eval_y, P), P
