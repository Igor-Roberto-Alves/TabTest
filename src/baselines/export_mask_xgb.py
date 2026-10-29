"""Exporta para o painel os mesmos clientes do AmEx lidos por um XGBoost (colunas cruas do último extrato).

Treina o XGBoost como em src/baselines/xgb.py --raw (primeira combinação da grade do AmEx, early stopping na AUC de
validação) e, para os clientes de runs/explain/amex_mask_examples.json, calcula p(default) escondendo cada coluna,
cada grupo e tudo. "Esconder" é tornar a coluna desconhecida (média pelos ramos, ver xgb_unknown.py), a mesma definição
do SHAP; o NULL (col_p_null) fica só para comparação, porque o XGBoost aprendeu o que fazer com NULL. O modelo só lê o
último extrato: esconder células dos extratos anteriores não muda p. Também salva a validação em
runs/explain/amex_val_raw.parquet para o explain_shap_xgb.py.
Saída: runs/explain/amex_mask_examples_xgboost.json, no mesmo formato do PRAGMA.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.pragma.config import load_config
from src.pragma.evaluate import amex_metric

from .common import ROOT, Labels, clean_names, log, subsample
from .xgb import align, load_features, raw_columns
from .xgb_unknown import Trees

THR = 0.5
OUT = ROOT / "runs/explain"


def main():
    pragma = json.loads((OUT / "amex_mask_examples.json").read_text())
    cfg = load_config(ROOT / "configs/baselines/xgb.yaml", [])
    labels = Labels("amex")
    index, names, A, _ = load_features("amex")
    keep = raw_columns("amex", names)
    A, names = A[:, keep], clean_names([names[j] for j in keep])
    pos = align(index, labels)
    ok = pos >= 0
    rows = {s: labels.rows(s)[ok[labels.rows(s)]] for s in ("train", "val")}
    y = labels.y[:, 0]
    params = {**cfg["params"], **cfg["grid"]["amex"][0], "seed": cfg["seed"]}
    dtr = xgb.DMatrix(A[pos[rows["train"]]], y[rows["train"]], feature_names=names)
    dva = xgb.DMatrix(A[pos[rows["val"]]], y[rows["val"]], feature_names=names)
    log(f"treino {len(rows['train'])} linhas, {len(names)} colunas")
    booster = xgb.train(params, dtr, cfg["max_rounds"], evals=[(dva, "val")], early_stopping_rounds=cfg["early_stopping"],
                        verbose_eval=200)
    pva = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    val_amex = amex_metric(y[rows["val"]], pva)
    log(f"val amex {val_amex:.4f}, melhor rodada {booster.best_iteration}")
    booster = booster[: booster.best_iteration + 1]  # só as árvores até a melhor rodada: é o modelo que gerou pva

    # coluna do PRAGMA ("payment 2", orig P_2) -> coluna crua do XGBoost (P_2_last)
    by_orig = {n[:-len("_last")]: j for j, n in enumerate(names)}
    cols = pragma["cols"]
    cmap = [by_orig[c["orig"]] for c in cols]
    groups = sorted({c["group"] for c in cols})
    clients_store = np.load(ROOT / "data/banking/prepared/amex/clients.npy")
    label_client = np.load(ROOT / "data/banking/prepared/amex/labels/client.npy")

    # a validação em parquet, com o id original do cliente, para o explain_shap_xgb.py
    vr = rows["val"]
    val = pd.DataFrame(A[pos[vr]], columns=names)
    val.insert(0, "customer_ID", clients_store[label_client[vr]].astype(str))
    val.to_parquet(OUT / "amex_val_raw.parquet")
    booster.save_model(str(OUT / "xgb_amex_raw.json"))
    trees = Trees(booster)

    clients = []
    for c in pragma["clients"]:
        r = np.nonzero(label_client == c["client"])[0]
        r = r[0] if len(r) else None
        assert r is not None and pos[r] >= 0, f"cliente {c['client']} sem linha de rótulo/features"
        x = A[pos[r]].copy()
        n_cols, E = len(cols), len(c["t"])
        unk = lambda idx: np.isin(np.arange(len(names)), idx)
        p_of = lambda idx: float(trees.proba(x, unk(idx)))
        p0 = p_of([])
        col_p = np.array([p_of([j]) for j in cmap])
        grp_p = np.array([p_of([cmap[q] for q, col in enumerate(cols) if col["group"] == g]) for g in groups])
        p_all = p_of(cmap)
        pn = np.tile(x, (n_cols, 1)); pn[np.arange(n_cols), cmap] = np.nan  # NULL, só para comparação
        col_p_null = booster.predict(xgb.DMatrix(pn, feature_names=names))
        assert abs(p0 - float(booster.predict(xgb.DMatrix(x[None], feature_names=names))[0])) < 1e-4
        r3 = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()
        cell_p = [[round(p0, 3)] * n_cols for _ in range(E - 1)] + [r3(col_p)]  # extratos antigos não entram no modelo
        event_p = [round(p0, 3)] * (E - 1) + [round(p_all, 3)]
        denied = p0 >= THR
        kind = ("negado" if denied else "aprovado") + (" que deu default" if c["y"] else " que não deu default")
        clients.append({"client": c["client"], "kind": kind, "y": c["y"], "p": round(p0, 4), "t": c["t"], "v": c["v"],
                        "cell_p": cell_p, "col_p": r3(col_p), "col_p_null": r3(col_p_null), "event_p": event_p,
                        "groups": [{"group": g, "p_all": round(float(q), 3), "p_last": round(float(q), 3)} for g, q in zip(groups, grp_p)],
                        "last_k": [round(p_all, 3)] * (E - 1), "all_events": round(p_all, 3), "all_cells": round(p_all, 3)})
        log(f"{c['client']} {kind}: p {p0:.3f} (PRAGMA {c['p']:.3f}), tudo desconhecido {p_all:.3f}")
    out = {"run": f"runs/baselines/xgb_amex_raw (val amex {val_amex:.3f}); esconder = coluna desconhecida", "threshold": THR, "cols": cols, "clients": clients}
    (OUT / "amex_mask_examples_xgboost.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
