"""TreeSHAP para o XGBoost do AmEx.

Calcula valores SHAP do modelo de árvores (runs/baselines/xgb_amex_raw) para:
  * uma amostra da validação, para a importância global de cada coluna e de cada grupo (D, S, P, B, R);
  * os clientes do estudo de máscaras, para a explicação individual (por que p deu o que deu).

Dois modos (veja SHAP_explicado.md):
  * path-dependent (padrão): usa as proporções de treino guardadas nas próprias árvores. Exato, rápido,
    não precisa de dados extras. É o `pred_contribs=True` do próprio XGBoost.
  * interventional (--interventional): substitui a coluna "desconhecida" pelos valores de um conjunto de
    clientes de referência (fundo). Usa a biblioteca `shap`. Mais lento; só nos clientes do estudo.

Os valores saem em log-odds (a escala antes da sigmoide), onde a soma é exata:
    base + soma(SHAP) = log-odds do modelo,  p = sigmoide(log-odds).

Saídas:
  * --out-json: tudo, para análise em Python/notebook;
  * --out-js:   o mesmo conteúdo como `window.SHAPX = {...}`, no formato dos mask_data_*.js do dashboard.

Exemplo:
    python explain_shap_xgb.py \
        --model runs/baselines/xgb_amex_raw/model.json \
        --data data/amex/val_last.parquet \
        --clients 254131 316386 62456 132266 167843 150691 \
        --interactions 10 --interventional \
        --out-json runs/explain/amex_shap_xgb.json --out-js runs/explain/mask_data_shap.js
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

GROUPS = {"D": "delinquency", "S": "spend", "P": "payment", "B": "balance", "R": "risk"}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def col_info(orig: str) -> dict:
    """D_59 -> {'orig': 'D_59', 'name': 'delinquency 59', 'group': 'delinquency'} (mesmo formato do dashboard)."""
    orig = orig.removesuffix("_last")  # colunas cruas do AmEx: P_2_last -> P_2
    pre, _, num = orig.partition("_")
    group = GROUPS.get(pre, "other")
    return {"orig": orig, "name": f"{group} {num}" if num else orig, "group": group}


def load_features(path: Path, id_col: str, date_col: str | None) -> pd.DataFrame:
    """Lê CSV ou Parquet. Se houver vários extratos por cliente, fica com o último (como o XGBoost foi treinado)."""
    df = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    if id_col in df.columns and df[id_col].duplicated().any():
        if date_col and date_col in df.columns:
            df = df.sort_values([id_col, date_col])
        df = df.groupby(id_col, sort=False).tail(1)
        print(f"[dados] vários extratos por cliente: fiquei com o último ({len(df)} clientes)", file=sys.stderr)
    return df.reset_index(drop=True)


def align(df: pd.DataFrame, booster: xgb.Booster) -> pd.DataFrame:
    """Põe as colunas na ordem do modelo e converte as categóricas como o modelo espera."""
    names = booster.feature_names
    if names is None:
        raise SystemExit("O modelo não guarda os nomes das colunas; salve-o a partir de um DataFrame ou passe os nomes.")
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise SystemExit(f"Faltam {len(missing)} colunas do modelo nos dados, por exemplo: {missing[:5]}")
    X = df[names].copy()
    types = booster.feature_types or []
    for c, t in zip(names, types):
        if t == "c":
            X[c] = X[c].astype("category")
        elif X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def dmatrix(X: pd.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(X, enable_categorical=True)


def tree_shap(booster: xgb.Booster, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """TreeSHAP path-dependent. Devolve (valores [n, m], base [n]), em log-odds."""
    contrib = booster.predict(dmatrix(X), pred_contribs=True)
    return contrib[:, :-1], contrib[:, -1]


def tree_interactions(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Valores de interação SHAP [n, m+1, m+1]; a diagonal é o efeito principal, fora dela a interação de cada par."""
    return booster.predict(dmatrix(X), pred_interactions=True)


def interventional_shap(booster, X, background, seed=0):
    """SHAP interventional com um fundo de clientes. Precisa da biblioteca `shap`.

    Sem colunas categóricas, usa o TreeExplainer (exato). Com categóricas (por exemplo D_63, D_64 no AmEx),
    o TreeExplainer da `shap` não lê as divisões categóricas do XGBoost; aí usa o PermutationExplainer,
    que estima o mesmo valor trocando colunas pelos valores do fundo em ordens sorteadas.
    Devolve (valores, base, nome do método)."""
    import shap

    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    if not cat:
        expl = shap.TreeExplainer(booster, data=background, feature_perturbation="interventional", model_output="raw")
        vals = np.asarray(expl.shap_values(X, check_additivity=False))
        base = np.full(len(X), float(np.ravel(expl.expected_value)[0]))
        return vals, base, "TreeSHAP interventional (shap.TreeExplainer)"

    # categorias viram códigos numéricos só para o masker; a previsão volta a montar as categorias
    cats = {c: X[c].cat.categories for c in cat}

    def encode(df):
        out = df.copy()
        for c in cat:
            out[c] = pd.Categorical(out[c], categories=cats[c]).codes.astype(float)
            out.loc[out[c] < 0, c] = np.nan
        return out.to_numpy(dtype=float)

    def predict(arr):
        df = pd.DataFrame(arr, columns=X.columns)
        for c in cat:
            codes = df[c].to_numpy()
            df[c] = pd.Categorical.from_codes(np.where(np.isnan(codes), -1, codes).astype(int), categories=cats[c])
        return booster.predict(dmatrix(df), output_margin=True)

    masker = shap.maskers.Independent(encode(background), max_samples=len(background))
    expl = shap.PermutationExplainer(predict, masker, seed=seed)
    res = expl(encode(X), max_evals=2 * X.shape[1] + 1 + 20 * X.shape[1])
    return np.asarray(res.values), np.asarray(res.base_values, dtype=float), \
        "SHAP interventional por permutação (shap.PermutationExplainer; há colunas categóricas)"


def group_sums(vals: np.ndarray, cols: list[dict]) -> dict[str, np.ndarray]:
    """Soma o SHAP de cada grupo de colunas. Vale pela aditividade: o grupo leva a soma das suas colunas."""
    out = {}
    for g in dict.fromkeys(c["group"] for c in cols):
        idx = [j for j, c in enumerate(cols) if c["group"] == g]
        out[g] = vals[:, idx].sum(axis=1)
    return out


def r(x, nd=4):
    """Arredonda para o JSON; NaN vira None."""
    if x is None:
        return None
    if isinstance(x, (np.floating, float)) and not np.isfinite(x):
        return None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), nd)
    return x if isinstance(x, (str, bool)) else str(x)


def waterfall(base: float, vals: np.ndarray, X_row: pd.Series, cols: list[dict], top: int) -> list[dict]:
    """Passo a passo da previsão: parte da base e soma as colunas da maior |SHAP| para a menor; o resto vira 'outras'."""
    order = np.argsort(-np.abs(vals))
    steps, z = [], base
    for j in order[:top]:
        z += vals[j]
        steps.append({"orig": cols[j]["orig"], "name": cols[j]["name"], "group": cols[j]["group"],
                      "value": r(X_row.iloc[j]), "shap": r(vals[j]), "p_after": r(sigmoid(z))})
    rest = vals[order[top:]].sum()
    z += rest
    steps.append({"orig": "outras", "name": f"outras {len(order) - top} colunas", "group": "other",
                  "value": None, "shap": r(rest), "p_after": r(sigmoid(z))})
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="Booster do XGBoost salvo (.json, .ubj ou .model)")
    ap.add_argument("--data", required=True, type=Path, help="Validação do AmEx (CSV ou Parquet)")
    ap.add_argument("--id-col", default="customer_ID")
    ap.add_argument("--date-col", default="S_2", help="Coluna de data dos extratos, para achar o último")
    ap.add_argument("--clients", nargs="*", default=[], help="Clientes a explicar (posições na validação, como no dashboard)")
    ap.add_argument("--clients-are", choices=["rows", "ids"], default="rows",
                    help="rows: posição da linha na validação; ids: valor da coluna --id-col")
    ap.add_argument("--sample", type=int, default=5000, help="Clientes da validação para a importância global")
    ap.add_argument("--interactions", type=int, default=0, help="Top-k interações entre pares por cliente (0 = não calcula)")
    ap.add_argument("--interventional", action="store_true", help="Também calcula o SHAP interventional nos clientes")
    ap.add_argument("--background", type=int, default=200, help="Tamanho do fundo do modo interventional")
    ap.add_argument("--waterfall-top", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--run", default="runs/baselines/xgb_amex_raw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=Path("runs/explain/amex_shap_xgb.json"))
    ap.add_argument("--out-js", type=Path, default=Path("runs/explain/mask_data_shap.js"))
    a = ap.parse_args()

    t0 = time.time()
    booster = xgb.Booster()
    booster.load_model(str(a.model))
    df = load_features(a.data, a.id_col, a.date_col)
    X = align(df, booster)
    names = list(X.columns)
    cols = [col_info(c) for c in names]
    rng = np.random.default_rng(a.seed)
    print(f"[dados] {len(X)} clientes, {len(names)} colunas", file=sys.stderr)

    # ---- importância global numa amostra da validação ----
    n = min(a.sample, len(X))
    idx = np.sort(rng.choice(len(X), n, replace=False))
    Xs = X.iloc[idx]
    vals, base = tree_shap(booster, Xs)
    margin = booster.predict(dmatrix(Xs), output_margin=True)
    add_err = float(np.max(np.abs(base + vals.sum(1) - margin)))
    print(f"[shap] amostra {n}: erro máximo de aditividade {add_err:.2e} (deve ser ~0)", file=sys.stderr)

    mean_abs = np.abs(vals).mean(0)
    order = np.argsort(-mean_abs)
    global_cols = [{**cols[j], "mean_abs": r(mean_abs[j]), "mean": r(vals[:, j].mean()),
                    "pos_share": r((vals[:, j] > 0).mean(), 3)} for j in order]
    gs = group_sums(vals, cols)
    global_groups = [{"group": g, "n": sum(c["group"] == g for c in cols), "mean_abs": r(np.abs(v).mean()),
                      "mean": r(v.mean())} for g, v in gs.items()]
    global_groups.sort(key=lambda x: -x["mean_abs"])
    # dependência: valor da coluna x SHAP, para as 6 colunas mais importantes (até 500 pontos cada)
    dep_n = min(500, n)
    dependence = []
    for j in order[:6]:
        dependence.append({"orig": names[j], "points": [[r(v), r(s)] for v, s in
                                                       zip(Xs.iloc[:dep_n, j].tolist(), vals[:dep_n, j])]})

    # ---- clientes do estudo ----
    if a.clients_are == "rows":
        rows = [int(c) for c in a.clients]
    else:
        pos = {str(v): i for i, v in enumerate(df[a.id_col].astype(str))}
        rows = [pos[str(c)] for c in a.clients if str(c) in pos]
    bad = [c for c in rows if not 0 <= c < len(X)]
    if bad:
        raise SystemExit(f"Clientes fora da validação: {bad}")

    clients, iv_method = [], None
    if rows:
        Xc = X.iloc[rows]
        cv, cb = tree_shap(booster, Xc)
        cm = booster.predict(dmatrix(Xc), output_margin=True)
        ci = tree_interactions(booster, Xc) if a.interactions else None
        if a.interventional:
            bg = X.iloc[rng.choice(len(X), min(a.background, len(X)), replace=False)]
            iv, ib, iv_method = interventional_shap(booster, Xc, bg, a.seed)
            iv_err = float(np.max(np.abs(ib + iv.sum(1) - cm)))
            print(f"[shap] {iv_method}: erro máximo de aditividade {iv_err:.2e}", file=sys.stderr)
        cg = group_sums(cv, cols)
        for q, row in enumerate(rows):
            item = {"client": row, "id": r(df[a.id_col].iloc[row]) if a.id_col in df.columns else None,
                    "base": r(cb[q]), "margin": r(cm[q]), "p": r(sigmoid(cm[q])), "p_base": r(sigmoid(cb[q])),
                    "shap": [r(v) for v in cv[q]], "groups": {g: r(v[q]) for g, v in cg.items()},
                    "waterfall": waterfall(float(cb[q]), cv[q], Xc.iloc[q], cols, a.waterfall_top)}
            if ci is not None:
                m = ci[q, :-1, :-1]
                iu = np.triu_indices(len(names), 1)
                pair = 2 * m[iu]  # a interação de um par aparece dividida em (i,j) e (j,i)
                top = np.argsort(-np.abs(pair))[: a.interactions]
                item["interactions"] = [{"a": names[iu[0][t]], "b": names[iu[1][t]], "shap": r(pair[t])} for t in top]
                item["main"] = [r(v) for v in np.diag(m)]
            if a.interventional:
                item["shap_interventional"] = [r(v) for v in iv[q]]
                item["base_interventional"] = r(ib[q])
            clients.append(item)

    out = {
        "meta": {"run": a.run, "method": "TreeSHAP path-dependent (xgboost pred_contribs)", "units": "log-odds",
                 "threshold": a.threshold, "n_sample": n, "additivity_max_err": add_err,
                 "base": r(float(base[0])), "p_base": r(float(sigmoid(base[0]))),
                 "interventional": iv_method if a.interventional and rows else None,
                 "background": a.background if a.interventional else None,
                 "xgboost": xgb.__version__, "seconds": round(time.time() - t0, 1), "cols": cols},
        "global": {"cols": global_cols, "groups": global_groups, "dependence": dependence},
        "clients": clients,
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, ensure_ascii=False))
    a.out_js.parent.mkdir(parents=True, exist_ok=True)
    a.out_js.write_text("window.SHAPX = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"[ok] {a.out_json} e {a.out_js} em {time.time() - t0:.1f}s", file=sys.stderr)

    # resumo no terminal
    print("\nImportância global (média de |SHAP|, log-odds):")
    for c in global_cols[:10]:
        print(f"  {c['orig']:>6}  {c['mean_abs']:.4f}  ({c['name']})")
    print("Por grupo:", ", ".join(f"{g['group']} {g['mean_abs']:.3f}" for g in global_groups))
    for c in clients:
        top = ", ".join(f"{w['orig']} {w['shap']:+.3f}" for w in c["waterfall"][:5])
        print(f"Cliente {c['client']}: p {c['p']:.3f} (base {c['p_base']:.3f}); maiores: {top}")


if __name__ == "__main__":
    main()
