"""Explicação por mascaramento da decisão de crédito de clientes da AmEx.

O pré-treino ensinou o modelo a ler entradas mascaradas: célula, evento inteiro e coluna em
todos os eventos (objective.py). Mascarar é então um jeito de tirar informação sem criar uma
entrada fora da distribuição (como zerar um valor): o embedding do cliente muda e a logística
do probe diz quanto a probabilidade de default muda.

    1. probe     logística sobre o embedding congelado (amostra do treino), avaliada na validação
    2. exemplo   entre os clientes de validação negados (p >= limiar) que de fato deram default
                 e têm todos os extratos, o mais próximo do centróide dos negados no embedding
    3. no exemplo
                 grupos de colunas (prefixo do AmEx), em todos os extratos e só no último
                 extratos (tempo): cada um sozinho, os k últimos e todos menos o último
                 cada coluna em todos os extratos (como a máscara de tipo do pré-treino)
                 busca gulosa do menor conjunto de colunas que, mascarado, leva à aprovação
    4. população grupos, tempo e colunas em N clientes negados, para ver se o exemplo é típico

Mascarar mede a contribuição que o modelo não consegue recuperar do resto da entrada: uma
coluna redundante com outra (ou com outro extrato) pode ser reconstruída e parecer irrelevante.
E mede o modelo, não a causa no mundo: ver Janzing et al. (2020) e Karimi et al. (2021).

Uso:
    python -m src.pragma.explain_masking --ckpt runs/pragma/base_v2b/ckpt_best.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.ldm.table import CAT, LOCAL_NULL

from .data import EventDataset, collate, to_device
from .evaluate import embed, task_metrics
from .store import Store
from .train import ROOT, load_model, store_metas

SOURCE = "statement"
GROUPS = ["delinquency", "spend", "payment", "balance", "risk"]
PREFIX = {"delinquency": "D", "spend": "S", "payment": "P", "balance": "B", "risk": "R"}


def original_name(col):
    group, num = col.rsplit(" ", 1)
    return f"{PREFIX[group]}_{num}"


class Masker:
    """Probabilidade de default de um cliente sob várias máscaras (uma cópia do cliente por máscara)."""

    def __init__(self, model, metas, scaler, clf, device, batch_size=64):
        self.model, self.metas, self.scaler, self.clf = model, metas, scaler, clf
        self.device, self.batch_size = device, batch_size

    @torch.no_grad()
    def __call__(self, item, specs):
        """specs: lista de (cells (E, C) bool, events (E,) bool). Retorna p (len(specs),)."""
        out = []
        for i in range(0, len(specs), self.batch_size):
            chunk = specs[i:i + self.batch_size]
            batch = to_device(collate([item] * len(chunk), [SOURCE]), self.device)
            cells = torch.as_tensor(np.concatenate([c | e[:, None] for c, e in chunk]), device=self.device)
            events = torch.as_tensor(np.concatenate([e for _, e in chunk]), device=self.device)
            masks = {"cells": {SOURCE: cells}, "events": {SOURCE: events}, "profile": None}
            emb = self.model(batch, self.metas, masks)["client"].float().cpu().numpy()
            out.append(self.clf.predict_proba(self.scaler.transform(emb))[:, 1])
        return np.concatenate(out)


def spec(E, C, cols=None, events=None, cells_events=None):
    """cols: colunas mascaradas; cells_events: em quais extratos (padrão: todos); events: extratos inteiros."""
    cells = np.zeros((E, C), dtype=bool)
    if cols is not None:
        rows = np.ones(E, dtype=bool) if cells_events is None else cells_events
        cells[np.ix_(rows, cols)] = True
    ev = np.zeros(E, dtype=bool) if events is None else events
    return cells, ev


def column_values(encoder, item):
    """Por coluna: valor na escala original e percentil (NUM) ou categoria (CAT) em cada extrato."""
    local, z = item["events"][SOURCE][0], item["events"][SOURCE][1]
    df = encoder.decode(local, z)
    out = {}
    for j, c in enumerate(encoder.columns):
        vals, pct = [], []
        for e in range(len(local)):
            k = int(local[e, j])
            if k == LOCAL_NULL:
                vals.append(None)
                pct.append(None)
            elif c.kind == CAT:
                vals.append(str(df.iloc[e, j]))
                pct.append(None)
            else:
                vals.append(float(df.iloc[e, j]))
                pct.append(float(c.bin_q[k - 1]))
        out[c.name] = {"values": vals, "pct": pct}
    return out


def explain_example(masker, item, columns, groups, thr, greedy_steps):
    E, C = item["events"][SOURCE][0].shape
    last = np.zeros(E, dtype=bool)
    last[-1] = True
    p0 = float(masker(item, [spec(E, C)])[0])

    gspecs = [spec(E, C, cols=groups[g]) for g in GROUPS] + [spec(E, C, cols=groups[g], cells_events=last) for g in GROUPS]
    gp = masker(item, gspecs)
    group_rows = [{"group": g, "prefix": PREFIX[g], "n_cols": int(len(groups[g])),
                   "p_all": float(gp[i]), "p_last": float(gp[len(GROUPS) + i])} for i, g in enumerate(GROUPS)]

    single = [spec(E, C, events=np.eye(E, dtype=bool)[e]) for e in range(E)]
    lastk = [spec(E, C, events=np.arange(E) >= E - k) for k in range(1, E)]
    only_last = spec(E, C, events=np.arange(E) < E - 1)
    tp = masker(item, single + lastk + [only_last])
    t = item["events"][SOURCE][2]
    time_rows = {"single": [{"event": e, "t": float(t[e]), "p": float(tp[e])} for e in range(E)],
                 "last_k": [{"k": k, "p": float(tp[E + k - 1])} for k in range(1, E)],
                 "only_last": float(tp[-1])}

    cp = masker(item, [spec(E, C, cols=[j]) for j in range(C)])

    # linhas de base: mascarar tudo e mascarar colunas sorteadas, do tamanho de cada grupo
    rng = np.random.default_rng(1)
    sizes = sorted({len(groups[g]) for g in GROUPS})
    rand = [(k, spec(E, C, cols=list(rng.choice(C, k, replace=False)))) for k in sizes for _ in range(20)]
    bp = masker(item, [spec(E, C, events=np.ones(E, dtype=bool)), spec(E, C, cols=list(range(C)))] + [s for _, s in rand])
    baselines = {"all_events": float(bp[0]), "all_cells": float(bp[1]),
                 "random_cols": [{"k": k, "p": [float(x) for x in bp[2:][[j for j, (kk, _) in enumerate(rand) if kk == k]]]}
                                 for k in sizes]}

    chosen, path, p = [], [], p0
    for _ in range(greedy_steps):
        rest = [j for j in range(C) if j not in chosen]
        trial = masker(item, [spec(E, C, cols=chosen + [j]) for j in rest])
        best = int(np.argmin(trial))
        chosen.append(rest[best])
        p = float(trial[best])
        path.append({"col": columns[rest[best]], "p": p})
        if p < thr:
            break
    return p0, group_rows, time_rows, cp, path, baselines


def population(masker, ds, rows, groups, C, rng, n):
    rows = rng.choice(rows, min(n, len(rows)), replace=False)
    G, D, T, R, flips, p0 = [], [], [], [], [], []
    for r in rows:
        item = ds[int(r)]
        E = item["events"][SOURCE][0].shape[0]
        # T: último extrato, todos menos o último, todos os extratos; R: colunas sorteadas do tamanho de cada grupo
        specs = ([spec(E, C)] + [spec(E, C, cols=groups[g]) for g in GROUPS]
                 + [spec(E, C, events=np.arange(E) == E - 1), spec(E, C, events=np.arange(E) < E - 1),
                    spec(E, C, events=np.ones(E, dtype=bool))]
                 + [spec(E, C, cols=list(rng.choice(C, len(groups[g]), replace=False))) for g in GROUPS]
                 + [spec(E, C, cols=[j]) for j in range(C)])
        p = masker(item, specs)
        g = len(GROUPS)
        p0.append(p[0])
        G.append(p[1:1 + g] - p[0])
        T.append(p[1 + g:4 + g] - p[0])
        R.append(p[4 + g:4 + 2 * g] - p[0])
        D.append(p[4 + 2 * g:] - p[0])
        flips.append(p[1:1 + g])
    return np.array(G), np.array(T), np.array(R), np.array(D), np.array(flips), np.array(p0), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/pragma/base_v2b/ckpt_best.pt")
    ap.add_argument("--n-train", type=int, default=50000, help="linhas de treino da logística")
    ap.add_argument("--threshold", type=float, default=0.5, help="p(default) >= limiar nega o crédito")
    ap.add_argument("--n-pop", type=int, default=400, help="clientes negados na análise de população")
    ap.add_argument("--greedy-steps", type=int, default=25)
    ap.add_argument("--max-events", type=int, default=64)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "explain" / "amex_masking.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    store = Store(ROOT / "data" / "banking" / "prepared" / "amex")
    metas = store_metas(store, device)
    model, ckpt = load_model(ROOT / args.ckpt, device)
    model.eval()
    encoder = store.sources[SOURCE].encoder
    columns = [c.name for c in encoder.columns]
    groups = {g: [j for j, c in enumerate(columns) if c.startswith(g + " ")] for g in GROUPS}
    kw = {"max_events": args.max_events}

    tr_rows = np.sort(rng.choice(store.label_rows("train"), args.n_train, replace=False))
    va_rows = store.label_rows("val")
    val_ds = EventDataset.labelled(store, "val", va_rows, **kw)
    cache = args.out.parent / f"amex_emb_{Path(args.ckpt).parent.name}_{ckpt.get('step', 0)}_{args.n_train}_{args.seed}.npz"
    if cache.exists():
        z = np.load(cache)
        e_tr, y_tr, e_va, y_va = z["e_tr"], z["y_tr"], z["e_va"], z["y_va"]
        print(f"embeddings do cache {cache.name}", flush=True)
    else:
        print(f"embeddings: treino {len(tr_rows)}, validação {len(va_rows)}", flush=True)
        # lotes pequenos: a análise divide a GPU com a fila de fine-tuning
        e_tr, y_tr = embed(model, metas, EventDataset.labelled(store, "train", tr_rows, **kw), 64, device, 4, amp=False)
        e_va, y_va = embed(model, metas, val_ds, 64, device, 4, amp=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, e_tr=e_tr, y_tr=y_tr, e_va=e_va, y_va=y_va)
    scaler = StandardScaler().fit(e_tr)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(e_tr), y_tr[:, 0])
    p_va = clf.predict_proba(scaler.transform(e_va))[:, 1]
    probe = task_metrics(store.task, y_va, p_va[:, None])
    print(f"probe: AUC {probe['auc']:.4f}  AmEx {probe['amex']:.4f}", flush=True)

    reject = p_va >= args.threshold
    n_ev = np.array([min(int(np.diff(store.sources[SOURCE].offsets[c:c + 2])[0]), args.max_events)
                     for c in store.label_client[va_rows]])
    Zva = scaler.transform(e_va)
    centroid = Zva[reject].mean(0)
    dist = np.linalg.norm(Zva - centroid, axis=1)
    cand = np.nonzero(reject & (y_va[:, 0] == 1) & (n_ev == n_ev.max()))[0]
    i = int(cand[np.argmin(dist[cand])])
    item = val_ds[i]
    print(f"exemplo: linha de validação {i}, p={p_va[i]:.3f}", flush=True)

    masker = Masker(model, metas, scaler, clf, device)
    p0, group_rows, time_rows, cp, path, baselines = explain_example(masker, item, columns, groups, args.threshold,
                                                                    args.greedy_steps)
    values = column_values(encoder, item)
    order = np.argsort(cp - p0)
    col_rows = [{"col": columns[j], "orig": original_name(columns[j]), "group": columns[j].rsplit(" ", 1)[0],
                 "p": float(cp[j]), "d": float(cp[j] - p0), **values[columns[j]]} for j in order]

    print(f"população: {args.n_pop} clientes negados", flush=True)
    pop_rows = np.nonzero(reject)[0]
    G, T, R, D, flips, pop_p0, sampled = population(masker, val_ds, pop_rows, groups, len(columns), rng, args.n_pop)
    rank = np.argsort(np.argsort(D, axis=1), axis=1)  # 0 = a que mais baixa p
    pop_cols = [{"col": columns[j], "orig": original_name(columns[j]), "mean_d": float(D[:, j].mean()),
                 "top5_rate": float((rank[:, j] < 5).mean())} for j in np.argsort(D.mean(0))]

    result = {
        "meta": {"ckpt": args.ckpt, "step": int(ckpt.get("step", -1)), "threshold": args.threshold,
                 "n_train": int(len(tr_rows)), "n_val": int(len(va_rows)),
                 "val_auc": float(probe["auc"]), "val_amex": float(probe["amex"]),
                 "reject_rate": float(reject.mean()), "default_rate": float(y_va[:, 0].mean()),
                 "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t_start)},
        "example": {"val_row": i, "client": int(store.label_client[va_rows[i]]), "y": int(y_va[i, 0]), "p": p0,
                    "p_pct_rejected": float((p_va[reject] <= p0).mean()),
                    "dist_pct_rejected": float((dist[reject] <= dist[i]).mean()),
                    "n_events": int(n_ev[i]), "t": [float(x) for x in item["events"][SOURCE][2]]},
        "groups": group_rows,
        "time": time_rows,
        "columns": col_rows,
        "greedy": path,
        "baselines": baselines,
        "population": {
            "n": int(len(sampled)), "mean_p": float(pop_p0.mean()),
            "groups": [{"group": g, "n_cols": int(len(groups[g])), "mean_d": float(G[:, k].mean()),
                        "median_d": float(np.median(G[:, k])), "random_mean_d": float(R[:, k].mean()),
                        "flip_rate": float((flips[:, k] < args.threshold).mean())} for k, g in enumerate(GROUPS)],
            "time": {"mask_last_mean_d": float(T[:, 0].mean()), "only_last_mean_d": float(T[:, 1].mean()),
                     "all_events_mean_d": float(T[:, 2].mean())},
            "columns": pop_cols,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"-> {args.out} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
