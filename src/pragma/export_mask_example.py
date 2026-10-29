"""Exporta para o painel alguns clientes do AmEx com as inferências do modelo fine-tunado pré-computadas.

Varre clientes de validação com todos os extratos, escolhe casos de cada tipo (negado que deu default, negado que
não deu, aprovado que não deu, aprovado que deu, caso limítrofe) e calcula p(default) para cada célula, coluna,
extrato, k últimos extratos e grupo. Saída: runs/explain/amex_mask_examples.json.
"""
import json
import numpy as np
import torch

from .config import load_config
from .data import EventDataset
from .explain_masking import GROUPS, SOURCE, column_values, explain_example, original_name, spec
from .explain_masking_ft import FtMasker
from .finetune import predict
from .model import ClassifierHead
from .store import Store
from .train import ROOT, load_model, store_metas

def main():
    RUN = "runs/pragma_ft/ft_amex_v2b_full_20k"
    THR = 0.5
    run = ROOT / RUN
    cfg = load_config(run / "config.yaml", [])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = Store(ROOT / "data/banking/prepared/amex")
    metas = store_metas(store, device)
    model, ckpt = load_model(run / "ckpt_best.pt", device)
    head = ClassifierHead(model.cfg.d_model, 1, **cfg["head"]).to(device)
    head.load_state_dict(ckpt["head"]); model.eval(); head.eval()
    max_ev = cfg["data"]["max_events"].get(SOURCE, cfg["data"]["max_events"]["default"])
    enc = store.sources[SOURCE].encoder
    cols = [c.name for c in enc.columns]
    groups = {g: [j for j, c in enumerate(cols) if c.startswith(g + " ")] for g in GROUPS}

    rng = np.random.default_rng(0)
    va = np.sort(rng.choice(store.label_rows("val"), 6000, replace=False))
    ds = EventDataset.labelled(store, "val", va, max_events=max_ev)
    y, p = predict(model, head, metas, ds, 64, device, 4, amp=False)
    y, p = y[:, 0], p[:, 0]
    n_ev = np.array([min(int(np.diff(store.sources[SOURCE].offsets[c:c + 2])[0]), max_ev) for c in store.label_client[va]])
    full = n_ev == n_ev.max()
    rj = np.random.default_rng(3)
    def pick(mask, target=None, n=1):
        idx = np.nonzero(mask & full)[0]
        if target is not None: idx = idx[np.argsort(np.abs(p[idx] - target))]
        else: idx = rj.permutation(idx)
        return list(idx[:n])
    sel = [("negado que deu default", pick((p >= THR) & (y == 1), 0.875)[0]),
           ("negado que deu default", pick((p >= THR) & (y == 1), 0.62)[0]),
           ("negado que não deu default", pick((p >= THR) & (y == 0), 0.8)[0]),
           ("aprovado que não deu default", pick((p < THR) & (y == 0), 0.05)[0]),
           ("aprovado que deu default", pick((p < THR) & (y == 1), 0.3)[0]),
           ("caso limítrofe", pick(np.abs(p - THR) < 0.03)[0])]
    m = FtMasker(model, head, metas, device)
    shared = None
    clients = []
    for label, i in sel:
        item = ds[int(i)]
        E, C = item["events"][SOURCE][0].shape
        p0, group_rows, time_rows, cp, path, base = explain_example(m, item, cols, groups, THR, 25)
        cells = m(item, [spec(E, C, cols=[j], cells_events=np.eye(E, dtype=bool)[e]) for e in range(E) for j in range(C)]).reshape(E, C)
        evp = m(item, [spec(E, C, events=np.eye(E, dtype=bool)[e]) for e in range(E)])
        vals = column_values(enc, item)
        r2 = lambda a: np.round(a, 3).tolist()
        clients.append({"client": int(store.label_client[va[i]]), "kind": label, "y": int(y[i]), "p": round(float(p0), 4),
            "t": [float(x) for x in item["events"][SOURCE][2]],
            "v": [[(round(v, 3) if isinstance(v, float) else v) for v in vals[c]["values"]] for c in cols],
            "cell_p": r2(cells), "col_p": r2(cp), "event_p": r2(evp),
            "groups": [{"group": g["group"], "p_all": round(g["p_all"], 3), "p_last": round(g["p_last"], 3)} for g in group_rows],
            "last_k": [round(x["p"], 3) for x in time_rows["last_k"]],
            "all_events": round(base["all_events"], 3), "all_cells": round(base["all_cells"], 3)})
        print(label, clients[-1]["client"], clients[-1]["p"], flush=True)
    out = {"run": RUN, "threshold": THR, "cols": [{"name": c, "orig": original_name(c), "group": c.rsplit(" ", 1)[0]} for c in cols], "clients": clients}
    (ROOT / "runs/explain/amex_mask_examples.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(len(json.dumps(out)))


if __name__ == "__main__":
    main()
