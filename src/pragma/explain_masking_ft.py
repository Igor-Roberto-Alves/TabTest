"""Mascaramento de um cliente real do AmEx pelo modelo fine-tunado (cabeça treinada), sem logística de probe.

A probabilidade de default vem direto do modelo da tarefa: sigmoid(head(embedding do cliente mascarado)).
Escolhe, entre os clientes de validação negados (p >= limiar) que de fato deram default, o de maior número de
extratos e p mais próxima da mediana dos negados, e roda os mesmos mascaramentos de explain_masking.py.

Uso:
    python -m src.pragma.explain_masking_ft --run runs/pragma_ft/ft_amex_v2b_full_20k
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import EventDataset, collate, to_device
from .explain_masking import (GROUPS, SOURCE, column_values, explain_example, original_name)
from .evaluate import task_metrics
from .finetune import predict
from .model import ClassifierHead
from .store import Store
from .train import ROOT, load_model, store_metas
from .config import load_config


class FtMasker:
    """Como Masker, mas com a cabeça do fine-tuning no lugar da logística."""

    def __init__(self, model, head, metas, device, batch_size=64):
        self.model, self.head, self.metas, self.device, self.batch_size = model, head, metas, device, batch_size

    @torch.no_grad()
    def __call__(self, item, specs):
        out = []
        for i in range(0, len(specs), self.batch_size):
            chunk = specs[i:i + self.batch_size]
            batch = to_device(collate([item] * len(chunk), [SOURCE]), self.device)
            cells = torch.as_tensor(np.concatenate([c | e[:, None] for c, e in chunk]), device=self.device)
            events = torch.as_tensor(np.concatenate([e for _, e in chunk]), device=self.device)
            masks = {"cells": {SOURCE: cells}, "events": {SOURCE: events}, "profile": None}
            logits = self.head(self.model(batch, self.metas, masks)["client"].float())
            out.append(torch.sigmoid(logits.float())[:, 0].cpu().numpy())
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/pragma_ft/ft_amex_v2b_full_20k")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-scan", type=int, default=6000, help="clientes de validação varridos para escolher o exemplo")
    ap.add_argument("--greedy-steps", type=int, default=25)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "explain" / "amex_masking_ft.json")
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = ROOT / args.run
    cfg = load_config(run / "config.yaml", [])
    store = Store(ROOT / "data" / "banking" / "prepared" / "amex")
    metas = store_metas(store, device)
    model, ckpt = load_model(run / "ckpt_best.pt", device)
    head = ClassifierHead(model.cfg.d_model, len(store.task["labels"]), **cfg["head"]).to(device)
    head.load_state_dict(ckpt["head"])
    model.eval(); head.eval()
    encoder = store.sources[SOURCE].encoder
    columns = [c.name for c in encoder.columns]
    groups = {g: [j for j, c in enumerate(columns) if c.startswith(g + " ")] for g in GROUPS}

    max_ev = cfg["data"]["max_events"].get(SOURCE, cfg["data"]["max_events"]["default"])
    rng = np.random.default_rng(0)
    va_rows = np.sort(rng.choice(store.label_rows("val"), args.n_scan, replace=False))
    ds = EventDataset.labelled(store, "val", va_rows, max_events=max_ev)
    y, p = predict(model, head, metas, ds, 64, device, 4, amp=False)
    p = p[:, 0]
    m = task_metrics(store.task, y, p[:, None])
    print(f"varredura: {len(va_rows)} clientes, AUC {m['auc']:.4f}  AmEx {m['amex']:.4f}", flush=True)
    n_ev = np.array([min(int(np.diff(store.sources[SOURCE].offsets[c:c + 2])[0]), max_ev)
                     for c in store.label_client[va_rows]])
    reject = p >= args.threshold
    cand = np.nonzero(reject & (y[:, 0] == 1) & (n_ev == n_ev.max()))[0]
    i = int(cand[np.argmin(np.abs(p[cand] - np.median(p[reject])))])
    item = ds[i]
    print(f"exemplo: cliente {int(store.label_client[va_rows[i]])}, p={p[i]:.3f}, {n_ev[i]} extratos", flush=True)

    masker = FtMasker(model, head, metas, device)
    p0, group_rows, time_rows, cp, path, baselines = explain_example(masker, item, columns, groups, args.threshold, args.greedy_steps)
    values = column_values(encoder, item)
    order = np.argsort(cp - p0)
    col_rows = [{"col": columns[j], "orig": original_name(columns[j]), "group": columns[j].rsplit(" ", 1)[0],
                 "p": float(cp[j]), "d": float(cp[j] - p0), **values[columns[j]]} for j in order]
    result = {"meta": {"run": args.run, "threshold": args.threshold, "n_scan": int(len(va_rows)), "scan_auc": float(m["auc"]),
                       "scan_amex": float(m["amex"]), "reject_rate": float(reject.mean()), "seconds": round(time.time() - t0)},
              "example": {"client": int(store.label_client[va_rows[i]]), "y": int(y[i, 0]), "p": p0, "n_events": int(n_ev[i]),
                          "t": [float(x) for x in item["events"][SOURCE][2]]},
              "groups": group_rows, "time": time_rows, "columns": col_rows, "greedy": path, "baselines": baselines}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"-> {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
