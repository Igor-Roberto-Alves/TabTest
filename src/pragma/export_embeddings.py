"""Embedding do cliente ([CLS]) de um checkpoint para cada linha de rótulo, como features do XGBoost.

Uso:
    python -m src.pragma.export_embeddings runs/pragma/base_v1/ckpt_best.pt --tag base_v1
    python -m src.baselines.xgb home_credit --emb base_v1          # features à mão + embedding
    python -m src.baselines.xgb home_credit --emb base_v1 --emb-only

Cada linha de rótulo do store (labels/client.npy, labels/cutoff.npy) vira uma amostra com o mesmo
corte do fine-tuning, então no MBD o embedding de (cliente, mês m) só vê eventos antes do dia 1 de m.
Use só checkpoints de pré-treino: o pré-treino não vê rótulos nem clientes de validação/teste
(pretrain_splits), então o embedding não vaza o alvo. Um checkpoint de fine-tuning vazaria, porque
já viu os rótulos do treino.

Saída: data/banking/features/emb_<tag>_<base>.npy, float16 (R, d), alinhado às linhas de rótulo.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import EventDataset
from .evaluate import embed
from .store import Store
from .train import load_model, store_metas

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "banking" / "features"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", type=Path)
    p.add_argument("--tag", required=True, help="nome curto do checkpoint no arquivo de saída")
    p.add_argument("--bases", nargs="+", default=["home_credit", "amex", "mbd_mini"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = load_config(ROOT / "configs" / "pragma" / "finetune.yaml")["data"]
    model, _ = load_model(args.ckpt, device)
    OUT.mkdir(parents=True, exist_ok=True)
    for base in args.bases:
        t0 = time.time()
        store = Store(ROOT / dc["root"] / base)
        ds = EventDataset(store, store.label_client, store.label_cutoff, max_events=dc["max_events"])
        with torch.no_grad():
            emb, _ = embed(model, store_metas(store, device), ds, args.batch_size, device, args.num_workers)
        path = OUT / f"emb_{args.tag}_{base}.npy"
        np.save(path, emb.astype(np.float16))
        print(f"{base}: {emb.shape} -> {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
