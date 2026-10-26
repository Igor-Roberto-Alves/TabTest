"""Pré-treina o Large Data Model tabular em várias tabelas.

Uso:
    python -m src.ldm.corpus_download                  # tabelas extras do OpenML -> data/corpus/
    python -m src.ldm.pretrain --exclude adult --steps 20000 --save checkpoints/ldm.pt
    python -m src.run --task adult_income --models ldm --ldm-checkpoint checkpoints/ldm.pt

Objetivo (--objective): "diffusion" (padrão), "mlm" (estilo BERT, focado em predição) ou
"icl" (em contexto: atenção também entre linhas, prediz sem fine-tuning). Veja diffusion.py.

Encoder de texto (--text-encoder): "hash" (n-gramas com hash, padrão), "frozen" (MiniLM
congelado, só a projeção treina) ou "finetune" (gradiente flui pelo MiniLM, com --text-lr).

Por padrão usa todo CSV em data/ e data/corpus/. Exclua (--exclude) os datasets em que
o modelo será avaliado: o pré-treino vê todas as linhas das tabelas do corpus.
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .corpus import TableCorpus, find_tables
from .diffusion import MaskedDiffusion
import numpy as np

from .model import LDMConfig, TabularLDM

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def build_diffusion(cfg: LDMConfig, schedule="cosine", weighting="elbo", objective="diffusion") -> MaskedDiffusion:
    return MaskedDiffusion(TabularLDM(cfg), schedule=schedule, weighting=weighting, objective=objective)


def save_checkpoint(path, diffusion, tables, encoder_kwargs, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(diffusion.model.cfg),
        "state_dict": diffusion.model.state_dict(),
        "schedule": diffusion.schedule.kind,
        "weighting": diffusion.weighting,
        "objective": diffusion.objective,
        "tables": tables,
        "encoder_kwargs": encoder_kwargs,
        "step": step,
    }, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    diffusion = build_diffusion(LDMConfig(**ckpt["config"]), ckpt["schedule"], ckpt["weighting"],
                                ckpt.get("objective", "diffusion"))
    diffusion.model.load_state_dict(ckpt["state_dict"])
    return diffusion.to(device), ckpt


def lr_lambda(warmup_steps, total_steps):
    """Warmup linear seguido de decaimento cosseno."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return f


@torch.no_grad()
def evaluate(diffusion, corpus, args):
    """Loss nas linhas de validação. Lotes e máscaras fixos (mesma semente): comparável entre execuções."""
    diffusion.eval()
    rng = np.random.default_rng(12345)
    sums = {}
    with torch.random.fork_rng(devices=[args.device] if args.device.startswith("cuda") else []):
        torch.manual_seed(12345)
        for _ in range(args.val_batches):
            _, local, z, meta = corpus.sample(args.batch_size, args.device, split="val", rng=rng,
                                              replace=args.objective != "icl")
            for k, v in diffusion.loss(local, z, meta).items():
                sums[k] = sums.get(k, 0.0) + v.item()
    return {k: v / args.val_batches for k, v in sums.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dirs", nargs="+", type=Path, default=[DATA_DIR, DATA_DIR / "corpus"])
    p.add_argument("--exclude", nargs="*", default=[], help="datasets fora do pré-treino (os de avaliação)")
    p.add_argument("--max-rows", type=int, default=50_000, help="linhas por tabela")
    p.add_argument("--max-cols", type=int, default=64, help="colunas por lote")
    p.add_argument("--temperature", type=float, default=0.5, help="amostragem de tabelas ∝ n^temperature")
    # modelo
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--text-encoder", choices=["hash", "frozen", "finetune"], default="hash")
    p.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--objective", choices=["diffusion", "mlm", "icl"], default="diffusion")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--weighting", choices=["elbo", "uniform"], default="elbo")
    # otimização
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--text-lr", type=float, default=3e-5, help="lr do encoder de texto em --text-encoder finetune")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.02, help="fração dos passos com warmup linear")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-frac", type=float, default=0.05, help="linhas de cada tabela reservadas para validação")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--log-file", type=Path, default=None, help="métricas em JSONL")
    p.add_argument("--save", type=Path, default=Path("checkpoints/ldm.pt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    paths = find_tables([d for d in args.dirs if d.exists()], args.exclude)
    corpus = TableCorpus(paths, args.max_rows, args.max_cols, args.temperature, args.seed, val_frac=args.val_frac)
    print(f"corpus: {len(corpus.tables)} tabelas, {sum(len(t) for t in corpus.tables)} linhas")
    for t, p in zip(corpus.tables, corpus.probs):
        print(f"  {t.name:40s} {len(t):>7} x {t.local.shape[1]:<4} p={p:.3f}")

    cfg = LDMConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, dropout=args.dropout,
                    text_encoder=args.text_encoder, text_model=args.text_model, in_context=args.objective == "icl")
    diffusion = build_diffusion(cfg, args.schedule, args.weighting, args.objective).to(args.device)
    trainable = [p for p in diffusion.parameters() if p.requires_grad]
    print(f"parâmetros: {sum(p.numel() for p in diffusion.parameters()) / 1e6:.1f}M "
          f"({sum(p.numel() for p in trainable) / 1e6:.1f}M treináveis)  objetivo={args.objective}  "
          f"texto={args.text_encoder}  device={args.device}")

    # o encoder de texto pré-treinado (só treinável em finetune) usa um lr menor
    text_encoder = getattr(diffusion.model.text, "encoder", None)
    text_params = {id(p) for p in text_encoder.parameters()} if text_encoder is not None else set()
    groups = [{"params": [p for p in trainable if id(p) not in text_params], "lr": args.lr},
              {"params": [p for p in trainable if id(p) in text_params], "lr": args.text_lr}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(max(1, int(args.warmup * args.steps)), args.steps))
    log = open(args.log_file, "w") if args.log_file else None
    sums, t0, t_log = {}, time.time(), time.time()
    for step in range(1, args.steps + 1):
        diffusion.train()
        _, local, z, meta = corpus.sample(args.batch_size, args.device, replace=args.objective != "icl")
        stats = diffusion.loss(local, z, meta)
        opt.zero_grad(set_to_none=True)
        stats["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        stats["grad_norm"] = grad_norm
        opt.step()
        sched.step()
        for k, v in stats.items():
            sums[k] = sums.get(k, 0.0) + v.item()
        if step % args.log_every == 0 or step == args.steps:
            n = step % args.log_every or args.log_every
            means = {k: v / n for k, v in sums.items()}
            msg = "  ".join(f"{k}={v:.4f}" for k, v in means.items())
            speed = n / (time.time() - t_log)
            print(f"passo {step:6d}  {msg}  lr={sched.get_last_lr()[0]:.2e}  {speed:.1f} passos/s  ({time.time() - t0:.0f}s)",
                  flush=True)
            if log:
                log.write(json.dumps({"split": "train", "step": step, **means, "lr": sched.get_last_lr()[0],
                                      "steps_per_sec": speed, "elapsed": time.time() - t0}) + "\n")
                log.flush()
            sums, t_log = {}, time.time()
        if args.val_frac > 0 and (step % args.val_every == 0 or step == args.steps):
            val = evaluate(diffusion, corpus, args)
            print(f"passo {step:6d}  val  " + "  ".join(f"{k}={v:.4f}" for k, v in val.items()), flush=True)
            if log:
                log.write(json.dumps({"split": "val", "step": step, **val, "elapsed": time.time() - t0}) + "\n")
                log.flush()
            t_log = time.time()
    save_checkpoint(args.save, diffusion, corpus.names, {}, args.steps)
    print(f"checkpoint salvo em {args.save}")


if __name__ == "__main__":
    main()
