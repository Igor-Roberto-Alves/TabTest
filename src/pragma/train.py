"""Pré-treino do modelo fundacional de eventos, com tudo registrado no TensorBoard.

Uso:
    python -m src.pragma.train --config configs/pragma/pretrain.yaml
    python -m src.pragma.train --config configs/pragma/pretrain.yaml --set optim.lr=1e-3 model.d_model=128
    python -m src.pragma.train --config configs/pragma/pretrain.yaml --resume runs/pragma/<run>/ckpt_last.pt
    tensorboard --logdir runs/pragma

Cada execução grava em <out_dir>/<run_name>/:
    config.yaml       config resolvida (com as sobrescritas)
    events.out.*      TensorBoard: train/*, val/*, probe/*, opt/*, perf/* e a aba HParams
    ckpt_last.pt      último checkpoint (modelo, otimizador, passo)
    ckpt_best.pt      menor val/loss
    metrics.jsonl     as mesmas métricas em texto

A cada passo sorteia uma base (prob. ∝ n_clientes^temperature) e um lote de clientes dela.
A validação usa clientes do split de validação, com lotes e máscaras fixos (comparável entre
execuções). O probe ajusta uma logística sobre os embeddings congelados e mede a tarefa
supervisionada de cada base: é o sinal de que o pré-treino está aprendendo algo útil.
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import flatten, load_config, save_config
from .data import EventDataset, make_loader, to_device
from .evaluate import embed, linear_probe
from .model import PragmaConfig, PragmaModel, meta_to_torch
from .objective import MaskConfig, mlm_loss
from .store import Store

ROOT = Path(__file__).resolve().parents[2]


def lr_lambda(warmup_steps, total_steps, min_ratio=0.0):
    """Warmup linear seguido de decaimento cosseno até min_ratio · lr."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


def load_stores(cfg):
    root = ROOT / cfg["data"]["root"]
    return {name: Store(root / name) for name in cfg["data"]["datasets"]}


def store_metas(store, device):
    return {"sources": {k: meta_to_torch(s.encoder.meta(), device) for k, s in store.sources.items()},
            "profile": meta_to_torch(store.profile.encoder.meta(), device) if store.profile else None}


def param_groups(model, cfg):
    """Sem weight decay em vieses, LayerNorm e embeddings; encoder de texto pré-treinado com text_lr."""
    text_enc = getattr(model.vocab.text, "encoder", None)
    text_ids = {id(p) for p in text_enc.parameters()} if text_enc is not None else set()
    emb_ids = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    groups = {"decay": [], "no_decay": [], "text": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in text_ids:
            groups["text"].append(p)
        elif p.ndim < 2 or id(p) in emb_ids:
            groups["no_decay"].append(p)
        else:
            groups["decay"].append(p)
    o = cfg["optim"]
    out = [{"params": groups["decay"], "weight_decay": o["weight_decay"], "lr": o["lr"]},
           {"params": groups["no_decay"], "weight_decay": 0.0, "lr": o["lr"]},
           {"params": groups["text"], "weight_decay": o["weight_decay"], "lr": o.get("text_lr", o["lr"] / 10)}]
    return [g for g in out if g["params"]]


def save_checkpoint(path, model, cfg, step, opt=None, sched=None, extra=None):
    torch.save({"model_config": asdict(model.cfg), "state_dict": model.state_dict(), "config": cfg, "step": step,
                "optimizer": opt.state_dict() if opt else None, "scheduler": sched.state_dict() if sched else None,
                **(extra or {})}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PragmaModel(PragmaConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt


class Logger:
    """TensorBoard + JSONL com as mesmas chaves."""

    def __init__(self, run_dir):
        self.tb = SummaryWriter(run_dir)
        self.jsonl = open(Path(run_dir) / "metrics.jsonl", "a")

    def scalars(self, step, values: dict):
        for k, v in values.items():
            if v is not None and np.isfinite(v):
                self.tb.add_scalar(k, v, step)
        self.jsonl.write(json.dumps({"step": step, **{k: float(v) for k, v in values.items()}}) + "\n")
        self.jsonl.flush()

    def close(self):
        self.tb.flush()
        self.tb.close()
        self.jsonl.close()


@torch.no_grad()
def validate(model, stores, metas, cfg, device, amp):
    """Loss mascarada em lotes fixos de clientes de validação de cada base."""
    model.eval()
    lc, oc = cfg["log"], cfg["optim"]
    mask_cfg = MaskConfig(**cfg["mask"])
    out, losses = {}, []
    for name, store in stores.items():
        clients = store.clients_in(["val"])
        if len(clients) == 0:
            continue
        rng = np.random.default_rng(12345)
        clients = rng.choice(clients, min(len(clients), lc["val_batches"] * oc["batch_size"]), replace=False)
        ds = EventDataset(store, np.sort(clients), max_events=cfg["data"]["max_events"])
        gen = torch.Generator(device=device).manual_seed(12345)
        sums, n = {}, 0
        for batch in make_loader(ds, oc["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"]):
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                stats = mlm_loss(model, batch, metas[name], mask_cfg, generator=gen)
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n += 1
        for k, v in sums.items():
            out[f"val_{name}/{k}"] = v / n
        losses.append(sums["loss"] / n)
    out["val/loss"] = float(np.mean(losses)) if losses else float("nan")
    return out


def probe(model, stores, metas, cfg, device, amp):
    """Linear probe: logística sobre embeddings congelados (treino -> validação), por base."""
    pc = cfg["probe"]
    out = {}
    for name, store in stores.items():
        tr_rows, va_rows = store.label_rows("train"), store.label_rows("val")
        if len(tr_rows) == 0 or len(va_rows) == 0:
            continue
        rng = np.random.default_rng(0)
        tr_rows = rng.choice(tr_rows, min(len(tr_rows), pc["n_train"]), replace=False)
        va_rows = rng.choice(va_rows, min(len(va_rows), pc["n_val"]), replace=False)
        kw = {"max_events": cfg["data"]["max_events"]}
        bs, nw = cfg["optim"]["batch_size"] * 2, cfg["data"]["num_workers"]
        e_tr, y_tr = embed(model, metas[name], EventDataset.labelled(store, "train", tr_rows, **kw), bs, device, nw, amp)
        e_va, y_va = embed(model, metas[name], EventDataset.labelled(store, "val", va_rows, **kw), bs, device, nw, amp)
        metrics, _ = linear_probe(e_tr, y_tr, e_va, y_va, store.task, C=pc["C"])
        out.update({f"probe_{name}/{k}": v for k, v in metrics.items() if "/" not in k})
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "pragma" / "pretrain.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="CHAVE=VALOR", help="sobrescreve a config")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", type=Path, default=None, help="checkpoint para continuar")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if args.run_name:
        cfg["run_name"] = args.run_name
    cfg["run_name"] = cfg.get("run_name") or time.strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / cfg["out_dir"] / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["optim"].get("amp", True)) and device.startswith("cuda")
    oc, lc, dc = cfg["optim"], cfg["log"], cfg["data"]

    stores = load_stores(cfg)
    metas = {name: store_metas(s, device) for name, s in stores.items()}
    datasets = {name: EventDataset.pretrain(s, dc["pretrain_splits"], max_events=dc["max_events"],
                                            time_cutoff=(dc.get("time_cutoff") or {}).get(name))
                for name, s in stores.items()}
    sizes = np.array([len(d) for d in datasets.values()], dtype=float)
    probs = sizes ** dc["temperature"] / (sizes ** dc["temperature"]).sum()
    loaders = {name: iter(make_loader(d, oc["batch_size"], infinite=True, num_workers=dc["num_workers"],
                                      seed=cfg["seed"] + i))
               for i, (name, d) in enumerate(datasets.items())}
    for (name, s), p in zip(stores.items(), probs):
        print(s.summary())
        print(f"  pré-treino: {len(datasets[name])} clientes, prob. por passo {p:.3f}")

    model = PragmaModel(PragmaConfig(**cfg["model"])).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg), betas=tuple(oc["betas"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(int(oc["warmup"] * oc["steps"]), oc["steps"], oc.get("min_lr_ratio", 0.0)))
    start, best = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start, best = ckpt["step"], ckpt.get("best", best)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parâmetros: {n_params / 1e6:.1f}M  device={device}  amp={amp}  run={run_dir}")

    log = Logger(run_dir)
    log.tb.add_text("config", "```yaml\n" + (run_dir / "config.yaml").read_text() + "\n```", start)
    mask_cfg = MaskConfig(**cfg["mask"])
    names = list(stores)
    sums, counts, t_log, events_seen = {}, {}, time.time(), 0
    last_val, last_probe = {}, {}
    try:
        for step in range(start + 1, oc["steps"] + 1):
            model.train()
            name = names[rng.choice(len(names), p=probs)]
            batch = to_device(next(loaders[name]), device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                stats = mlm_loss(model, batch, metas[name], mask_cfg)
            opt.zero_grad(set_to_none=True)
            stats["loss"].backward()
            stats["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"])
            opt.step()
            sched.step()
            events_seen += int(batch["seq_valid"].sum())
            for k, v in stats.items():
                for key in (f"train/{k}", f"train_{name}/{k}"):
                    sums[key] = sums.get(key, 0.0) + v.item()
                    counts[key] = counts.get(key, 0) + 1

            if step % lc["log_every"] == 0:
                dt = time.time() - t_log
                values = {k: v / counts[k] for k, v in sums.items()}
                values.update({"opt/lr": sched.get_last_lr()[0], "perf/steps_per_sec": lc["log_every"] / dt,
                               "perf/events_per_sec": events_seen / dt})
                if device.startswith("cuda"):
                    values["perf/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                log.scalars(step, values)
                print(f"passo {step:7d}  loss={values['train/loss']:.4f}  ce_cell={values['train/ce_cell']:.3f}  "
                      f"ce_event={values['train/ce_event']:.3f}  lr={values['opt/lr']:.2e}  "
                      f"{values['perf/steps_per_sec']:.1f} passos/s", flush=True)
                sums, counts, t_log, events_seen = {}, {}, time.time(), 0

            if step % lc["val_every"] == 0 or step == oc["steps"]:
                last_val = validate(model, stores, metas, cfg, device, amp)
                log.scalars(step, last_val)
                print(f"passo {step:7d}  val/loss={last_val['val/loss']:.4f}", flush=True)
                if last_val["val/loss"] < best:
                    best = last_val["val/loss"]
                    save_checkpoint(run_dir / "ckpt_best.pt", model, cfg, step, extra={"best": best})
                t_log = time.time()

            if cfg["probe"]["every"] and (step % cfg["probe"]["every"] == 0 or step == oc["steps"]):
                last_probe = probe(model, stores, metas, cfg, device, amp)
                log.scalars(step, last_probe)
                print(f"passo {step:7d}  " + "  ".join(f"{k}={v:.4f}" for k, v in last_probe.items()
                                                      if k.endswith(("/auc", "/amex"))), flush=True)
                t_log = time.time()

            if step % lc["ckpt_every"] == 0 or step == oc["steps"]:
                save_checkpoint(run_dir / "ckpt_last.pt", model, cfg, step, opt, sched, extra={"best": best})
    finally:
        hparams = {k: v for k, v in flatten(cfg).items() if not k.startswith(("log.", "out_dir", "run_name"))}
        final = {"hparam/best_val_loss": best if np.isfinite(best) else float("nan"),
                 **{f"hparam/{k}": v for k, v in last_probe.items()}}
        log.tb.add_hparams(hparams, final, run_name=".")
        log.close()


if __name__ == "__main__":
    main()
