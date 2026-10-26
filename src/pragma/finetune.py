"""Tarefas supervisionadas sobre o modelo pré-treinado (ou do zero), com TensorBoard.

Uso:
    python -m src.pragma.finetune --config configs/pragma/finetune.yaml \
        --set init=runs/pragma/<run>/ckpt_best.pt data.dataset=home_credit mode=lora
    python -m src.pragma.finetune --config configs/pragma/finetune.yaml --set init=null mode=full   # do zero

Modos (mode):
    probe   embeddings congelados + logística (sklearn); sem gradiente
    linear  backbone congelado, só a head treina
    lora    adaptadores LoRA nas projeções de atenção + head
    full    tudo treina (o backbone com lr, a head com head_lr)
init = null treina o backbone do zero, com a arquitetura de model: (a referência "sem pré-treino").

data.label_frac < 1 usa só essa fração das linhas de treino (curva de rótulos). A melhor
época na validação (métrica principal da tarefa: AUC, ou a métrica AmEx) é avaliada no teste,
e o resultado vai para results.json e para a aba HParams.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .config import flatten, load_config, save_config
from .data import EventDataset, make_loader, to_device
from .evaluate import embed, linear_probe, primary_metric, task_metrics
from .model import ClassifierHead, PragmaConfig, PragmaModel, apply_lora
from .store import Store
from .train import ROOT, Logger, load_model, lr_lambda, store_metas


def subsample(rows, n, seed):
    if n is None or len(rows) <= n:
        return rows
    return np.sort(np.random.default_rng(seed).choice(rows, n, replace=False))


@torch.no_grad()
def predict(model, head, metas, dataset, batch_size, device, num_workers, amp):
    model.eval()
    head.eval()
    P, Y = [], []
    for batch in make_loader(dataset, batch_size, shuffle=False, num_workers=num_workers):
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = head(model(batch, metas)["client"])
        P.append(torch.sigmoid(logits.float()).cpu())
        Y.append(batch["y"].cpu())
    return torch.cat(Y).numpy(), torch.cat(P).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "pragma" / "finetune.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="CHAVE=VALOR")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    if args.run_name:
        cfg["run_name"] = args.run_name
    dc, oc, lc = cfg["data"], cfg["optim"], cfg["log"]
    cfg["run_name"] = cfg.get("run_name") or f"{dc['dataset']}_{cfg['mode']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = ROOT / cfg["out_dir"] / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    torch.manual_seed(cfg["seed"])
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(oc.get("amp", True)) and device.startswith("cuda")
    nw = dc["num_workers"]

    store = Store(ROOT / dc["root"] / dc["dataset"])
    task = store.task
    metas = store_metas(store, device)
    tr_rows = store.label_rows("train")
    tr_rows = subsample(tr_rows, int(round(dc["label_frac"] * len(tr_rows))), cfg["seed"])
    va_rows = subsample(store.label_rows("val"), lc.get("val_max"), 0)
    te_rows = subsample(store.label_rows("test"), lc.get("test_max"), 0)
    kw = {"max_events": dc["max_events"]}
    train_ds = EventDataset.labelled(store, "train", tr_rows, **kw)
    val_ds = EventDataset.labelled(store, "val", va_rows, **kw)
    test_ds = EventDataset.labelled(store, "test", te_rows, **kw)
    print(store.summary())
    print(f"linhas: treino {len(train_ds)} (label_frac={dc['label_frac']}), validação {len(val_ds)}, teste {len(test_ds)}")

    if cfg["init"]:
        model, _ = load_model(ROOT / cfg["init"], device)
    else:
        model = PragmaModel(PragmaConfig(**cfg["model"])).to(device)
    log = Logger(run_dir)
    log.tb.add_text("config", "```yaml\n" + (run_dir / "config.yaml").read_text() + "\n```", 0)
    metric = primary_metric(task)
    bs = oc["batch_size"]

    if cfg["mode"] == "probe":
        e_tr, y_tr = embed(model, metas, train_ds, bs * 2, device, nw, amp)
        e_va, y_va = embed(model, metas, val_ds, bs * 2, device, nw, amp)
        e_te, y_te = embed(model, metas, test_ds, bs * 2, device, nw, amp)
        val, _ = linear_probe(e_tr, y_tr, e_va, y_va, task, C=cfg["probe_C"])
        test, _ = linear_probe(e_tr, y_tr, e_te, y_te, task, C=cfg["probe_C"])
        best_step = 0
    else:
        head = ClassifierHead(model.cfg.d_model, len(task["labels"]), **cfg["head"]).to(device)
        if cfg["mode"] == "linear":
            model.requires_grad_(False)
        elif cfg["mode"] == "lora":
            apply_lora(model, **cfg["lora"])
            model.to(device)
        backbone = [q for q in model.parameters() if q.requires_grad]
        groups = [{"params": list(head.parameters()), "lr": oc["head_lr"]}]
        if backbone:
            groups.append({"params": backbone, "lr": oc["lr"]})
        opt = torch.optim.AdamW(groups, weight_decay=oc["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(int(oc["warmup"] * oc["steps"]), oc["steps"]))
        pos_weight = None
        if oc.get("pos_weight") == "auto":
            prev = np.clip(store.label_y[tr_rows].mean(0), 1e-4, None)
            pos_weight = torch.tensor(np.minimum((1 - prev) / prev, oc.get("max_pos_weight", 50.0)),
                                      dtype=torch.float32, device=device)
        print(f"modo={cfg['mode']}  parâmetros treináveis: {(sum(q.numel() for q in backbone) + sum(q.numel() for q in head.parameters())) / 1e6:.2f}M")

        loader = iter(make_loader(train_ds, bs, infinite=True, num_workers=nw, seed=cfg["seed"]))
        best, best_step, best_state = -np.inf, 0, None
        run_loss, t_log = 0.0, time.time()
        for step in range(1, oc["steps"] + 1):
            model.train(cfg["mode"] != "linear")
            head.train()
            batch = to_device(next(loader), device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = head(model(batch, metas)["client"]).float()
            y = batch["y"]
            known = ~torch.isnan(y)
            loss = F.binary_cross_entropy_with_logits(logits[known], y[known],
                                                      pos_weight=pos_weight.expand_as(y)[known] if pos_weight is not None else None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], oc["grad_clip"])
            opt.step()
            sched.step()
            run_loss += loss.item()
            if step % lc["log_every"] == 0:
                log.scalars(step, {"train/loss": run_loss / lc["log_every"], "opt/lr": sched.get_last_lr()[-1],
                                   "opt/grad_norm": grad_norm.item(),
                                   "perf/steps_per_sec": lc["log_every"] / (time.time() - t_log)})
                run_loss, t_log = 0.0, time.time()
            if step % lc["eval_every"] == 0 or step == oc["steps"]:
                y_va, p_va = predict(model, head, metas, val_ds, bs * 2, device, nw, amp)
                val = task_metrics(task, y_va, p_va)
                log.scalars(step, {f"val/{k}": v for k, v in val.items()})
                print(f"passo {step:6d}  " + "  ".join(f"val/{k}={val[k]:.4f}" for k in (metric, "ks", "brier") if k in val),
                      flush=True)
                if val.get(metric, -np.inf) > best:
                    best, best_step = val[metric], step
                    best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
                t_log = time.time()
        model.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
        torch.save({"model_config": vars(model.cfg), "state_dict": model.state_dict(), "head": head.state_dict(),
                    "config": cfg, "step": best_step}, run_dir / "ckpt_best.pt")
        y_va, p_va = predict(model, head, metas, val_ds, bs * 2, device, nw, amp)
        val = task_metrics(task, y_va, p_va)
        y_te, p_te = predict(model, head, metas, test_ds, bs * 2, device, nw, amp)
        test = task_metrics(task, y_te, p_te)

    log.scalars(best_step, {f"test/{k}": v for k, v in test.items()})
    results = {"best_step": best_step, "val": val, "test": test}
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("teste: " + "  ".join(f"{k}={test[k]:.4f}" for k in dict.fromkeys((metric, "auc", "ks", "brier")) if k in test))
    hparams = {k: v for k, v in flatten(cfg).items() if not k.startswith(("log.", "out_dir", "run_name"))}
    log.tb.add_hparams(hparams, {f"hparam/val_{metric}": val.get(metric, float("nan")),
                                 f"hparam/test_{metric}": test.get(metric, float("nan")),
                                 "hparam/test_ks": test.get("ks", float("nan"))}, run_name=".")
    log.close()


if __name__ == "__main__":
    main()
