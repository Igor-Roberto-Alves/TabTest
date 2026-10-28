# Rodada 3: base_v2b no PC da esquerda (fine-tuning 20 mil e protocolo B'), base_v1 com protocolo B' no PC da direita.
# python experiments/gen_round3.py .   (regera os dois YAML)
"""Gera experiments/round3_esquerda.yaml (base_v2b) e experiments/round3_direita.yaml (base_v1, protocolo B')."""
import sys, yaml
from pathlib import Path
ROOT = Path(sys.argv[1])
TASK = {"home_credit": "Home Credit (default, AUC)", "amex": "AmEx (default, métrica AmEx)",
        "mbd_mini": "MBD-mini (propensão a 4 produtos, AUC média)"}
CKPT = {"v1": "runs/pragma/base_v1/ckpt_best.pt", "v2b": "runs/pragma/base_v2b/ckpt_best.pt"}
# protocolo B': taxa menor no backbone e avaliação frequente, para não decorar as poucas linhas
BP = {0.01: dict(tag="frac01b", steps=1500, eval=50), 0.1: dict(tag="frac10b", steps=4000, eval=200)}

def item(name, desc, sets):
    return {"name": name, "script": "finetune", "description": desc, "set": sets}

def bprime(model, b):
    pre = "" if model == "v1" else "_v2b"
    out = []
    for frac, p in BP.items():
        for mode in ("full", "lora"):
            lr = "3.0e-5" if mode == "full" else "1.0e-4"
            what = "fine-tuning completo" if mode == "full" else "LoRA (r=8)"
            out.append(item(f"ft_{b}{pre}_{mode}_{p['tag']}",
                f"Rodada 3, protocolo B': {TASK[b]}. {what} do base_{model} com {frac:.0%} das linhas de treino, "
                f"{p['steps']} passos, lr do backbone {lr}, avaliação a cada {p['eval']} passos.",
                [f"init={CKPT[model]}", f"data.dataset={b}", f"mode={mode}", f"data.label_frac={frac}",
                 f"optim.steps={p['steps']}", f"optim.lr={lr}", f"log.eval_every={p['eval']}"]))
    return out

left = [item(f"ft_{b}_v2b_full_20k",
             f"Rodada 3 (PC da esquerda): {TASK[b]}. Fine-tuning completo do base_v2b (pré-treino de 300 mil passos), "
             f"20 mil passos, semente 0. Comparar com ft_{b}_full_20k_s0..s2 (base_v1).",
             [f"init={CKPT['v2b']}", f"data.dataset={b}", "mode=full", "optim.steps=20000", "log.eval_every=2000", "seed=0"])
        for b in TASK]
left += [x for b in TASK for x in bprime("v2b", b)]
right = [x for b in TASK for x in bprime("v1", b)]
for name, items in (("round3_esquerda", left), ("round3_direita", right)):
    (ROOT / "experiments" / f"{name}.yaml").write_text(
        yaml.safe_dump({"name": name, "items": items}, allow_unicode=True, sort_keys=False, width=200))
    print(name, len(items))
