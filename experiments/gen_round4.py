# Rodada 4: fine-tuning de pré-treinos de ablação e do sweep (20 mil passos), para entrarem nos gráficos de teste.
# python experiments/gen_round4.py .   (regera experiments/round4.yaml; roda no PC onde estão os checkpoints)
"""Gera experiments/round4.yaml: LoRA com 10% dos rótulos (protocolo B') e depois fine-tuning completo com 100%."""
import sys, yaml
from pathlib import Path
ROOT = Path(sys.argv[1])
TASK = {"home_credit": "Home Credit (default, AUC)", "amex": "AmEx (default, métrica AmEx)"}
# abl_ref é a referência (probe médio 0.667); trial_03 é o melhor do sweep pela média dos probes; as três ablações mudam uma coisa cada
PRE = {"abl_ref": ("abl_ref", "referência: config padrão nas três bases"),
       "abl_no_event": ("abl_no_event", "sem a máscara de evento inteiro, p_event=0"),
       "abl_no_type": ("abl_no_type", "sem a máscara de tipo semântico, p_type=0"),
       "abl_minilm": ("abl_minilm", "nomes de coluna e categorias via MiniLM congelado, em vez de hash de n-gramas"),
       "trial_03": ("sweep_pretrain/trial_03", "melhor trial da busca: lr 5.4e-4, d_model 256, 4 camadas de histórico, p_event 0.19, p_type 0.2")}

def item(tag, b, pre, what, desc, sets):
    return {"name": f"ft_{b}_{tag}_{what}", "script": "finetune",
            "description": f"Rodada 4: {TASK[b]}. {desc} do pré-treino {PRE[pre][0]} ({PRE[pre][1]}, 20 mil passos). "
                           f"Comparar com abl_ref e com o probe.",
            "set": [f"init=runs/pragma/{PRE[pre][0]}/ckpt_best.pt", f"data.dataset={b}"] + sets}

items = []
for b in TASK:
    for pre in PRE:
        items.append(item(pre, b, pre, "lora_frac10b", "LoRA (r=8) com 10% das linhas de treino, 4000 passos, lr 1.0e-4, avaliação a cada 200 passos",
                          ["mode=lora", "data.label_frac=0.1", "optim.steps=4000", "optim.lr=1.0e-4", "log.eval_every=200"]))
for b in TASK:
    for pre in PRE:
        items.append(item(pre, b, pre, "full_20k", "Fine-tuning completo com todas as linhas de treino, 20 mil passos, semente 0",
                          ["mode=full", "optim.steps=20000", "log.eval_every=2000", "seed=0"]))
(ROOT / "experiments" / "round4.yaml").write_text(yaml.safe_dump({"name": "round4", "items": items}, allow_unicode=True, sort_keys=False, width=200))
print("round4", len(items))
