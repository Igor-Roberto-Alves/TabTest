"""Busca de hiperparâmetros: gera execuções de train.py ou finetune.py e roda uma após a outra.

Uso:
    python -m src.pragma.sweep configs/pragma/sweep_pretrain.yaml --dry-run   # só lista os comandos
    python -m src.pragma.sweep configs/pragma/sweep_pretrain.yaml
    tensorboard --logdir runs/pragma        # aba HParams compara as execuções

Spec (YAML):
    name: lr_mask                 # execuções em <out_dir>/<name>/trial_XX
    script: train                 # train | finetune
    base: configs/pragma/pretrain.yaml
    method: random                # grid (produto cartesiano) | random (trials sorteios)
    trials: 8
    seed: 0
    fixed: [optim.steps=20000]    # sobrescritas iguais em todas as execuções
    params:
      optim.lr: {dist: loguniform, low: 1.0e-4, high: 1.0e-3}
      model.d_model: [128, 256]   # lista = escolha entre valores
      mask.p_event: {dist: uniform, low: 0.05, high: 0.3}

Uma execução que falha não para as outras. --start N pula as N primeiras (para retomar).
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from .config import load_config

ROOT = Path(__file__).resolve().parents[2]


def sample_value(spec, rng):
    if isinstance(spec, list):
        return spec[rng.integers(len(spec))]
    dist = spec["dist"]
    if dist == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    if dist == "loguniform":
        return float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
    if dist == "int":
        return int(rng.integers(spec["low"], spec["high"] + 1))
    raise ValueError(f"distribuição desconhecida: {dist}")


def trials(spec):
    params = spec.get("params", {})
    if spec.get("method", "random") == "grid":
        if any(not isinstance(v, list) for v in params.values()):
            raise ValueError("grid só aceita listas de valores")
        keys = list(params)
        return [dict(zip(keys, combo)) for combo in itertools.product(*params.values())]
    rng = np.random.default_rng(spec.get("seed", 0))
    return [{k: sample_value(v, rng) for k, v in params.items()} for _ in range(spec["trials"])]


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args()
    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    out_dir = load_config(ROOT / spec["base"])["out_dir"]
    commands = []
    for i, values in enumerate(trials(spec)):
        sets = list(spec.get("fixed", [])) + [f"{k}={fmt(v)}" for k, v in values.items()]
        cmd = [sys.executable, "-m", f"src.pragma.{spec['script']}", "--config", str(ROOT / spec["base"]),
               "--run-name", f"{spec['name']}/trial_{i:02d}", "--set", *sets]
        commands.append(cmd)
    print(f"{len(commands)} execuções -> {ROOT / out_dir / spec['name']}")
    for i, cmd in enumerate(commands):
        shown = " ".join(c if " " not in c else f"'{c}'" for c in cmd[1:])
        if args.dry_run or i < args.start:
            print(f"[{i:02d}] {'(pulada) ' if i < args.start and not args.dry_run else ''}python {shown}")
            continue
        print(f"[{i:02d}] python {shown}", flush=True)
        code = subprocess.call(cmd, cwd=ROOT)
        if code:
            print(f"[{i:02d}] falhou com código {code}", flush=True)


if __name__ == "__main__":
    main()
