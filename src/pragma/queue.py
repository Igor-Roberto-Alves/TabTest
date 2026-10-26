"""Fila de experimentos: roda em sequência os itens de um YAML, cada um com nome e descrição.

Uso:
    python -m src.pragma.queue experiments/round1.yaml            # roda (pula itens já concluídos)
    python -m src.pragma.queue experiments/round1.yaml --dry-run

Formato:
    name: round1
    items:
      - name: base_v1                       # vira --run-name
        script: train                       # train | finetune (src/pragma/<script>.py)
        description: "o que esta execução testa"
        set: [optim.steps=100000]           # sobrescritas da config (--set)
        config: configs/pragma/pretrain.yaml   # opcional; padrão do script

O estado da fila vai para runs/queues/<name>.json (o painel mostra os itens pendentes).
Um item que falha não para a fila. Um item cujo status.json diz "done" é pulado.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GROUP = {"train": "pragma", "finetune": "pragma_ft"}


def run_status(script, name):
    path = ROOT / "runs" / GROUP[script] / name / "status.json"
    return json.loads(path.read_text())["status"] if path.exists() else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    spec = yaml.safe_load(args.spec.read_text())
    state_path = ROOT / "runs" / "queues" / f"{spec['name']}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    items = [{**it, "group": GROUP[it["script"]], "state": "pending"} for it in spec["items"]]

    def save():
        state_path.write_text(json.dumps({"name": spec["name"], "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "items": [{k: it[k] for k in ("name", "group", "script", "description", "state")}
                                                    for it in items]}, ensure_ascii=False))

    for it in items:
        if run_status(it["script"], it["name"]) == "done":
            it["state"] = "done"
    save()
    for it in items:
        if it["state"] == "done":
            print(f"[fila] {it['name']}: já concluído, pulando")
            continue
        cmd = [sys.executable, "-m", f"src.pragma.{it['script']}", "--run-name", it["name"],
               "--description", it["description"]]
        if it.get("config"):
            cmd += ["--config", str(ROOT / it["config"])]
        if it.get("set"):
            cmd += ["--set", *it["set"]]
        print(f"[fila] {it['name']}: {' '.join(cmd[1:])}", flush=True)
        if args.dry_run:
            continue
        it["state"] = "running"
        save()
        code = subprocess.call(cmd, cwd=ROOT)
        it["state"] = "done" if code == 0 else "failed"
        save()
        print(f"[fila] {it['name']}: {it['state']} (código {code})", flush=True)


if __name__ == "__main__":
    main()
