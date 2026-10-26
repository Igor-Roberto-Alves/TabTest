"""Registro comum de execuções (pré-treino, fine-tuning e baselines) para o painel ao vivo.

Cada execução tem um diretório runs/<grupo>/<nome>/ com:
    status.json    nome, grupo, descrição, estado (running | done | failed), passo atual e total,
                   início, última atualização e as métricas mais recentes
    metrics.jsonl  uma linha por registro: {"step": ..., "<métrica>": valor, ...}
    results.json   resultado final (opcional)

src/pragma/dashboard_export.py lê esses arquivos de todos os grupos e gera o que o painel mostra.
Uso:
    run = RunStatus(run_dir, name="base_v1", group="pragma", description="...", total_steps=100000)
    run.update(step, {"val/loss": 1.23})     # a cada registro
    run.finish({"test/auc": 0.78})           # ou run.fail("mensagem")
"""

import json
import time
from pathlib import Path


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class RunStatus:
    def __init__(self, run_dir, name, group, description="", total_steps=None, config=None):
        self.path = Path(run_dir) / "status.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.state = {"name": name, "group": group, "description": description or old.get("description", ""),
                      "status": "running", "started": old.get("started", _now()), "updated": _now(),
                      "step": old.get("step", 0), "total_steps": total_steps, "metrics": old.get("metrics", {}),
                      "config": config or {}}
        self._write()

    def _write(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, default=float))
        tmp.replace(self.path)  # troca atômica: o exportador nunca lê um arquivo pela metade

    def update(self, step, metrics=None):
        self.state["step"] = int(step)
        self.state["metrics"].update({k: float(v) for k, v in (metrics or {}).items()})
        self.state["updated"] = _now()
        self._write()

    def finish(self, metrics=None):
        self.state["metrics"].update({k: float(v) for k, v in (metrics or {}).items()})
        self.state.update(status="done", updated=_now())
        self._write()

    def fail(self, message):
        self.state.update(status="failed", error=str(message)[:500], updated=_now())
        self._write()
