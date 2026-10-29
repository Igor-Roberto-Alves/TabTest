"""Junta o estado de todas as execuções num único JSON para o painel ao vivo (artefato).

Lê runs/<grupo>/**/status.json e metrics.jsonl (ver runlog.py) dos grupos pragma, pragma_ft e
baselines, mais as filas em runs/queues/*.json, e escreve um documento com:
    experiments  um item por execução: descrição, estado, progresso, métricas mais recentes e
                 séries reduzidas (até MAX_POINTS pontos por métrica) para os gráficos
    queue        itens de fila ainda não iniciados
    gpu          uso e memória da GPU agora

Execuções cujo nome começa com "_" (testes de fumaça) ficam de fora.
Uso: python -m src.pragma.dashboard_export --out <arquivo.json> [--rows] [--machine direita|esquerda]

Cada máquina exporta só as próprias execuções; --machine (padrão: pelo hostname, ver MACHINES) marca as
linhas para o painel dizer em qual GPU cada execução rodou. A direita alimenta o dataset "monitor" do
painel e a esquerda o "monitor_esquerda".

Com --rows, a saída é uma lista plana de linhas com uma coluna "kind" (exp, series, queue,
system): é o arquivo de dados do painel, que separa os tipos com cálculos na própria página.
"""

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ["pragma", "pragma_ft", "baselines"]
MACHINES = {"c1-03-04-16116": "direita", "c1-02-03-16117": "esquerda"}
MAX_POINTS = 120
# métricas desenhadas por grupo (prefixos); o resto fica só no TensorBoard
SERIES = {
    "pragma": ["train/loss", "val/loss", "train/ce_cell", "train/ce_event", "train/ce_type", "train/ce_profile",
               "probe_", "perf/steps_per_sec"],
    "pragma_ft": ["train/loss", "val/auc", "val/amex", "val/ks", "val/brier"],
    "baselines": ["train/", "val/"],
}
LATEST = ("val/", "probe_", "test/", "train/loss", "best_val_loss", "perf/steps_per_sec", "perf/gpu_mem_gb")


def wanted(group, key):
    if key.startswith("probe_"):
        return key.endswith(("/auc", "/amex"))
    return any(key.startswith(p) for p in SERIES.get(group, [])) and key.count("/") <= 1


def keep_latest(key):
    if key.startswith("probe_"):
        return key.endswith(("/auc", "/amex"))
    return key.startswith(LATEST) and key.count("/") <= 1


def downsample(points):
    if len(points) <= MAX_POINTS:
        return points
    stride = len(points) / MAX_POINTS
    out = [points[int(i * stride)] for i in range(MAX_POINTS - 1)]
    return out + [points[-1]]


def read_series(run_dir, group):
    path = run_dir / "metrics.jsonl"
    series = {}
    if not path.exists():
        return series
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # linha sendo escrita
        step = row.get("step")
        for k, v in row.items():
            if k != "step" and step is not None and wanted(group, k) and v == v:
                series.setdefault(k, []).append((step, round(float(v), 5)))
    # arrays aninhados não são aceitos pelo banco: cada série vira {"s": passos, "v": valores}
    return {k: {"s": [p[0] for p in downsample(pts)], "v": [p[1] for p in downsample(pts)]}
            for k, pts in series.items()}


def gpu():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                       "--format=csv,noheader,nounits"], text=True, timeout=10)
        util, used, total = [float(x) for x in out.strip().split("\n")[0].split(",")]
        return {"util": util, "mem_used_gb": round(used / 1024, 1), "mem_total_gb": round(total / 1024, 1)}
    except Exception:
        return None


def collect():
    experiments = []
    for group in GROUPS:
        base = ROOT / "runs" / group
        for status_path in sorted(base.glob("**/status.json")) if base.exists() else []:
            run_dir = status_path.parent
            run_id = str(run_dir.relative_to(base))
            if any(part.startswith("_") for part in Path(run_id).parts):
                continue
            try:
                st = json.loads(status_path.read_text())
            except json.JSONDecodeError:
                continue
            latest = {k: round(v, 5) for k, v in st.get("metrics", {}).items() if keep_latest(k)}
            best_step = None
            if (run_dir / "results.json").exists():
                try:
                    best_step = json.loads((run_dir / "results.json").read_text()).get("best_step")
                except json.JSONDecodeError:
                    pass
            experiments.append({
                "id": f"{group}/{run_id}", "name": st.get("name", run_id), "group": group,
                "description": st.get("description", ""), "status": st.get("status", "?"),
                "step": st.get("step", 0), "total_steps": st.get("total_steps"), "started": st.get("started"),
                "updated": st.get("updated"), "error": st.get("error"), "config": st.get("config", {}), "model": st.get("model"),
                "metrics": latest, "best_step": best_step, "series": read_series(run_dir, group)})
    started = {e["id"] for e in experiments}
    queue = []
    for qpath in sorted((ROOT / "runs" / "queues").glob("*.json")) if (ROOT / "runs" / "queues").exists() else []:
        q = json.loads(qpath.read_text())
        for it in q["items"]:
            if f"{it['group']}/{it['name']}" not in started and it["state"] in ("pending", "running"):
                queue.append({"queue": q["name"], "name": it["name"], "group": it["group"],
                              "description": it["description"]})
    return {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpu": gpu(), "experiments": experiments, "queue": queue}


GROUP_LABEL = {"pragma": "pré-treino", "pragma_ft": "tarefa", "baselines": "XGBoost"}
GPU_GROUPS = {"pragma", "pragma_ft"}  # o XGBoost roda na CPU


def kind_of(e):
    """Rótulo do tipo e se usa GPU: em baselines, tabstar_* é TabSTAR (GPU); o resto é XGBoost (CPU)."""
    if e["group"] == "baselines" and e["name"].startswith("tabstar_"):
        return "TabSTAR", True
    return GROUP_LABEL.get(e["group"], e["group"]), e["group"] in GPU_GROUPS
PROBE_LABEL = {"home_credit": "HC", "amex": "AmEx", "mbd_mini": "MBD"}


def model_of(e):
    """Modelo da execução: o campo "model" do status.json; sem ele, deduz do grupo (execuções antigas)."""
    if e.get("model"):
        return e["model"]
    if e["group"] == "baselines":
        return "tabstar" if e["name"].startswith("tabstar_") else "xgboost"
    return "pragma"


def model_rows():
    """Uma linha por modelo conhecido, para o seletor do painel: PRAGMA (eventos) e os do registro."""
    from src.models.registry import MODELS
    rows = [{"kind": "model", "name": "pragma", "label": "PRAGMA", "data": "events",
             "capabilities": ["pretrain", "finetune", "explain_mask"]}]
    rows += [{"kind": "model", "name": m.name, "label": m.label, "data": m.data,
              "capabilities": sorted(m.capabilities)} for m in MODELS.values() if m.name != "pragma"]
    return rows


def primary(e):
    """Métrica principal de uma execução: (rótulo, valor, maior_é_melhor)."""
    m = e["metrics"]
    if e["group"] == "pragma":
        for k in ("val/loss", "train/loss"):
            if k in m:
                return k, m[k], False
    else:
        for k in ("test/amex", "test/auc", "val/amex", "val/auc"):
            if k in m:
                return k, m[k], True
    return None, None, None


def summary(e):
    m, parts = e["metrics"], []
    for k in ("train/loss", "val/loss"):
        if k in m:
            parts.append(f"{k} {m[k]:.3f}")
    for k, v in m.items():
        if k.startswith("probe_"):
            base, metric = k[len("probe_"):].split("/")
            parts.append(f"probe {PROBE_LABEL.get(base, base)} {metric.upper()} {v:.3f}")
    for k in ("val/auc", "val/amex", "val/ks", "test/auc", "test/amex", "test/ks", "test/brier"):
        if k in m:
            parts.append(f"{k} {m[k]:.3f}")
    return " · ".join(parts)


def to_rows(doc, machine):
    now = time.time()
    rows = []
    for e in doc["experiments"]:
        label, value, higher = primary(e)
        total = e.get("total_steps") or 0
        progress = min(1.0, e["step"] / total) if total else (1.0 if e["status"] == "done" else None)
        eta = None
        sps = e["metrics"].get("perf/steps_per_sec")
        if e["status"] == "running" and total and sps:
            eta = round((total - e["step"]) / sps / 60, 1)
        updated = time.mktime(time.strptime(e["updated"], "%Y-%m-%dT%H:%M:%S")) if e.get("updated") else now
        started = time.mktime(time.strptime(e["started"], "%Y-%m-%dT%H:%M:%S")) if e.get("started") else None
        # duração de parede: do início à última atualização (ou até agora, se ainda roda)
        duration = None if started is None else round(((now if e["status"] == "running" else updated) - started) / 60, 1)
        group_label, on_gpu = kind_of(e)
        gpu_hours = None if duration is None else round(duration / 60, 2) if on_gpu else 0.0
        mem = e["metrics"].get("perf/gpu_mem_gb")
        rows.append({"kind": "exp", "id": e["id"], "name": e["name"], "group": e["group"],
                     "group_label": group_label, "status": e["status"],
                     "stale": e["status"] == "running" and now - updated > 20 * 60,
                     "description": e["description"], "step": e["step"], "total_steps": total or None,
                     "progress": progress, "eta_min": eta, "started": e.get("started"), "updated": e.get("updated"),
                     "primary_label": label, "primary_value": value, "higher_is_better": higher,
                     "summary": summary(e), "error": e.get("error"), "duration_min": duration,
                     "gpu_hours": gpu_hours, "gpu_mem_gb": None if mem is None else round(mem, 1),
                     "best_step": e.get("best_step"), "machine": machine, "model": model_of(e),
                     # nota final por métrica (validação no checkpoint escolhido e teste), para os cards de validação
                     **{f"{split}_{k}": e["metrics"].get(f"{split}/{k}")
                        for split in ("val", "test") for k in ("auc", "amex", "ks", "brier")}})
        for metric, ser in e["series"].items():
            rows += [{"kind": "series", "run": e["id"], "metric": metric, "step": st, "value": v}
                     for st, v in zip(ser["s"], ser["v"])]
    for i, q in enumerate(doc["queue"]):
        rows.append({"kind": "queue", "name": q["name"], "group": q["group"],
                     "group_label": GROUP_LABEL.get(q["group"], q["group"]), "description": q["description"],
                     "position": i + 1, "queue": q["queue"], "machine": machine})
    rows += model_rows()
    g = doc["gpu"] or {}
    rows.append({"kind": "system", "machine": machine, "updated": doc["updated"], "gpu_util": g.get("util"),
                 "gpu_mem_used": g.get("mem_used_gb"), "gpu_mem_total": g.get("mem_total_gb")})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", action="store_true", help="lista plana de linhas (arquivo de dados do painel)")
    p.add_argument("--machine", choices=sorted(set(MACHINES.values())), default=MACHINES.get(socket.gethostname()),
                   help="em qual PC as execuções rodam (padrão: pelo hostname)")
    args = p.parse_args()
    if args.rows and not args.machine:
        p.error(f"hostname {socket.gethostname()} fora de MACHINES: passe --machine")
    doc = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_rows(doc, args.machine) if args.rows else doc, ensure_ascii=False, separators=(",", ":"))
    args.out.write_text(text)
    print(f"{len(doc['experiments'])} execuções, {len(doc['queue'])} na fila, {len(text) / 1024:.0f} KB -> {args.out}")


if __name__ == "__main__":
    main()
