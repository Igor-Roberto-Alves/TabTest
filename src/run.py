"""Roda os modelos nas tarefas sugeridas e imprime as métricas de teste.

Uso:
    python -m src.run --list
    python -m src.run --list-models
    python -m src.run --task adult_income
    python -m src.run --task adult_income california_price --models xgboost tabpfn
    python -m src.run --task iris --models tabpfn --tabpfn-version V2
    python -m src.run --all --models xgboost logistic probit ridge --out results.csv
    python -m src.run --task adult_income --models ldm --ldm-checkpoint checkpoints/ldm.pt
"""

import argparse
import importlib
import time

import pandas as pd
import yaml

from src.data import load_task
from src.metrics import evaluate
from src.models.registry import MODELS as REGISTRY
from src.tasks import TASKS

MODELS = {n: m.module for n, m in REGISTRY.items()}
INSTALL_HINT = {n: m.install for n, m in REGISTRY.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--task", nargs="+", default=[], choices=TASKS, metavar="TASK")
    parser.add_argument("--all", action="store_true", help="todas as tarefas")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, help="limite de treino para tabpfn/tabstar")
    parser.add_argument("--tabpfn-version", help="ex.: V2 (pesos Apache 2.0)")
    parser.add_argument("--tabstar-time-limit", type=int)
    parser.add_argument("--ldm-checkpoint", help="checkpoint de src.ldm.pretrain (sem ele: treina do zero)")
    parser.add_argument("--ldm-steps", type=int, help="passos de fine-tuning (0 = zero-shot)")
    parser.add_argument("--model-arg", nargs="+", default=[], metavar="MODELO.CHAVE=VALOR",
                        help="argumento extra de um modelo (valor lido como YAML), ex.: xgboost.max_depth=8")
    parser.add_argument("--out", help="salva os resultados em CSV")
    parser.add_argument("--list-models", action="store_true", help="lista os modelos registrados")
    parser.add_argument("--list", action="store_true", help="lista as tarefas sugeridas")
    args = parser.parse_args()

    if args.list_models:
        for m in REGISTRY.values():
            print(f"{m.name:10s} {m.data:8s} {','.join(sorted(m.kinds)):28s} {m.description}")
        return

    if args.list:
        for t in TASKS.values():
            print(f"{t.name:22s} {t.kind:10s} {t.dataset}.{t.target:15s} {t.description}")
        return

    task_names = list(TASKS) if args.all else args.task
    if not task_names:
        parser.error("informe --task ou --all (veja --list)")

    extra = {
        "tabpfn": {"version": args.tabpfn_version},
        "tabstar": {"time_limit": args.tabstar_time_limit},
        "ldm": {"checkpoint": args.ldm_checkpoint, "steps": args.ldm_steps},
    }
    if args.max_train:
        for name in ("tabpfn", "tabstar"):
            extra[name]["max_train"] = args.max_train

    for item in args.model_arg:
        key, _, raw = item.partition("=")
        model, _, option = key.partition(".")
        if model not in REGISTRY or not option or not _:
            parser.error(f"--model-arg inválido: {item} (use MODELO.CHAVE=VALOR)")
        extra.setdefault(model, {})[option] = yaml.safe_load(raw)

    rows = []
    for task_name in task_names:
        task = TASKS[task_name]
        split = load_task(task, seed=args.seed)
        print(f"\n== {task.name} ({task.kind}) train={len(split.X_train)} test={len(split.X_test)}")
        for model_name in args.models:
            module = importlib.import_module(MODELS[model_name])
            if task.kind not in module.KINDS:
                continue
            start = time.perf_counter()
            try:
                pred = module.fit_predict(split, seed=args.seed, **extra.get(model_name, {}))
            except ImportError as e:
                print(f"  {model_name:10s} pulado: {e} (pip install {INSTALL_HINT.get(model_name, '')})")
                continue
            except Exception as e:  # um modelo quebrado não deve parar o benchmark
                print(f"  {model_name:10s} FALHOU: {type(e).__name__}: {e}")
                continue
            scores = evaluate(task.kind, split.y_test, pred)
            seconds = time.perf_counter() - start
            print(f"  {model_name:10s} " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()) + f"  ({seconds:.1f}s)")
            rows.append({"task": task.name, "model": model_name, "seconds": seconds, **scores})

    if args.out and rows:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nResultados salvos em {args.out}")


if __name__ == "__main__":
    main()
