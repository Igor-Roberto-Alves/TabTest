"""Roda os modelos sobre tarefas definidas em specs YAML (tasks/**/*.yaml ou --spec).

Uso:
    python -m src.run --list
    python -m src.run --describe --task adult_income          # checa o spec antes de treinar
    python -m src.run --task adult_income --models xgboost logistic
    python -m src.run --spec meu_dataset.yaml                  # todas as tarefas do spec
    python -m src.run --spec meu_dataset.yaml --task minha_tarefa --seeds 0 1 2
    python -m src.run --all --models xgboost logistic probit ridge --experiment baselines
    python -m src.run --task iris --models tabpfn --tabpfn-version V2
    python -m src.run --task adult_income --models ldm --ldm-checkpoint checkpoints/ldm.pt

Cada execução é um experimento em runs/<experimento>/ com results.csv e um run do
TensorBoard por (tarefa, modelo, seed). Veja tudo junto com:  tensorboard --logdir runs
"""

import argparse
import dataclasses
import importlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import describe, load_task
from src.metrics import evaluate
from src.tasks import discover, load_spec
from src.tracking import DEFAULT_LOGDIR, log_result, open_writer, timestamp

MODELS = {
    "xgboost": "src.models.xgb",
    "logistic": "src.models.logistic",
    "probit": "src.models.probit",
    "ridge": "src.models.ridge",
    "tabpfn": "src.models.tabpfn_model",
    "tabpfn_ft": "src.models.tabpfn_ft_model",
    "tabstar": "src.models.tabstar_model",
    "ldm": "src.models.ldm_model",
}
INSTALL_HINT = {"xgboost": "xgboost", "probit": "statsmodels", "tabpfn": "tabpfn", "tabpfn_ft": "tabpfn",
                "tabstar": "tabstar", "ldm": "torch"}


def parse_model_args(items, parser):
    """['tabpfn.context=20000', 'tabstar.lora_batch=128'] -> {'tabpfn': {'context': 20000}, ...}"""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        model, dot, param = key.partition(".")
        if not (sep and dot and model in MODELS):
            parser.error(f"--model-arg espera modelo.parametro=valor com modelo em {list(MODELS)}; veio {item!r}")
        out.setdefault(model, {})[param] = yaml.safe_load(value)  # 10 -> int, null -> None, V2 -> str
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--spec", nargs="+", default=[], help="YAML(s) de dataset fora de tasks/")
    p.add_argument("--task", nargs="+", default=[], metavar="TASK")
    p.add_argument("--all", action="store_true", help="todas as tarefas registradas")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0], help="uma rodada por seed (split + modelo)")
    p.add_argument("--max-train", type=int, help="limite de linhas de treino para tabpfn/tabpfn_ft/tabstar "
                                                 "(padrão: todas)")
    p.add_argument("--train-size", type=int, help="sorteia N linhas de treino por seed, as mesmas para TODOS os "
                                                  "modelos (o teste fica inteiro); para comparar com poucos dados")
    p.add_argument("--tabpfn-version", help="ex.: V2 (pesos Apache 2.0); vale para tabpfn e tabpfn_ft")
    p.add_argument("--tabstar-time-limit", type=int)
    p.add_argument("--ldm-checkpoint", help="checkpoint de src.ldm.pretrain (sem ele: treina do zero)")
    p.add_argument("--ldm-steps", type=int, help="passos de fine-tuning (0 = zero-shot)")
    p.add_argument("--model-arg", nargs="+", default=[], metavar="MODELO.PARAM=VALOR",
                   help="parâmetro extra de fit_predict, ex.: tabpfn.coverage=2 tabstar.max_epochs=20")
    p.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR, help="raiz do TensorBoard (padrão: runs/)")
    p.add_argument("--experiment", help="subpasta do experimento (padrão: data-hora)")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--out", help="cópia extra do results.csv")
    p.add_argument("--list", action="store_true", help="lista as tarefas registradas")
    p.add_argument("--describe", action="store_true", help="mostra resumo/alertas das tarefas e sai")
    return p, p.parse_args()


def select_tasks(parser, args, registry):
    if args.all:
        return list(registry)
    if args.task:
        unknown = [t for t in args.task if t not in registry]
        if unknown:
            parser.error(f"tarefa(s) desconhecida(s): {', '.join(unknown)} (veja --list)")
        return args.task
    if args.spec:  # sem --task: todas as tarefas dos specs passados
        return [t.name for s in args.spec for t in load_spec(s)]
    parser.error("informe --task, --spec ou --all (veja --list)")


def shrink_train(split, n, seed):
    """Treino com n linhas (estratificado na classificação), igual para todos os modelos."""
    if len(split.X_train) <= n:
        return split
    from sklearn.model_selection import train_test_split
    stratify = None
    if not split.is_regression and np.bincount(split.y_train).min() >= 2:
        stratify = split.y_train
    X, _, y, _ = train_test_split(split.X_train, split.y_train, train_size=n, random_state=seed, stratify=stratify)
    return dataclasses.replace(split, X_train=X, y_train=y)


def call_model(module, split, seed, extra, writer):
    kwargs = dict(extra)
    if "writer" in inspect.signature(module.fit_predict).parameters:
        kwargs["writer"] = writer
    return module.fit_predict(split, seed=seed, **kwargs)


def main():
    parser, args = parse_args()
    registry = discover(args.spec)

    if args.list:
        for t in registry.values():
            print(f"{t.name:28s} {t.kind:10s} {t.dataset}.{t.target:22s} {t.description}")
        return

    task_names = select_tasks(parser, args, registry)
    if args.describe:
        for name in task_names:
            try:
                print(describe(registry[name], seed=args.seeds[0]), end="\n\n")
            except Exception as e:
                print(f"{name}: ERRO: {type(e).__name__}: {e}\n")
        return

    extra = {
        "tabpfn": {"version": args.tabpfn_version},
        "tabpfn_ft": {"version": args.tabpfn_version or "V2"},
        "tabstar": {"time_limit": args.tabstar_time_limit},
        "ldm": {"checkpoint": args.ldm_checkpoint, "steps": args.ldm_steps},
    }
    if args.max_train:
        for name in ("tabpfn", "tabpfn_ft", "tabstar"):
            extra[name]["max_train"] = args.max_train
    for model, params in parse_model_args(args.model_arg, parser).items():
        extra.setdefault(model, {}).update(params)

    experiment = args.experiment or timestamp()
    exp_dir = args.logdir / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(
        {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "tasks": task_names},
        indent=2, ensure_ascii=False))
    print(f"experimento: {exp_dir}")

    rows = []
    for task_name in task_names:
        task = registry[task_name]
        for seed in args.seeds:
            try:
                split = load_task(task, seed=seed)
            except Exception as e:
                print(f"\n== {task.name}: FALHOU ao carregar: {type(e).__name__}: {e}")
                break
            if args.train_size:
                split = shrink_train(split, args.train_size, seed)
            print(f"\n== {task.name} ({task.kind}) seed={seed} train={len(split.X_train)} test={len(split.X_test)}")
            for model_name in args.models:
                module = importlib.import_module(MODELS[model_name])
                if task.kind not in module.KINDS:
                    continue
                writer = None if args.no_tensorboard else open_writer(exp_dir, task.name, model_name, f"seed{seed}")
                start = time.perf_counter()
                try:
                    pred = call_model(module, split, seed, extra.get(model_name, {}), writer)
                except ImportError as e:
                    print(f"  {model_name:10s} pulado: {e} (pip install {INSTALL_HINT.get(model_name, '')})")
                    writer and writer.close()
                    continue
                except Exception as e:  # um modelo quebrado não deve parar o benchmark
                    print(f"  {model_name:10s} FALHOU: {type(e).__name__}: {e}")
                    writer and writer.close()
                    continue
                seconds = time.perf_counter() - start
                scores = evaluate(task.kind, split.y_test, pred)
                print(f"  {model_name:10s} " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()) + f"  ({seconds:.1f}s)")

                hparams = {"experiment": experiment, "task": task.name, "dataset": task.dataset, "kind": task.kind,
                           "model": model_name, "seed": seed, "split": task.split.method,
                           "n_train": len(split.X_train), "n_test": len(split.X_test),
                           "n_features": split.X_train.shape[1],
                           **{f"{model_name}.{k}": v for k, v in extra.get(model_name, {}).items() if v is not None}}
                metrics = {f"test/{k}": v for k, v in scores.items()} | {"time/seconds": seconds}
                info = (f"**{task.name}** ({task.kind}) — {task.description}\n\n"
                        f"spec: `{task.source}` · alvo: `{task.target}`" + (f" = `{task.expr}`" if task.expr else "")
                        + f" · split: {task.split}")
                log_result(writer, hparams, metrics, info)
                writer and writer.close()

                rows.append({"experiment": experiment, "task": task.name, "model": model_name, "seed": seed,
                             "n_train": len(split.X_train), "n_test": len(split.X_test), "seconds": seconds, **scores})
                pd.DataFrame(rows).to_csv(exp_dir / "results.csv", index=False)  # salva a cada modelo

    if rows:
        print(f"\nResultados em {exp_dir / 'results.csv'}")
        if args.out:
            pd.DataFrame(rows).to_csv(args.out, index=False)
        if not args.no_tensorboard:
            print(f"TensorBoard: tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
