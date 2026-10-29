"""TabSTAR nas tarefas P1, sobre a tabela crua e, opcionalmente, com o embedding do modelo fundacional.

Uso:
    python -m src.baselines.tabstar amex                        # só as colunas cruas (as mesmas do XGB --raw)
    python -m src.baselines.tabstar amex --emb base_v2b         # cruas + embedding [CLS] congelado
    python -m src.baselines.tabstar home_credit --max-train 5000 --run-name _teste_hc   # teste rápido
    python -m src.baselines.tabstar amex --max-train 0          # todas as linhas de treino (execução ..._full)

Protocolo igual ao de src/baselines/xgb.py: linhas de rótulo e split do store, subamostra do treino
com a mesma semente (aqui limitada por --max-train, porque o TabSTAR faz fine-tuning LoRA na GPU), early
stopping numa amostra da validação (--val-max) e avaliação na validação e no teste inteiros. Métricas de
src/pragma/evaluate.py.

A tabela crua é a do XGB --raw: no Home Credit, as 120 colunas do pedido (application_train); na AmEx,
as 188 do último extrato. Os nomes vão sem o prefixo APP_ e sem o sufixo _last, porque o TabSTAR lê o
nome da coluna como texto. As categóricas chegam como códigos numéricos, como no XGBoost.

O TabSTAR aceita no máximo 200 colunas: acima disso ele mesmo fica com as numéricas de maior estatística
F e comprime o resto por PCA (tabstar/.../feature_selection.py). Com o embedding (256 dimensões) é isso
que acontece, e a seleção sai no log.

Cada execução grava em runs/baselines/<run>/: status.json, metrics.jsonl e results.json.
Requer Python 3.10–3.12 segundo a documentação do TabSTAR; aqui roda no .venv (3.14) com tabstar 1.1.16.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.pragma.evaluate import primary_metric, task_metrics
from src.pragma.runlog import RunStatus

from .common import ROOT, Labels, Recorder, log, subsample
from .xgb import RAW_TEXT, align, load_embedding, load_features, raw_columns

BASES = ["home_credit", "amex"]


def tidy(name):
    return name.removeprefix("APP_").removesuffix("_last")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base", choices=BASES)
    p.add_argument("--emb", default=None, help="tag do embedding (src/pragma/export_embeddings.py) a juntar")
    p.add_argument("--max-train", type=int, default=20000,
                   help="linhas de treino (subamostra com a semente); 0 = todas as linhas de treino")
    p.add_argument("--val-max", type=int, default=5000, help="linhas de validação usadas no early stopping")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lora-batch", type=int, default=64,
                   help="lote por passo do LoRA; o lote global fica em 128 por acúmulo (32 cabe na AmEx com embedding)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    max_train = args.max_train or None
    n_tag = "_full" if max_train is None else f"_n{max_train // 1000}k" if max_train % 1000 == 0 else f"_n{max_train}"
    run_name = args.run_name or f"tabstar_{args.base}_raw" + (f"_emb_{args.emb}" if args.emb else "") + n_tag
    run_dir = ROOT / "runs" / "baselines" / run_name
    labels = Labels(args.base)
    task, metric = labels.task, primary_metric(labels.task)
    features = RAW_TEXT[args.base]
    if args.emb:
        features += f"; mais o embedding [CLS] congelado do checkpoint {args.emb} (256 dimensões)"
    description = (f"TabSTAR (fine-tuning LoRA na GPU) em {args.base}, tarefa {task['name']}, com "
                   f"{'todas as' if max_train is None else max_train} linhas de treino. Features: {features}. Early stopping em {args.val_max} linhas da validação.")
    run = RunStatus(run_dir, name=run_name, group="baselines", model="tabstar", description=description, total_steps=args.max_epochs,
                    config={"base": args.base, "emb": args.emb, "max_train": max_train, "val_max": args.val_max,
                            "max_epochs": args.max_epochs, "patience": args.patience, "lora_batch": args.lora_batch,
                            "seed": args.seed, "raw": True})
    record = Recorder(run_dir, run)
    t_start = time.time()
    try:
        from tabstar.tabstar_model import TabSTARClassifier

        index, names, A, _ = load_features(args.base)
        keep = raw_columns(args.base, names)
        A, names = A[:, keep], [tidy(names[j]) for j in keep]
        pos = align(index, labels)
        found = pos >= 0
        E, emb_names = load_embedding(args.emb, args.base, len(labels.y)) if args.emb else (None, [])
        emb_names = [f"client embedding {j}" for j in range(len(emb_names))]
        cols = names + emb_names
        log(f"{args.base}: {len(cols)} colunas ({len(names)} cruas + {len(emb_names)} do embedding), "
            f"{found.sum()} de {len(found)} linhas de rótulo com features")

        rows = {s: labels.rows(s)[found[labels.rows(s)]] for s in ("train", "val", "test")}
        rows["train"] = subsample(rows["train"], max_train, args.seed)
        rows["es"] = subsample(rows["val"], args.val_max, args.seed)
        log(f"linhas: treino {len(rows['train'])}, early stopping {len(rows['es'])}, "
            f"validação {len(rows['val'])}, teste {len(rows['test'])}")

        def frame(r):
            parts = [A[pos[r]]] + ([] if E is None else [np.asarray(E[r], dtype=np.float32)])
            return pd.DataFrame(np.hstack(parts) if len(parts) > 1 else parts[0], columns=cols)

        y = labels.y[:, 0].astype(int)
        target = task["labels"][0]
        # sem device, o TabSTAR recusa uma GPU que já tenha outro processo
        model = TabSTARClassifier(random_state=args.seed, max_epochs=args.max_epochs, patience=args.patience,
                                  lora_batch=args.lora_batch, global_batch=128, verbose=True, device="cuda")
        t_train = time.time()
        model.fit(frame(rows["train"]), pd.Series(y[rows["train"]], name=target),
                  frame(rows["es"]), pd.Series(y[rows["es"]], name=target))
        train_s = time.time() - t_train
        log(f"treino: {train_s:.0f}s")

        def proba(r):
            out = []
            for i in range(0, len(r), 20000):
                P = np.asarray(model.predict_proba(frame(r[i:i + 20000])))
                out.append(P[:, 1] if P.ndim == 2 else P)
            return np.concatenate(out)

        P = {s: proba(rows[s]) for s in ("val", "test")}
        val = task_metrics(task, labels.y[rows["val"]], P["val"][:, None])
        test = task_metrics(task, labels.y[rows["test"]], P["test"][:, None])
        total_s = time.time() - t_start
        log(f"val: {metric} {val[metric]:.4f} auc {val['auc']:.4f} | test: {metric} {test[metric]:.4f} "
            f"auc {test['auc']:.4f} | {total_s:.0f}s")
        results = {"run_name": run_name, "base": args.base, "task": task, "label_frac": None,
                   "max_train": max_train, "primary_metric": metric, "n_features": len(cols), "emb": args.emb,
                   "n_rows": {s: int(len(r)) for s, r in rows.items()}, "val": val, "test": test,
                   "seconds": {"train": train_s, "total": total_s}}
        (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
        record(args.max_epochs, {f"val/{m}": v for m, v in val.items()})
        run.finish({**{f"val/{m}": v for m, v in val.items()}, **{f"test/{m}": v for m, v in test.items()},
                    "time/total_s": total_s})
    except Exception as e:
        run.fail(str(e))
        raise


if __name__ == "__main__":
    main()
