"""Baselines XGBoost nas tarefas P1, com o mesmo protocolo do fine-tuning do modelo fundacional.

Uso:
    python -m src.baselines.xgb home_credit
    python -m src.baselines.xgb amex --label-frac 0.1            # curva de rótulos (semente 0)
    python -m src.baselines.xgb mbd_mini --run-name xgb_mbd_mini
    python -m src.baselines.xgb amex --max-rows 200000 --run-name _teste_amex   # teste rápido
    python -m src.baselines.xgb amex --emb base_v1               # + embedding do modelo fundacional
    python -m src.baselines.xgb amex --emb base_v1 --emb-only    # só o embedding
    python -m src.baselines.xgb amex --raw                       # só colunas cruas, sem engenharia

Protocolo: split por cliente e linhas de rótulo de data/banking/prepared/<base>/; treino no split
de treino (ou numa fração dele, com a mesma subamostra de src/pragma/finetune.py), early stopping
na AUC de validação, uma grade pequena (configs/baselines/xgb.yaml) escolhida só na validação, e o
teste avaliado uma vez com a combinação vencedora. No MBD há um modelo por rótulo, e a combinação
é escolhida por rótulo. Métricas de src/pragma/evaluate.py.

Cada execução grava em runs/baselines/<run>/: status.json (RunStatus), metrics.jsonl,
results.json e feature_importance.csv. Execuções de teste começam com "_" e ficam fora do painel.

--emb <tag> junta às features as colunas emb_000..emb_<d-1> de data/banking/features/emb_<tag>_<base>.npy
(src/pragma/export_embeddings.py), uma linha por linha de rótulo do store.

--raw fica só com colunas cruas, sem agregação nem razões: no Home Credit, as 120 colunas do pedido
(application_train), sem o histórico; na AmEx, as 188 colunas do último extrato. Mede quanto a
engenharia de features vale. O MBD não tem versão crua: sem agregar eventos não sobra coluna.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.pragma.config import load_config
from src.pragma.evaluate import amex_metric, binary_metrics, primary_metric, task_metrics
from src.pragma.runlog import RunStatus

from .common import FEATURES, ROOT, Labels, Recorder, clean_names, fit, importance, log, predict, subsample

BUILDERS = {"home_credit": "home_credit", "amex": "amex", "mbd_mini": "mbd"}
FEATURE_TEXT = {
    "home_credit": "application_train + agregações por cliente (count, mean, min, max, sum e categorias) de bureau, "
                   "bureau_balance, previous_application, POS_CASH, credit_card e installments, mais razões clássicas",
    "amex": "agregações por customer_ID dos extratos (last, mean, std, min, max e last - mean nas numéricas; last, "
            "nunique e count nas 11 categóricas; nº de extratos)",
    "mbd_mini": "agregados mensais de transações, geolocalização e diálogos nas janelas antes do mês do rótulo (mês "
                "anterior, 3 meses, histórico todo, meses desde o último evento); um modelo por produto",
}


def read_cache(path):
    """Lê o parquet direto para um array float32, sem passar por um DataFrame: na AmEx (459 mil x
    1096) o caminho pelo pandas fazia várias cópias de 2 GB e o pico passava de 15 GB."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    keys = [c for c in ("id", "cutoff") if c in table.column_names]
    cols = [c for c in table.column_names if c not in keys and not c.startswith("__index")]
    A = np.empty((table.num_rows, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        A[:, j] = table.column(c).to_numpy().astype(np.float32, copy=False)
    arrays = [table.column(k).to_numpy(zero_copy_only=False) for k in keys]
    index = pd.MultiIndex.from_arrays(arrays, names=keys) if len(keys) == 2 else pd.Index(arrays[0], name=keys[0])
    return index, cols, A


def load_features(base, max_rows=None, rebuild=False):
    """Features em cache em data/banking/features/<base>.parquet (ou <base>_dev com --max-rows).
    Retorna (índice das chaves, nomes, A (U, F) float32, segundos de feature engineering)."""
    path = FEATURES / f"{base}{'_dev' if max_rows else ''}.parquet"
    fe_s = 0.0
    if path.exists() and not rebuild:
        log(f"features em cache: {path}")
    else:
        import importlib

        module = importlib.import_module(f"src.baselines.{BUILDERS[base]}")
        t0 = time.time()
        feats = module.build(max_rows=max_rows)
        FEATURES.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(path)
        fe_s = time.time() - t0
        log(f"features: {feats.shape} -> {path} ({fe_s:.0f}s)")
        del feats
    return (*read_cache(path), fe_s)


HC_DERIVED = {"APP_CREDIT_INCOME", "APP_ANNUITY_INCOME", "APP_CREDIT_TERM", "APP_GOODS_CREDIT", "APP_EMPLOYED_BIRTH",
              "APP_INCOME_PER_PERSON", "APP_EXT_MEAN", "APP_EXT_STD", "APP_EXT_PROD"}
RAW_TEXT = {"home_credit": "só as 120 colunas cruas do pedido (application_train), sem histórico nem razões",
            "amex": "só as 188 colunas cruas do último extrato, sem agregação"}


def raw_columns(base, names):
    if base == "home_credit":
        return [j for j, c in enumerate(names) if c.startswith("APP_") and c not in HC_DERIVED]
    if base == "amex":
        return [j for j, c in enumerate(names) if c.endswith("_last")]
    raise ValueError(f"{base} não tem versão crua (ver --raw)")


def load_embedding(tag, base, n_rows):
    E = np.load(FEATURES / f"emb_{tag}_{base}.npy", mmap_mode="r")
    if len(E) != n_rows:
        raise ValueError(f"embedding com {len(E)} linhas, o store tem {n_rows} linhas de rótulo")
    return E, [f"emb_{j:03d}" for j in range(E.shape[1])]


def align(index, labels):
    """pos (R,): linha de A de cada linha de rótulo do store, pelo id (e corte, no MBD); -1 se não há features."""
    if index.nlevels == 2:
        keys = pd.MultiIndex.from_arrays([labels.client.astype(str), labels.cutoff.astype(np.float32)])
        index = index.set_levels(index.levels[1].astype(np.float32), level=1)
    else:
        keys = pd.Index(labels.client.astype(str))
        index = index.astype(str)
    return index.get_indexer(keys)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base", choices=list(BUILDERS))
    p.add_argument("--label-frac", type=float, default=1.0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--config", default=str(ROOT / "configs" / "baselines" / "xgb.yaml"))
    p.add_argument("--set", nargs="*", default=[], help="sobrescritas, ex.: max_rounds=200 params.max_depth=7")
    p.add_argument("--max-rows", type=int, default=None, help="só para testes: limita os dados brutos lidos")
    p.add_argument("--rebuild", action="store_true", help="recalcula as features mesmo com cache")
    p.add_argument("--emb", default=None, help="tag do embedding (src/pragma/export_embeddings.py) a juntar")
    p.add_argument("--emb-only", action="store_true", help="usa só o embedding, sem as features à mão")
    p.add_argument("--raw", action="store_true", help="só colunas cruas, sem engenharia de features")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    frac = args.label_frac
    if args.emb_only and not args.emb:
        p.error("--emb-only exige --emb")
    emb_tag = "" if not args.emb else f"_emb{'only' if args.emb_only else ''}_{args.emb}"
    if args.raw and (args.emb_only or args.base not in RAW_TEXT):
        p.error("--raw só vale para home_credit e amex, e não combina com --emb-only")
    emb_tag = ("_raw" if args.raw else "") + emb_tag
    run_name = args.run_name or f"xgb_{args.base}{emb_tag}" + (f"_frac{frac:g}" if frac < 1 else "")
    run_dir = ROOT / "runs" / "baselines" / run_name
    labels = Labels(args.base)
    task, metric = labels.task, primary_metric(labels.task)
    k = labels.y.shape[1]
    grid = [{**cfg["params"], **g} for g in cfg["grid"][args.base]]
    features = RAW_TEXT[args.base] if args.raw else FEATURE_TEXT[args.base]
    if args.emb:
        emb_text = f"embedding [CLS] congelado do checkpoint {args.emb} (src/pragma/export_embeddings.py)"
        features = emb_text if args.emb_only else f"{features}; mais o {emb_text}"
    description = (f"XGBoost (CPU) em {args.base}, tarefa {task['name']}, com {frac:.0%} das linhas de treino. "
                   f"Features: {features}. Grade de {len(grid)} combinações escolhida na validação.")
    run = RunStatus(run_dir, name=run_name, group="baselines", description=description,
                    total_steps=len(grid) * cfg["max_rounds"] * k,
                    config={"base": args.base, "label_frac": frac, "max_rounds": cfg["max_rounds"],
                            "early_stopping": cfg["early_stopping"], "grid": cfg["grid"][args.base],
                            "params": cfg["params"], "max_rows": args.max_rows,
                            "emb": args.emb, "emb_only": args.emb_only, "raw": args.raw})
    record = Recorder(run_dir, run)
    t_start = time.time()
    try:
        if args.emb_only:
            A, names, fe_s = None, [], 0.0
            found = np.ones(len(labels.y), dtype=bool)
        else:
            index, names, A, fe_s = load_features(args.base, args.max_rows, args.rebuild)
            if args.raw:
                keep = raw_columns(args.base, names)
                A, names = A[:, keep], [names[j] for j in keep]
            names = clean_names(names)
            pos = align(index, labels)
            found = pos >= 0
        E, emb_names = load_embedding(args.emb, args.base, len(labels.y)) if args.emb else (None, [])
        n_feat = len(names) + len(emb_names)
        log(f"{args.base}: {n_feat} features, {found.sum()} de {len(found)} linhas de rótulo com features")
        record(0, {"fe/n_features": n_feat, "fe/seconds": fe_s, "fe/rows_found": found.sum()})

        rows = {s: labels.rows(s)[found[labels.rows(s)]] for s in ("train", "val", "test")}
        rows["train"] = subsample(rows["train"], int(round(frac * len(rows["train"]))), cfg["seed"])
        log(f"linhas: treino {len(rows['train'])} (label_frac={frac}), validação {len(rows['val'])}, "
            f"teste {len(rows['test'])}")
        def matrix(r):
            parts = ([] if A is None else [A[pos[r]]]) + ([] if E is None else [np.asarray(E[r], dtype=np.float32)])
            return parts[0] if len(parts) == 1 else np.hstack(parts)

        Xtr, Xva, Xte = (matrix(rows[s]) for s in ("train", "val", "test"))
        del A
        names = names + emb_names
        Y = labels.y
        P = {s: np.zeros((len(rows[s]), k), dtype=np.float64) for s in ("val", "test")}
        chosen, boosters, offset = [], [], 0
        t_train = time.time()
        for j in range(k):
            ytr, yva = Y[rows["train"], j], Y[rows["val"], j]
            label = task["labels"][j]
            if len(np.unique(ytr)) < 2:  # pode acontecer com 1% dos rótulos num produto raro
                log(f"{label}: só uma classe no treino, prevê a taxa constante")
                for s in P:
                    P[s][:, j] = ytr.mean()
                chosen.append({"label": label, "constant": float(ytr.mean())})
                continue
            extra = None
            if metric == "amex":
                extra = lambda model, dva, epoch, yva=yva: {
                    "val/amex": amex_metric(yva, model.predict(dva, iteration_range=(0, epoch + 1)))}
            best, trials = None, []
            for i, params in enumerate(grid):
                params = {**params, "seed": cfg["seed"]}
                t0 = time.time()
                b = fit(params, Xtr, ytr, Xva, yva, cfg, record, offset, names, extra)
                offset += b.num_boosted_rounds()
                pva = predict(b, Xva, names)
                m = binary_metrics(yva, pva, amex=metric == "amex")
                trials.append({"combo": i, "best_iteration": b.best_iteration, "val": m, "seconds": time.time() - t0})
                log(f"{label} combo {i}: melhor rodada {b.best_iteration}, val {metric} {m[metric]:.4f} "
                    f"auc {m['auc']:.4f} ({time.time() - t0:.0f}s)")
                if best is None or m[metric] > best[1][metric]:
                    best = (b, m, i, pva)
            b, m, i, pva = best
            P["val"][:, j], P["test"][:, j] = pva, predict(b, Xte, names)
            boosters.append(b)
            chosen.append({"label": label, "combo": i, "params": grid[i], "best_iteration": b.best_iteration,
                           "trials": trials})
        train_s = time.time() - t_train

        val = task_metrics(task, Y[rows["val"]], P["val"])
        test = task_metrics(task, Y[rows["test"]], P["test"])
        total_s = time.time() - t_start
        log(f"val: {metric} {val[metric]:.4f} auc {val['auc']:.4f} | test: {metric} {test[metric]:.4f} "
            f"auc {test['auc']:.4f} | {total_s:.0f}s")
        results = {"run_name": run_name, "base": args.base, "task": task, "label_frac": frac,
                   "primary_metric": metric, "n_features": len(names),
                   "n_rows": {s: int(len(r)) for s, r in rows.items()}, "chosen": chosen,
                   "val": val, "test": test,
                   "seconds": {"features": fe_s, "train": train_s, "total": total_s}}
        (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
        importance(boosters, names).to_csv(run_dir / "feature_importance.csv", index=False)
        run.finish({**{f"val/{m}": v for m, v in val.items()}, **{f"test/{m}": v for m, v in test.items()},
                    "time/total_s": total_s})
    except Exception as e:
        run.fail(str(e))
        raise


if __name__ == "__main__":
    main()
