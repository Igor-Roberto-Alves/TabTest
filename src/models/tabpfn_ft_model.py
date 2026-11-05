"""TabPFN com fine-tuning dos pesos na tarefa (tabpfn.finetuning).

Diferente do `tabpfn` (só in-context learning), aqui o transformer recebe gradiente:
a cada passo, uma fatia de `context` linhas do treino é dividida em contexto e consulta
e o modelo aprende a prever as consultas. Assim todas as linhas passam pelo treino ao
longo das épocas. Validação interna (validation_split_ratio) faz early stopping.
A inferência final usa o mesmo ensemble de fatias do `tabpfn` (ver tabpfn_model.py).

Curvas no TensorBoard: finetune/* (loss por passo, métricas de validação por época).
Custo: ~5x o `tabpfn` no redistrib_erro (137k linhas), sem ganho ali; vale testar
em datasets em que o prior sintético do TabPFN encaixa pior.
"""

import math

import numpy as np

from src.data import Split, subsample
from src.models.base import Prediction
from src.models.tabpfn_model import choose_sampling
from src.preprocessing import ordinal_encode

KINDS = {"binary", "multiclass", "regression"}


class _TensorBoardLogger:
    """Adapta o FinetuningLogger do TabPFN para um SummaryWriter."""

    def __init__(self, writer):
        self.writer = writer

    def setup(self, config):
        pass

    def log_step(self, metrics, step):
        for k, v in metrics.items():
            if k in ("train/loss", "train/lr"):
                self.writer.add_scalar(f"finetune/{k.split('/')[-1]}", v, step)

    def log_epoch(self, metrics, step):
        epoch = int(metrics.get("train/epoch", step)) + 1
        for k, v in metrics.items():
            if k != "train/epoch" and isinstance(v, int | float) and " " not in k:
                self.writer.add_scalar(f"finetune/{k.replace('/', '_')}", v, epoch)

    def finish(self):
        self.writer.flush()


def fit_predict(split: Split, seed=0, max_train: int | None = None, version: str | None = "V2",
                context: int = 10_000, coverage: float = 1.0, sampling: str = "auto", epochs: int = 30,
                time_limit: int | None = None, learning_rate: float = 1e-5, device: str = "cuda",
                predict_batch: int = 50_000, writer=None) -> Prediction:
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier, FinetunedTabPFNRegressor

    X_train, X_test, cat_idx = ordinal_encode(split)
    X_train, y_train = subsample(X_train, split.y_train, max_train, seed)
    n = len(X_train)

    inner = {"categorical_features_indices": cat_idx, "ignore_pretraining_limits": True}
    if n > context:
        sampling = choose_sampling(split, n, y_train, context) if sampling == "auto" else sampling
        inner["inference_config"] = {"SAMPLE_SUBSAMPLING_METHOD": sampling}

    common = dict(device=device, epochs=epochs, time_limit=time_limit, learning_rate=learning_rate,
                  random_state=seed, model_version=ModelVersion[version.upper()] if version else None,
                  n_finetune_ctx_plus_query_samples=min(n, context),
                  n_inference_subsample_samples=context if n > context else None,
                  n_estimators_final_inference=max(8, math.ceil(coverage * n / context)),
                  experiment_logger=_TensorBoardLogger(writer) if writer is not None else None)
    if split.is_regression:
        model = FinetunedTabPFNRegressor(extra_regressor_kwargs=inner, **common)
    else:
        model = FinetunedTabPFNClassifier(extra_classifier_kwargs=inner,
                                          eval_metric="roc_auc" if split.task.kind == "binary" else "log_loss",
                                          **common)
    model.fit(X_train, y_train)

    chunks = range(0, len(X_test), predict_batch)
    if split.is_regression:
        return Prediction(np.concatenate([model.predict(X_test[i:i + predict_batch]) for i in chunks]), model=model)
    proba = np.concatenate([model.predict_proba(X_test[i:i + predict_batch]) for i in chunks])
    return Prediction(proba.argmax(axis=1), proba, model)
