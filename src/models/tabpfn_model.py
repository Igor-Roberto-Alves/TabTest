"""TabPFN: transformer pré-treinado que faz in-context learning (o fit não treina pesos).

Pesos no Hugging Face (org Prior-Labs), baixados automaticamente no primeiro uso. As
versões >= 2.5 (o padrão do pacote) são gated e não-comerciais: aceite a licença em
https://ux.priorlabs.ai e exporte TABPFN_TOKEN=<api key>. Sem token, use
--tabpfn-version V2 (Apache 2.0 com atribuição).

Dados grandes: o contexto de cada passada do TabPFN tem `context` linhas (V2 foi
pré-treinado até ~10k; contextos maiores pioraram nos nossos testes). Para usar o treino
inteiro, cada membro do ensemble recebe uma fatia diferente e o número de membros cresce
com N: n_estimators = max(8, ceil(coverage * N / context)).

Amostragem de cada fatia (`sampling`):
- "auto": classificação com classe minoritária pequena (< metade do contexto) usa
  "majority_downsample" (toda fatia leva todos os exemplos raros e a probabilidade é
  corrigida para o prior do treino); demais classificações usam "stratified" e
  regressão usa "balanced".
- ou qualquer valor de SAMPLE_SUBSAMPLING_METHOD do TabPFN.
"""

import math

import numpy as np

from src.data import Split, subsample
from src.models.base import Prediction
from src.preprocessing import ordinal_encode

KINDS = {"binary", "multiclass", "regression"}


def choose_sampling(split: Split, n_train: int, y_train, context: int) -> str:
    if split.is_regression:
        return "balanced"
    counts = np.bincount(y_train)
    minority = n_train - counts.max()
    return "majority_downsample" if minority < context // 2 else "stratified"


def fit_predict(split: Split, seed=0, max_train: int | None = None, version: str | None = None,
                context: int = 10_000, coverage: float = 1.0, sampling: str = "auto",
                n_estimators: int | None = None, predict_batch: int = 50_000, writer=None) -> Prediction:
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    X_train, X_test, cat_idx = ordinal_encode(split)
    X_train, y_train = subsample(X_train, split.y_train, max_train, seed)
    n = len(X_train)

    kwargs = dict(random_state=seed, categorical_features_indices=cat_idx, ignore_pretraining_limits=True)
    config = {"context": min(n, context), "n_estimators": n_estimators or "auto", "sampling": "none"}
    if n > context:  # treino maior que o contexto: ensemble de fatias cobrindo tudo
        sampling = choose_sampling(split, n, y_train, context) if sampling == "auto" else sampling
        n_estimators = n_estimators or max(8, math.ceil(coverage * n / context))
        kwargs["inference_config"] = {"SUBSAMPLE_SAMPLES": context, "SAMPLE_SUBSAMPLING_METHOD": sampling}
        config.update(n_estimators=n_estimators, sampling=sampling)
    if n_estimators:
        kwargs["n_estimators"] = n_estimators

    cls = TabPFNRegressor if split.is_regression else TabPFNClassifier
    if version:  # ex.: "V2" para os pesos Apache; padrão = versão mais nova do pacote
        from tabpfn.constants import ModelVersion

        model = cls.create_default_for_version(ModelVersion[version.upper()], **kwargs)
    else:
        model = cls(**kwargs)

    if writer is not None:
        writer.add_text("tabpfn/config", ", ".join(f"{k}={v}" for k, v in config.items()), 0)
    print(f"    tabpfn: n_train={n} " + " ".join(f"{k}={v}" for k, v in config.items()))

    model.fit(X_train, y_train)
    chunks = range(0, len(X_test), predict_batch)
    if split.is_regression:
        return Prediction(np.concatenate([model.predict(X_test[i:i + predict_batch]) for i in chunks]), model=model)
    proba = np.concatenate([model.predict_proba(X_test[i:i + predict_batch]) for i in chunks])
    return Prediction(proba.argmax(axis=1), proba, model)
