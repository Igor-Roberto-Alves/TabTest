"""TabPFN: transformer pré-treinado que faz in-context learning (o fit não treina pesos).

Pesos no Hugging Face (org Prior-Labs: tabpfn_3, tabpfn_3_5, tabpfn_2_5, ...),
baixados automaticamente no primeiro uso. As versões >= 2.5 são gated e
não-comerciais: aceite a licença na página do modelo e faça `hf auth login`
(ou exporte HF_TOKEN). A v2 tem licença Apache 2.0 com atribuição.
Em CPU, acima de ~1000 linhas de treino é preciso TABPFN_ALLOW_CPU_LARGE_DATASET=1.
"""

from src.data import Split, subsample
from src.models.base import Prediction
from src.preprocessing import ordinal_encode

NAME = "tabpfn"
INSTALL = "tabpfn"
KINDS = {"binary", "multiclass", "regression"}
MAX_TRAIN = 10_000  # o contexto do TabPFN é o próprio treino; acima disso fica lento


def fit_predict(split: Split, seed=0, max_train=MAX_TRAIN, version: str | None = None) -> Prediction:
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    X_train, X_test, cat_idx = ordinal_encode(split)
    X_train, y_train = subsample(X_train, split.y_train, max_train, seed)

    cls = TabPFNRegressor if split.is_regression else TabPFNClassifier
    kwargs = dict(random_state=seed, categorical_features_indices=cat_idx, ignore_pretraining_limits=True)
    if version:  # ex.: "V2" para os pesos Apache; padrão = versão mais nova do pacote
        from tabpfn.constants import ModelVersion

        model = cls.create_default_for_version(ModelVersion[version.upper()], **kwargs)
    else:
        model = cls(**kwargs)

    model.fit(X_train, y_train)
    if split.is_regression:
        return Prediction(model.predict(X_test), model=model)
    proba = model.predict_proba(X_test)
    return Prediction(proba.argmax(axis=1), proba, model)
