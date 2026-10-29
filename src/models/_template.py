"""<Nome do modelo>: uma linha sobre a ideia (esta linha aparece em `python -m src.run --list-models`)."""

import numpy as np

from src.data import Split
from src.models.base import Prediction

NAME = "__NAME__"          # nome na linha de comando (--models __NAME__) e no painel
LABEL = "__NAME__"         # rótulo de exibição
DATA = "tabular"           # "tabular" | "series" | "events"
KINDS = {"binary"}         # tarefas suportadas: binary, multiclass, regression
INSTALL = ""               # pacote pip exigido, se houver
CAPABILITIES = {"fit_predict"}  # acrescente "pretrain", "finetune", "explain_mask" se implementar


def fit_predict(split: Split, seed=0, **kw) -> Prediction:
    """Treina em split.X_train/y_train e prediz split.X_test.

    Classificação: devolva y_pred (classe inteira) e proba (n, n_classes).
    Regressão: devolva só y_pred. Importe dependências pesadas aqui dentro, não no topo.
    """
    raise NotImplementedError("implemente fit_predict")
    # exemplo:
    # model = MeuModelo(random_state=seed).fit(X_train, split.y_train)
    # proba = model.predict_proba(X_test)
    # return Prediction(proba.argmax(axis=1), proba, model)
