"""TabSTAR: modelo fundacional tabular que verbaliza as células como texto.

Pesos em `alana89/TabSTAR` no Hugging Face, baixados automaticamente no fit().
Diferente do TabPFN, o fit faz fine-tuning LoRA (até 50 épocas, early stopping),
então GPU é fortemente recomendada. Recebe o DataFrame bruto: nomes de colunas
e rótulos de classe são usados semanticamente, por isso passamos os originais.
Requer Python 3.10–3.12.
"""

import numpy as np
import pandas as pd

from src.data import Split, subsample
from src.models.base import Prediction

KINDS = {"binary", "multiclass", "regression"}
MAX_TRAIN = 5_000


def fit_predict(split: Split, seed=0, max_train=MAX_TRAIN, time_limit: int | None = None) -> Prediction:
    from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

    X_train, y_train = subsample(split.X_train, split.y_train, max_train, seed)
    X_train, X_test = X_train.reset_index(drop=True), split.X_test.reset_index(drop=True)

    if split.is_regression:
        model = TabSTARRegressor(random_state=seed, time_limit=time_limit)
        model.fit(X_train, pd.Series(y_train, name=split.task.target))
        return Prediction(np.asarray(model.predict(X_test)).ravel(), model=model)

    model = TabSTARClassifier(random_state=seed, time_limit=time_limit)
    model.fit(X_train, pd.Series(split.classes[y_train], name=split.task.target))
    labels = np.asarray(model.predict(X_test)).astype(str)
    y_pred = np.searchsorted(split.classes, labels)  # rótulo original -> inteiro

    # assume colunas na ordem ordenada dos rótulos (mesma do LabelEncoder);
    # se o formato não bater, avalia só as predições duras
    proba = np.asarray(model.predict_proba(X_test))
    if proba.ndim == 1 and len(split.classes) == 2:
        proba = np.column_stack([1 - proba, proba])
    if proba.ndim != 2 or proba.shape[1] != len(split.classes):
        proba = None
    return Prediction(y_pred, proba, model)
