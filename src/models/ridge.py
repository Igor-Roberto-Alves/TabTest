"""Baseline linear para regressão (contraparte da logística/probit), alpha por CV."""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline

from src.data import Split
from src.models.base import Prediction
from src.preprocessing import linear_preprocessor

KINDS = {"regression"}


def fit_predict(split: Split, seed=0) -> Prediction:
    model = make_pipeline(linear_preprocessor(split), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    model.fit(split.X_train, split.y_train)
    return Prediction(model.predict(split.X_test), model=model)
