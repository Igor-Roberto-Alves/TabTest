"""Regressão logística (L2) com one-hot + padronização. Multiclasse via softmax."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.data import Split
from src.models.base import Prediction
from src.preprocessing import linear_preprocessor

NAME = "logistic"
INSTALL = ""
KINDS = {"binary", "multiclass"}


def fit_predict(split: Split, seed=0, C=1.0) -> Prediction:
    model = make_pipeline(linear_preprocessor(split), LogisticRegression(C=C, max_iter=5000, random_state=seed))
    model.fit(split.X_train, split.y_train)
    proba = model.predict_proba(split.X_test)
    return Prediction(proba.argmax(axis=1), proba, model)
