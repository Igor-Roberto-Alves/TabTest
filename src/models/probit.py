"""Regressão probit via statsmodels: P(y=1|x) = Φ(xβ).

Binária por natureza; multiclasse é feita em one-vs-rest com probabilidades
renormalizadas. `results_[k].summary()` dá coeficientes, erros-padrão e p-valores.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline

from src.data import Split
from src.models.base import Prediction
from src.preprocessing import linear_preprocessor

NAME = "probit"
INSTALL = "statsmodels"
KINDS = {"binary", "multiclass"}


class ProbitClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, maxiter=1000):
        self.maxiter = maxiter

    def _fit_binary(self, X, y):
        import statsmodels.api as sm

        # lbfgs tolera colinearidade (ex.: education vs education-num no adult),
        # onde o Newton padrão falha com matriz singular
        return sm.Probit(y, X).fit(method="lbfgs", maxiter=self.maxiter, disp=0)

    def fit(self, X, y):
        import statsmodels.api as sm

        X = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            self.results_ = [self._fit_binary(X, (y == self.classes_[1]).astype(float))]
        else:
            self.results_ = [self._fit_binary(X, (y == c).astype(float)) for c in self.classes_]
        return self

    def predict_proba(self, X):
        import statsmodels.api as sm

        X = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
        scores = np.column_stack([r.predict(X) for r in self.results_])
        if len(self.classes_) == 2:
            return np.column_stack([1 - scores[:, 0], scores[:, 0]])
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def fit_predict(split: Split, seed=0) -> Prediction:
    model = make_pipeline(linear_preprocessor(split, drop_first=True), ProbitClassifier())
    model.fit(split.X_train, split.y_train)
    proba = model.predict_proba(split.X_test)
    return Prediction(proba.argmax(axis=1), proba, model)
