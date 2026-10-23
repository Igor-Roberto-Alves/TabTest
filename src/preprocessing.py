"""Pré-processamentos compartilhados entre os modelos."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data import Split


def linear_preprocessor(split: Split, drop_first=False) -> ColumnTransformer:
    """Imputa, padroniza numéricas e faz one-hot das categóricas (modelos lineares).

    drop_first evita a armadilha das dummies em modelos sem regularização (probit).
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    onehot = OneHotEncoder(
        drop="first" if drop_first else None,
        handle_unknown="ignore" if drop_first else "infrequent_if_exist",
        min_frequency=10,
        sparse_output=False,
    )
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)])
    return ColumnTransformer([("num", numeric, split.num_cols), ("cat", categorical, split.cat_cols)])


def to_category(split: Split):
    """Converte colunas texto para dtype category (XGBoost enable_categorical)."""
    X_train, X_test = split.X_train.copy(), split.X_test.copy()
    for col in split.cat_cols:
        dtype = pd.CategoricalDtype(X_train[col].dropna().unique())
        X_train[col] = X_train[col].astype(dtype)
        X_test[col] = X_test[col].astype(dtype)  # categoria nova vira NaN
    return X_train, X_test


def ordinal_encode(split: Split):
    """Arrays numéricos com categóricas codificadas como inteiros e NaN preservado.

    Retorna também os índices das colunas categóricas (TabPFN usa isso).
    """
    cat_cols, num_cols = split.cat_cols, split.num_cols
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan)

    def encode(X, fit):
        num = X[num_cols].to_numpy(dtype=float)
        if not cat_cols:
            return num
        cat = encoder.fit_transform(X[cat_cols]) if fit else encoder.transform(X[cat_cols])
        return np.hstack([num, cat])

    X_train, X_test = encode(split.X_train, fit=True), encode(split.X_test, fit=False)
    cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    return X_train, X_test, cat_idx
