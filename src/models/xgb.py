"""XGBoost com suporte nativo a categóricas e nulos (sem pesos pré-treinados: treina do zero).

Com `writer`, registra a curva da métrica de avaliação por árvore em treino e teste.
O teste entra só como monitoramento (não há early stopping nem seleção com ele).
"""

from src.data import Split
from src.models.base import Prediction
from src.preprocessing import to_category

KINDS = {"binary", "multiclass", "regression"}


def _log_curves(model, writer):
    names = {"validation_0": "train", "validation_1": "test"}
    for split_name, curves in model.evals_result().items():
        for metric, values in curves.items():
            for step, value in enumerate(values, 1):
                writer.add_scalar(f"curves/{metric}/{names.get(split_name, split_name)}", value, step)


def fit_predict(split: Split, seed=0, n_estimators=500, learning_rate=0.05, max_depth=6, writer=None) -> Prediction:
    import xgboost as xgb

    X_train, X_test = to_category(split)
    params = dict(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, random_state=seed, n_jobs=-1,
    )
    cls = xgb.XGBRegressor if split.is_regression else xgb.XGBClassifier
    model = cls(**params)
    fit_kw = {}
    if writer is not None:
        fit_kw = {"eval_set": [(X_train, split.y_train), (X_test, split.y_test)], "verbose": False}
    model.fit(X_train, split.y_train, **fit_kw)
    if writer is not None:
        _log_curves(model, writer)

    if split.is_regression:
        return Prediction(model.predict(X_test), model=model)
    proba = model.predict_proba(X_test)
    return Prediction(proba.argmax(axis=1), proba, model)
