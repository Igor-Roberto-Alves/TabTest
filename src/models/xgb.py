"""XGBoost com suporte nativo a categóricas e nulos (sem pesos pré-treinados: treina do zero)."""

from src.data import Split
from src.models.base import Prediction
from src.preprocessing import to_category

KINDS = {"binary", "multiclass", "regression"}


def fit_predict(split: Split, seed=0, n_estimators=500, learning_rate=0.05, max_depth=6) -> Prediction:
    import xgboost as xgb

    X_train, X_test = to_category(split)
    params = dict(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, random_state=seed, n_jobs=-1,
    )
    if split.is_regression:
        model = xgb.XGBRegressor(**params).fit(X_train, split.y_train)
        return Prediction(model.predict(X_test), model=model)

    model = xgb.XGBClassifier(**params).fit(X_train, split.y_train)
    proba = model.predict_proba(X_test)
    return Prediction(proba.argmax(axis=1), proba, model)
