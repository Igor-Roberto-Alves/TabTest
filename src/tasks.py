"""Problemas de regressão e classificação sugeridos sobre os CSVs de data/.

Cada Task diz qual CSV ler, qual coluna prever e quais colunas descartar
(vazamento de alvo ou identificadores). `derive` cria alvos derivados.
"""

from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

Kind = Literal["binary", "multiclass", "regression"]


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    target: str
    kind: Kind
    description: str
    drop: tuple[str, ...] = ()
    derive: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    max_rows: int | None = None  # subamostra datasets muito grandes


def _california_expensive(df):
    # binariza o preço mediano: bairro acima da mediana do estado
    df = df.copy()
    df["expensive"] = (df["MedHouseVal"] > df["MedHouseVal"].median()).map({True: "yes", False: "no"})
    return df.drop(columns=["MedHouseVal"])


TASKS = {t.name: t for t in [
    # --- classificação binária ---
    Task("adult_income", "adult", "class", "binary",
         "Renda anual > 50K a partir de dados do censo (num + cat, nulos)."),
    Task("titanic_survival", "titanic", "survived", "binary",
         "Sobrevivência no Titanic. 'boat' e 'body' vazam o alvo e são removidas.",
         drop=("boat", "body", "name", "ticket", "home.dest")),
    Task("credit_risk", "credit_g", "class", "binary",
         "Bom ou mau pagador (German Credit), quase tudo categórico."),
    Task("breast_cancer", "breast_cancer", "target", "binary",
         "Tumor maligno ou benigno, 30 features numéricas."),
    Task("california_expensive", "california_housing", "expensive", "binary",
         "Bairro com preço acima da mediana (alvo derivado do housing).",
         derive=_california_expensive),
    # --- classificação multiclasse ---
    Task("iris", "iris", "target", "multiclass", "Espécie de íris, 3 classes (sanidade)."),
    Task("wine", "wine", "target", "multiclass", "Cultivar do vinho, 3 classes."),
    Task("covertype", "covertype", "Cover_Type", "multiclass",
         "Tipo de cobertura florestal, 7 classes (subamostrado para 20k linhas).",
         max_rows=20_000),
    # --- regressão ---
    Task("california_price", "california_housing", "MedHouseVal", "regression",
         "Preço mediano dos imóveis por bairro da Califórnia."),
    Task("house_prices", "house_prices", "SalePrice", "regression",
         "Preço de venda (Ames), 79 features e muitos nulos.", drop=("Id",)),
    Task("kc_house_price", "kc_house", "price", "regression",
         "Preço de casas em King County.", drop=("id", "date")),
    Task("bike_demand", "bike_sharing", "count", "regression",
         "Aluguéis de bicicleta por hora (efeitos de calendário e clima)."),
    Task("diabetes", "diabetes", "target", "regression",
         "Progressão do diabetes em um ano, 442 linhas (dados pequenos)."),
    Task("adult_hours", "adult", "hours-per-week", "regression",
         "Horas trabalhadas por semana a partir do perfil do censo."),
    Task("titanic_fare", "titanic", "fare", "regression",
         "Valor da passagem a partir de classe, porto e família.",
         drop=("boat", "body", "name", "ticket", "home.dest", "cabin")),
]}
