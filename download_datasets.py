"""Baixa datasets tabulares do scikit-learn e salva como CSV em ./data.

Uso:
    python download_datasets.py              # todos (inclui os que exigem internet)
    python download_datasets.py --offline    # apenas os embutidos no scikit-learn
    python download_datasets.py iris titanic # apenas os nomeados
    python download_datasets.py --list
"""

import argparse
from pathlib import Path

from sklearn import datasets

DATA_DIR = Path(__file__).parent / "data"

# Embutidos no pacote (sem internet): nome -> loader
BUILTIN = {
    "iris": datasets.load_iris,                    # classificação, 150x4, 3 classes
    "wine": datasets.load_wine,                    # classificação, 178x13, 3 classes
    "breast_cancer": datasets.load_breast_cancer,  # classificação binária, 569x30
    "diabetes": datasets.load_diabetes,            # regressão, 442x10
    "linnerud": datasets.load_linnerud,            # regressão multi-alvo, 20x3
}

def openml(name, version=None):
    def fetch():
        try:
            return datasets.fetch_openml(name, version=version, as_frame=True)
        except TypeError:
            # o parser pandas do sklearn quebra em alguns ARFF (titanic, bike_sharing, kc_house)
            return datasets.fetch_openml(name, version=version, as_frame=True, parser="liac-arff")
    return fetch


# Baixados na primeira execução (cache em ~/scikit_learn_data)
REMOTE = {
    "california_housing": lambda: datasets.fetch_california_housing(as_frame=True),  # regressão, 20640x8
    "adult": openml("adult", 2),                      # classif., misto num/cat, valores nulos
    "titanic": openml("titanic", 1),                  # classif., misto, nulos
    "house_prices": openml("house_prices"),           # regressão, 80 colunas, muitos nulos
    "credit_g": openml("credit-g", 1),                # classif., categóricas
    "bike_sharing": openml("Bike_Sharing_Demand", 2), # regressão, série temporal
    "kc_house": openml("house_sales", 3),             # regressão, 21613 linhas
    "covertype": lambda: datasets.fetch_covtype(as_frame=True),  # classif. 7 classes, 581k linhas
}


def save(name, bunch):
    """Salva bunch.frame (features + alvo) em data/<name>.csv."""
    frame = bunch.frame
    if frame is None:  # loaders sem as_frame embutido
        frame = bunch.data.copy()
        frame["target"] = bunch.target
    path = DATA_DIR / f"{name}.csv"
    frame.to_csv(path, index=False)
    print(f"  {name:20s} {frame.shape[0]:>7} x {frame.shape[1]:<3} -> {path}")


def main():
    all_names = {**BUILTIN, **REMOTE}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("names", nargs="*", help="datasets a baixar (padrão: todos)")
    parser.add_argument("--offline", action="store_true", help="apenas datasets embutidos")
    parser.add_argument("--list", action="store_true", help="lista datasets disponíveis")
    args = parser.parse_args()

    if args.list:
        print("Embutidos:", ", ".join(BUILTIN))
        print("Online:   ", ", ".join(REMOTE))
        return

    unknown = [n for n in args.names if n not in all_names]
    if unknown:
        parser.error(f"desconhecido(s): {', '.join(unknown)} (use --list)")

    names = args.names or (list(BUILTIN) if args.offline else list(all_names))
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Salvando em {DATA_DIR}")
    for name in names:
        loader = all_names[name]
        try:
            bunch = loader(as_frame=True) if name in BUILTIN else loader()
            save(name, bunch)
        except Exception as e:  # falha de rede em um não deve parar os demais
            print(f"  {name:20s} FALHOU: {e}")


if __name__ == "__main__":
    main()
