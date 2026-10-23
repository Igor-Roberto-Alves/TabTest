"""Baixa tabelas extras do OpenML para o corpus de pré-treino (data/corpus/*.csv).

Uso:
    python -m src.ldm.corpus_download
    python -m src.ldm.corpus_download --max-rows 100000

Quanto mais tabelas (e mais variadas), melhor o pré-treino: a lista abaixo é só um
ponto de partida. Datasets já baixados são pulados.
"""

import argparse
from pathlib import Path

from sklearn.datasets import fetch_openml

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"

# (nome no OpenML, versão). Não inclua cópias dos datasets de avaliação de src/tasks.py:
# "houses", por exemplo, é o California Housing e vazaria as tarefas california_*.
OPENML = [
    ("phoneme", 1), ("spambase", 1), ("kc1", 1), ("pc4", 1), ("bank-marketing", 1),
    ("electricity", 1), ("MagicTelescope", 1), ("blood-transfusion-service-center", 1),
    ("ilpd", 1), ("qsar-biodeg", 1), ("kr-vs-kp", 1), ("segment", 1), ("vehicle", 1),
    ("car", 1), ("mushroom", 1), ("abalone", 1), ("credit-approval", 1),
    ("banknote-authentication", 1), ("cmc", 1), ("vowel", 1), ("wilt", 1),
    ("ozone-level-8hr", 1), ("steel-plates-fault", 1), ("churn", 1), ("cpu_act", 1),
    ("pol", 1), ("elevators", 1), ("sulfur", 1), ("wine_quality", 1),
    ("eucalyptus", 1), ("diabetes", 1), ("Australian", 4), ("nursery", 1), ("letter", 1),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", type=Path, default=CORPUS_DIR)
    parser.add_argument("--max-rows", type=int, default=100_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, version in OPENML:
        # prefixo evita colisão com data/*.csv (ex.: o "diabetes" do OpenML é o Pima, não o do sklearn)
        path = args.out / f"openml_{name}.csv"
        if path.exists():
            continue
        try:
            frame = fetch_openml(name, version=version, as_frame=True, parser="auto").frame
        except Exception:
            try:  # o parser pandas falha em alguns ARFF (ver titanic em download_datasets.py)
                frame = fetch_openml(name, version=version, as_frame=True, parser="liac-arff").frame
            except Exception as e:
                print(f"  {name:35s} FALHOU: {e}")
                continue
        if len(frame) > args.max_rows:
            frame = frame.sample(args.max_rows, random_state=0)
        frame.to_csv(path, index=False)
        print(f"  {name:35s} {frame.shape[0]:>7} x {frame.shape[1]:<4} -> {path}")


if __name__ == "__main__":
    main()
