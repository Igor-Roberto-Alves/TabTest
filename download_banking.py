"""Baixa as bases de eventos bancários da branch pragma-banking para ./data/banking.

- mbd:         Multimodal Banking Dataset (Hugging Face, ai-lab/MBD ou ai-lab/MBD-mini)
- home_credit: Home Credit Default Risk (Kaggle)
- amex:        American Express Default Prediction (Kaggle)

As bases do Kaggle exigem credenciais (`kaggle auth login`, a variável KAGGLE_API_TOKEN ou
~/.kaggle/access_token) e aceitar as regras de cada competição no site antes do primeiro download.

Uso:
    python download_banking.py                          # as três (MBD-mini, AmEx só treino)
    python download_banking.py home_credit amex         # apenas as nomeadas
    python download_banking.py mbd --mbd-full           # MBD completo (~69 GB compactado)
    python download_banking.py mbd --mbd-parts client_split targets
    python download_banking.py amex --amex parquet      # versão parquet da comunidade (menor)
"""

import argparse
import tarfile
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "banking"

# MBD: arquivos do repositório no Hugging Face. detail = eventos em formato longo,
# ptls = os mesmos eventos agregados por cliente (formato pytorch-lifestream).
MBD_PARTS = ["client_split", "targets", "detail", "ptls"]
MBD_DEFAULT_PARTS = ["client_split", "targets", "detail"]

# AmEx: o CSV oficial tem ~16 GB (treino) e ~33 GB (teste).
AMEX_TRAIN_FILES = ["train_data.csv", "train_labels.csv"]
AMEX_PARQUET = "raddar/amex-data-integer-dtypes-parquet-format"  # features desquantizadas para inteiros


def extract_tar(path, dest):
    with tarfile.open(path) as tar:
        tar.extractall(dest, filter="data")  # recusa caminhos absolutos, "..", links para fora


def extract_zips(folder, keep):
    for z in sorted(folder.glob("*.zip")):
        with zipfile.ZipFile(z) as f:
            f.extractall(folder)
        if not keep:
            z.unlink()


def mbd(args):
    from huggingface_hub import hf_hub_download

    repo = "ai-lab/MBD" if args.mbd_full else "ai-lab/MBD-mini"
    dest = DATA_DIR / ("mbd" if args.mbd_full else "mbd_mini")
    for part in args.mbd_parts:
        if (dest / part).exists():
            print(f"  {repo}/{part}: já extraído em {dest / part}")
            continue
        print(f"  {repo}/{part}.tar.gz ...")
        archive = Path(hf_hub_download(repo, f"{part}.tar.gz", repo_type="dataset", local_dir=dest))
        extract_tar(archive, dest)
        if not args.keep_archives:
            archive.unlink()
    print(f"  mbd -> {dest}")


def kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise RuntimeError("pacote kaggle ausente: pip install kaggle")
    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit:  # o kaggle chama exit() sem credenciais, o que pararia as outras bases
        raise RuntimeError("sem credenciais do Kaggle: rode `kaggle auth login` ou defina KAGGLE_API_TOKEN")
    return api


def home_credit(args):
    dest = DATA_DIR / "home_credit"
    dest.mkdir(parents=True, exist_ok=True)
    kaggle_api().competition_download_files("home-credit-default-risk", path=dest, quiet=False)
    extract_zips(dest, args.keep_archives)
    print(f"  home_credit -> {dest}")


def amex(args):
    dest = DATA_DIR / ("amex_parquet" if args.amex == "parquet" else "amex")
    dest.mkdir(parents=True, exist_ok=True)
    api = kaggle_api()
    if args.amex == "parquet":
        api.dataset_download_files(AMEX_PARQUET, path=dest, quiet=False)
    elif args.amex == "all":
        api.competition_download_files("amex-default-prediction", path=dest, quiet=False)
    else:
        for name in AMEX_TRAIN_FILES:
            api.competition_download_file("amex-default-prediction", name, path=dest, quiet=False)
    extract_zips(dest, args.keep_archives)
    print(f"  amex -> {dest}")


DATASETS = {"mbd": mbd, "home_credit": home_credit, "amex": amex}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"bases a baixar entre {', '.join(DATASETS)} (padrão: todas)")
    parser.add_argument("--mbd-full", action="store_true", help="MBD completo em vez do MBD-mini (10%% dos clientes)")
    parser.add_argument("--mbd-parts", nargs="+", choices=MBD_PARTS, default=MBD_DEFAULT_PARTS)
    parser.add_argument("--amex", choices=["train", "all", "parquet"], default="train",
                        help="train: CSVs oficiais de treino; all: com teste; parquet: " + AMEX_PARQUET)
    parser.add_argument("--keep-archives", action="store_true", help="não apaga .zip/.tar.gz após extrair")
    args = parser.parse_args()
    unknown = [n for n in args.names if n not in DATASETS]
    if unknown:
        parser.error(f"desconhecidos: {unknown}")

    for name in args.names or list(DATASETS):
        print(name)
        try:
            DATASETS[name](args)
        except Exception as e:
            print(f"  {name} FALHOU: {e}")


if __name__ == "__main__":
    main()
