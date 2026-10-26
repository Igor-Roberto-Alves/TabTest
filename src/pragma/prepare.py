"""Converte as bases brutas (download_banking.py) para o formato comum de store.py.

Uso:
    python -m src.pragma.prepare home_credit amex mbd_mini
    python -m src.pragma.prepare home_credit --max-clients 2000 --suffix _dev   # amostra para testes

Por base:
    home_credit  perfil = application_{train,test}; fontes = bureau, bureau_balance,
                 previous_application, pos_cash, credit_card, installments. Tempo em dias
                 relativos à solicitação (os MONTHS_BALANCE viram dias). Rótulo TARGET; os clientes
                 do teste do Kaggle entram sem rótulo (só pré-treino).
    amex         sem perfil; fonte = statement (um evento por extrato). Tempo em dias relativos
                 ao último extrato do cliente. Colunas renomeadas pelo prefixo
                 (D_39 -> "delinquency 39") para o encoder de texto ter alguma semântica.
    mbd_mini/mbd sem perfil; fontes = trx, geo, dialog (embedding de 768 dims reduzido por PCA a
                 --dialog-pca componentes numéricas). Tempo em dias desde 2021-01-01. Rótulos
                 por (cliente, mês): só eventos antes do 1º dia do mês do rótulo entram.

Os encoders (bins de quantil, categorias) são ajustados numa amostra de linhas de cada fonte.
O split é por cliente: 80/10/10 no Home Credit e na AmEx, e pelos folds oficiais no MBD
(0-2 treino, 3 validação, 4 teste).
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ldm.table import CAT, TableEncoder

from .store import NO_CAL, SPLITS, save_source, save_store

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "banking"
OUT = RAW / "prepared"
DAYS_PER_MONTH = 30.44


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_encoder(df, enc_kw, kinds=None, sample=300_000, seed=0):
    if len(df) > sample:
        df = df.sample(sample, random_state=seed)
    return TableEncoder(**enc_kw).fit(df, kinds)


def encode_chunked(encoder, df, chunk=1_000_000):
    local = np.empty((len(df), len(encoder.columns)), dtype=np.int16)
    z = np.empty((len(df), len(encoder.columns)), dtype=np.float16)
    for i in range(0, len(df), chunk):
        part = encoder.encode(df.iloc[i:i + chunk])
        local[i:i + chunk] = part["local"]
        z[i:i + chunk] = part["z"]
    return local, z


def calendar(ts: pd.Series) -> np.ndarray:
    """Timestamps -> (E, 3) int8: hora, dia da semana, dia do mês."""
    return np.stack([ts.dt.hour, ts.dt.day_of_week, ts.dt.day], axis=1).astype(np.int8)


def random_split(n, rng, frac=(0.8, 0.1, 0.1)):
    split = np.zeros(n, dtype=np.int8)
    perm = rng.permutation(n)
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    split[perm[n_train:n_train + n_val]] = SPLITS["val"]
    split[perm[n_train + n_val:]] = SPLITS["test"]
    return split


def add_source(sources, out, name, df, cidx, t, cal, n_clients, enc_kw, kinds=None, seed=0):
    keep = cidx >= 0
    df, cidx, t = df[keep].reset_index(drop=True), cidx[keep], t[keep]
    cal = np.broadcast_to(NO_CAL, (len(df), 3)) if cal is None else cal[keep]
    encoder = fit_encoder(df, enc_kw, kinds, seed=seed)
    local, z = encode_chunked(encoder, df)
    sources[name] = save_source(out, name, encoder, local, z, cidx, np.asarray(t, dtype=np.float64), cal, n_clients)
    log(f"  {name}: {len(df)} eventos x {df.shape[1]} colunas")


# ---------------------------------------------------------------- Home Credit
def prepare_home_credit(out, max_clients, seed, enc_kw, **_):
    raw = RAW / "home_credit"
    read = lambda f: pd.read_csv(raw / f"{f}.csv", low_memory=False)
    rng = np.random.default_rng(seed)

    train, test = read("application_train"), read("application_test")
    app = pd.concat([train, test], ignore_index=True)
    if max_clients:
        app = app.sample(min(max_clients, len(app)), random_state=seed).reset_index(drop=True)
    index = pd.Index(app["SK_ID_CURR"])
    n = len(app)
    labelled = app["TARGET"].notna().to_numpy()
    split = np.full(n, SPLITS["unlabeled"], dtype=np.int8)
    split[labelled] = random_split(labelled.sum(), rng)
    log(f"home_credit: {n} clientes ({labelled.sum()} rotulados)")

    profile_df = app.drop(columns=["SK_ID_CURR", "TARGET"])
    p_enc = fit_encoder(profile_df, enc_kw, seed=seed)
    p_local, p_z = encode_chunked(p_enc, profile_df)

    def clean(df, drop):
        df = df.drop(columns=[c for c in drop if c in df.columns])
        for c in df.columns:  # 365243 = "sem data" no previous_application
            if c.startswith("DAYS_"):
                df[c] = df[c].where(df[c] != 365243)
        return df

    sources = {}
    bureau = read("bureau")
    cidx = index.get_indexer(bureau["SK_ID_CURR"])
    add_source(sources, out, "bureau", clean(bureau, ["SK_ID_CURR", "SK_ID_BUREAU"]), cidx,
               bureau["DAYS_CREDIT"].to_numpy(), None, n, enc_kw, seed=seed)

    bb = read("bureau_balance")
    owner = pd.Series(cidx, index=bureau["SK_ID_BUREAU"])
    bb_cidx = owner.reindex(bb["SK_ID_BUREAU"]).fillna(-1).to_numpy().astype(np.int64)
    add_source(sources, out, "bureau_balance", clean(bb, ["SK_ID_BUREAU"]), bb_cidx,
               bb["MONTHS_BALANCE"].to_numpy() * DAYS_PER_MONTH, None, n, enc_kw, seed=seed)
    del bureau, bb, owner

    specs = [("previous_application", "previous_application", lambda d: d["DAYS_DECISION"]),
             ("pos_cash", "POS_CASH_balance", lambda d: d["MONTHS_BALANCE"] * DAYS_PER_MONTH),
             ("credit_card", "credit_card_balance", lambda d: d["MONTHS_BALANCE"] * DAYS_PER_MONTH),
             ("installments", "installments_payments", lambda d: d["DAYS_INSTALMENT"])]
    for name, file, time_of in specs:
        df = read(file)
        t = time_of(df).fillna(0).to_numpy()
        add_source(sources, out, name, clean(df, ["SK_ID_CURR", "SK_ID_PREV"]), index.get_indexer(df["SK_ID_CURR"]),
                   t, None, n, enc_kw, seed=seed)
        del df

    rows = np.nonzero(labelled)[0]
    labels = (rows, np.full(len(rows), np.inf), app["TARGET"].to_numpy()[rows])
    task = {"name": "default", "labels": ["TARGET"], "type": "binary", "metric": "auc"}
    save_store(out, out.name, app["SK_ID_CURR"].to_numpy(), split, sources, task, labels,
               profile=(p_enc, p_local, p_z), notes="t em dias relativos à solicitação (<= 0)")


# ----------------------------------------------------------------------- AmEx
AMEX_PREFIX = {"D": "delinquency", "S": "spend", "P": "payment", "B": "balance", "R": "risk"}
AMEX_CAT = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]


def amex_name(col):
    m = re.fullmatch(r"([DSPBR])_(\d+)", col)
    return f"{AMEX_PREFIX[m.group(1)]} {m.group(2)}" if m else col


def prepare_amex(out, max_clients, seed, enc_kw, chunk=500_000, **_):
    raw = RAW / "amex"
    rng = np.random.default_rng(seed)
    kinds = {amex_name(c): CAT for c in AMEX_CAT}
    encoder, parts, seen = None, [], {}
    for df in pd.read_csv(raw / "train_data.csv", chunksize=chunk):
        if max_clients:  # o arquivo vem agrupado por cliente: fica com os primeiros max_clients
            for c in df["customer_ID"].unique():
                if len(seen) < max_clients:
                    seen.setdefault(c, len(seen))
            df = df[df["customer_ID"].isin(list(seen))]
            if df.empty:
                break
        dates = pd.to_datetime(df["S_2"])
        feats = df.drop(columns=["customer_ID", "S_2"]).rename(columns=amex_name)
        if encoder is None:
            encoder = TableEncoder(**enc_kw).fit(feats, kinds)
        local, z = encode_chunked(encoder, feats)
        parts.append((df["customer_ID"].to_numpy(), (dates - pd.Timestamp("2017-01-01")).dt.days.to_numpy(),
                      calendar(dates), local, z))
        log(f"amex: {sum(len(p[0]) for p in parts)} extratos lidos")
    ids_rows = np.concatenate([p[0] for p in parts])
    days = np.concatenate([p[1] for p in parts]).astype(np.float64)
    cal = np.concatenate([p[2] for p in parts])
    cal[:, 0] = -1  # extrato não tem hora
    local = np.concatenate([p[3] for p in parts])
    z = np.concatenate([p[4] for p in parts])
    del parts
    cidx, clients = pd.factorize(ids_rows)
    n = len(clients)
    last = np.full(n, -np.inf)
    np.maximum.at(last, cidx, days)
    t = days - last[cidx]  # 0 = último extrato

    sources = {"statement": save_source(out, "statement", encoder, local, z, cidx, t, cal, n)}
    log(f"  statement: {len(t)} eventos x {local.shape[1]} colunas")
    labels_df = pd.read_csv(raw / "train_labels.csv")
    y = labels_df.set_index("customer_ID")["target"].reindex(clients).to_numpy()
    rows = np.nonzero(~np.isnan(y))[0]
    split = np.full(n, SPLITS["unlabeled"], dtype=np.int8)
    split[rows] = random_split(len(rows), rng)
    task = {"name": "default", "labels": ["target"], "type": "binary", "metric": "amex"}
    save_store(out, out.name, np.asarray(clients), split, sources, task, (rows, np.full(len(rows), np.inf), y[rows]),
               notes="t em dias relativos ao último extrato (<= 0); negativos subamostrados a 5% na origem")


# ------------------------------------------------------------------------ MBD
MBD_EPOCH = pd.Timestamp("2021-01-01")
MBD_TRX_CAT = ["event_type", "event_subtype", "currency", "src_type11", "src_type12", "dst_type11", "dst_type12",
               "src_type21", "src_type22", "src_type31", "src_type32"]


def read_mbd(path, columns, ids):
    """Lê uma modalidade em lotes, trocando o client_id (hash de 64 caracteres) pelo índice do cliente."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    value_set = pa.array(ids)
    filt = ds.field("client_id").isin(value_set) if len(ids) < 1_000_000 else None
    frames = []
    for batch in dataset.to_batches(columns=["client_id", "event_time", *columns], filter=filt, batch_size=2_000_000):
        cidx = pc.index_in(batch.column("client_id"), value_set=value_set).fill_null(-1).to_numpy()
        df = pa.Table.from_batches([batch]).drop(["client_id"]).to_pandas()
        df["__cidx"] = cidx
        frames.append(df[cidx >= 0])
    return pd.concat(frames, ignore_index=True)


def prepare_mbd(out, max_clients, seed, enc_kw, raw_name="mbd_mini", dialog_pca=16, **_):
    raw = RAW / raw_name
    split_df = pd.read_parquet(raw / "client_split")
    if max_clients:
        split_df = split_df.sample(min(max_clients, len(split_df)), random_state=seed)
    split_df = split_df.reset_index(drop=True)
    ids = split_df["client_id"].astype(str).to_numpy()
    n = len(ids)
    fold = split_df["fold"].astype(int).to_numpy()
    split = np.select([fold <= 2, fold == 3], [SPLITS["train"], SPLITS["val"]], SPLITS["test"]).astype(np.int8)
    log(f"{raw_name}: {n} clientes")

    def times(df):
        ts = pd.to_datetime(df["event_time"])
        return ((ts - MBD_EPOCH) / pd.Timedelta(days=1)).to_numpy(), calendar(ts)

    sources = {}
    trx = read_mbd(raw / "detail" / "trx", ["amount", *MBD_TRX_CAT], ids)
    t, cal = times(trx)
    add_source(sources, out, "trx", trx[["amount", *MBD_TRX_CAT]], trx["__cidx"].to_numpy(), t, cal, n, enc_kw,
               kinds={c: CAT for c in MBD_TRX_CAT}, seed=seed)
    del trx

    geo_cols = ["geohash_4", "geohash_5", "geohash_6"]
    geo = read_mbd(raw / "detail" / "geo", geo_cols, ids)
    t, cal = times(geo)
    add_source(sources, out, "geo", geo[geo_cols], geo["__cidx"].to_numpy(), t, cal, n, enc_kw,
               kinds={c: CAT for c in geo_cols}, seed=seed)
    del geo

    from sklearn.decomposition import PCA

    dialog = read_mbd(raw / "detail" / "dialog", ["embedding"], ids)
    emb = np.stack(dialog["embedding"].to_numpy()).astype(np.float32)
    comps = PCA(dialog_pca, random_state=seed).fit(emb[:200_000]).transform(emb)
    feats = pd.DataFrame(comps, columns=[f"dialog component {i}" for i in range(dialog_pca)])
    t, cal = times(dialog)
    add_source(sources, out, "dialog", feats, dialog["__cidx"].to_numpy(), t, cal, n, enc_kw, seed=seed)
    del dialog, emb

    targets = pd.read_parquet(raw / "targets")
    tcols = ["target_1", "target_2", "target_3", "target_4"]
    tcidx = pd.Index(ids).get_indexer(targets["client_id"].astype(str))
    targets = targets[tcidx >= 0]
    tcidx = tcidx[tcidx >= 0]
    month_start = pd.to_datetime(targets["mon"]).dt.to_period("M").dt.start_time
    cutoff = ((month_start - MBD_EPOCH) / pd.Timedelta(days=1)).to_numpy()
    task = {"name": "propensity", "labels": tcols, "type": "multilabel", "metric": "auc"}
    save_store(out, out.name, ids, split, sources, task, (tcidx, cutoff, targets[tcols].to_numpy()),
               notes="t em dias desde 2021-01-01; rótulo do mês m usa eventos antes do 1º dia de m")


PREPARERS = {"home_credit": prepare_home_credit, "amex": prepare_amex,
             "mbd_mini": lambda out, **kw: prepare_mbd(out, raw_name="mbd_mini", **kw),
             "mbd": lambda out, **kw: prepare_mbd(out, raw_name="mbd", **kw)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+", choices=list(PREPARERS))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--suffix", default="", help="sufixo do diretório de saída (ex.: _dev)")
    p.add_argument("--max-clients", type=int, default=None, help="amostra de clientes, para testes")
    p.add_argument("--n-bins", type=int, default=32)
    p.add_argument("--max-categories", type=int, default=128)
    p.add_argument("--dialog-pca", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    enc_kw = {"n_bins": args.n_bins, "max_categories": args.max_categories}
    for name in args.names:
        out = args.out / f"{name}{args.suffix}"
        t0 = time.time()
        PREPARERS[name](out, max_clients=args.max_clients, seed=args.seed, enc_kw=enc_kw, dialog_pca=args.dialog_pca)
        log(f"{name} -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
