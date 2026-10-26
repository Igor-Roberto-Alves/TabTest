"""Features da AmEx: agregações por customer_ID dos extratos de train_data.csv.

Numéricas: last, mean, std, min, max e last - mean. Categóricas oficiais: last, nunique e count.
Mais o número de extratos. "last" é o último extrato de fato (pela data S_2), mesmo que nulo.

O CSV tem 16 GB: é lido em blocos com pyarrow (float32) e agregado bloco a bloco. O arquivo vem
agrupado por cliente, então só o último cliente de cada bloco pode estar cortado: as linhas dele
passam para o bloco seguinte. No fim, um cliente repetido indicaria que a premissa falhou.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv

from .common import RAW, log

SRC = RAW / "amex" / "train_data.csv"
CAT = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
STRING_CODES = {"D_63": ["CL", "CO", "CR", "XL", "XM", "XZ"], "D_64": ["-1", "O", "R", "U"]}


def aggregate(df):
    df = df.sort_values(["customer_ID", "S_2"], kind="stable")
    num = [c for c in df.columns if c not in CAT and c not in ("customer_ID", "S_2")]
    g = df.groupby("customer_ID", sort=True)
    last = df.drop_duplicates("customer_ID", keep="last").set_index("customer_ID")
    stats = g[num].agg(["mean", "std", "min", "max"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    diff = last[num] - stats[[f"{c}_mean" for c in num]].to_numpy()
    cat_n = g[CAT].agg(["nunique", "count"])
    cat_n.columns = [f"{c}_{s}" for c, s in cat_n.columns]
    out = pd.concat([g.size().rename("n_statements"), last[num].add_suffix("_last"), stats,
                     diff.add_suffix("_last_minus_mean"), last[CAT].add_suffix("_last"), cat_n], axis=1)
    return out.astype(np.float32)


def build(max_rows=None, block_rows=400_000):
    pa.set_cpu_count(12)  # o treino do modelo fundacional também usa a CPU
    # blocos pequenos: com 256 MB o leitor antecipa vários blocos e passa de 9 GB de RAM
    header = pcsv.open_csv(SRC).schema.names
    types = {c: pa.float32() for c in header}
    types.update({"customer_ID": pa.string(), "S_2": pa.string(), "D_63": pa.string(), "D_64": pa.string()})
    reader = pcsv.open_csv(SRC, read_options=pcsv.ReadOptions(block_size=16 << 20),
                           convert_options=pcsv.ConvertOptions(column_types=types))
    parts, pending, carry, seen = [], [], None, 0

    def flush(df, final):
        nonlocal carry
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        if not final:  # o último cliente pode continuar no próximo bloco
            tail = df["customer_ID"].iloc[-1]
            cut = df["customer_ID"].to_numpy() == tail
            carry, df = df[cut], df[~cut]
        for c, cats in STRING_CODES.items():
            df[c] = pd.Categorical(df[c], categories=cats).codes.astype(np.float32)
            df.loc[df[c] < 0, c] = np.nan
        parts.append(aggregate(df))

    for batch in reader:
        pending.append(batch)
        n = sum(len(b) for b in pending)
        if n >= block_rows:
            seen += n
            flush(pa.Table.from_batches(pending).to_pandas(), final=False)
            pending = []
            log(f"amex: {seen} extratos lidos, {sum(len(p) for p in parts)} clientes")
            if max_rows and seen >= max_rows:
                break
    if pending or carry is not None:
        rest = pa.Table.from_batches(pending, schema=reader.schema).to_pandas() if pending else carry.iloc[:0]
        flush(rest, final=True)
    feats = pd.concat(parts)
    if feats.index.duplicated().any():
        raise RuntimeError("cliente repetido entre blocos: o CSV não está agrupado por customer_ID")
    log(f"amex: {len(feats)} clientes x {feats.shape[1]} features")
    return feats.rename_axis("id")
