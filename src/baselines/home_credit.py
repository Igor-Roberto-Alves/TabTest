"""Features do Home Credit: application_train + agregações por cliente das 6 tabelas de histórico.

Segue o padrão das soluções públicas do Kaggle: numéricas agregadas por cliente (mean, min, max,
sum, mais o número de linhas de cada tabela), contagens e proporções das categorias principais,
e razões clássicas (crédito/renda, anuidade/renda, atraso das parcelas, pagamento/parcela).
O bureau_balance é agregado primeiro por crédito do bureau e depois por cliente. O sentinela
365243 dos DAYS_* vira nulo, como no preparo do modelo fundacional.

Só entram clientes do application_train: os do teste do Kaggle não têm rótulo.
"""

import numpy as np
import pandas as pd

from .common import RAW, log, to_f32

SRC = RAW / "home_credit"
STATS = ["mean", "min", "max", "sum"]


def read(name, usecols=None):
    df = pd.read_csv(SRC / f"{name}.csv", engine="pyarrow", usecols=usecols)
    for c in df.columns:
        if c.startswith("DAYS_"):
            df[c] = df[c].where(df[c] != 365243)
    floats = df.columns[df.dtypes == np.float64]
    df[floats] = df[floats].astype(np.float32)
    return df


def num_agg(df, key, prefix, cols=None, stats=STATS):
    """count (linhas) + stats de cada numérica, por `key`."""
    cols = cols or [c for c in df.columns if c != key and not c.startswith("SK_ID") and
                    pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(key)
    out = g[cols].agg(stats)
    out.columns = [f"{prefix}_{c}_{s}" for c, s in out.columns]
    out.insert(0, f"{prefix}_count", g.size())
    return out.astype(np.float32)


def cat_agg(df, key, prefix, cols, top=8):
    """Contagem e proporção das `top` categorias mais frequentes de cada coluna, por `key`."""
    parts = []
    for c in cols:
        cats = df[c].value_counts().index[:top]
        dummies = pd.get_dummies(df[c].where(df[c].isin(cats)), prefix=f"{prefix}_{c}", dtype=np.float32)
        dummies[key] = df[key].to_numpy()
        g = dummies.groupby(key)
        parts += [g.sum().add_suffix("_n"), g.mean().add_suffix("_frac")]
    return pd.concat(parts, axis=1).astype(np.float32)


def application():
    app = read("application_train")
    app = app.set_index("SK_ID_CURR").drop(columns="TARGET")
    for c in app.columns:
        if not pd.api.types.is_numeric_dtype(app[c]):
            app[c] = app[c].astype("category").cat.codes.replace(-1, np.nan)
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].where(app["DAYS_EMPLOYED"] != 365243)
    app["CREDIT_INCOME"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
    app["ANNUITY_INCOME"] = app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"]
    app["CREDIT_TERM"] = app["AMT_ANNUITY"] / app["AMT_CREDIT"]
    app["GOODS_CREDIT"] = app["AMT_GOODS_PRICE"] / app["AMT_CREDIT"]
    app["EMPLOYED_BIRTH"] = app["DAYS_EMPLOYED"] / app["DAYS_BIRTH"]
    app["INCOME_PER_PERSON"] = app["AMT_INCOME_TOTAL"] / app["CNT_FAM_MEMBERS"]
    ext = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    app["EXT_MEAN"], app["EXT_STD"], app["EXT_PROD"] = ext.mean(axis=1), ext.std(axis=1), ext.prod(axis=1, min_count=3)
    return app.add_prefix("APP_")


def bureau():
    bb = read("bureau_balance")
    status = pd.get_dummies(bb["STATUS"], prefix="BB_STATUS", dtype=np.float32)
    status["SK_ID_BUREAU"] = bb["SK_ID_BUREAU"].to_numpy()
    per_credit = pd.concat([bb.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].agg(["size", "min", "max"])
                            .add_prefix("BB_MONTHS_"), status.groupby("SK_ID_BUREAU").mean()], axis=1)
    del bb, status
    b = read("bureau").join(per_credit, on="SK_ID_BUREAU")
    b["DEBT_CREDIT"] = b["AMT_CREDIT_SUM_DEBT"] / b["AMT_CREDIT_SUM"]
    b["OVERDUE_DEBT"] = b["AMT_CREDIT_SUM_OVERDUE"] / b["AMT_CREDIT_SUM_DEBT"]
    out = [num_agg(b, "SK_ID_CURR", "BUREAU"),
           cat_agg(b, "SK_ID_CURR", "BUREAU", ["CREDIT_ACTIVE", "CREDIT_TYPE", "CREDIT_CURRENCY"])]
    cols = ["DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
            "CREDIT_DAY_OVERDUE", "DEBT_CREDIT", "BB_MONTHS_size"]
    for state in ["Active", "Closed"]:
        out.append(num_agg(b[b["CREDIT_ACTIVE"] == state], "SK_ID_CURR", f"BUREAU_{state.upper()}", cols))
    return pd.concat(out, axis=1)


def previous():
    p = read("previous_application")
    p["APP_CREDIT"] = p["AMT_APPLICATION"] / p["AMT_CREDIT"]
    p["CREDIT_GOODS"] = p["AMT_CREDIT"] / p["AMT_GOODS_PRICE"]
    cats = ["NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS", "NAME_YIELD_GROUP", "CODE_REJECT_REASON",
            "NAME_PRODUCT_TYPE", "NAME_CLIENT_TYPE", "NAME_PAYMENT_TYPE", "PRODUCT_COMBINATION"]
    out = [num_agg(p, "SK_ID_CURR", "PREV"), cat_agg(p, "SK_ID_CURR", "PREV", cats)]
    cols = ["AMT_APPLICATION", "AMT_CREDIT", "APP_CREDIT", "AMT_ANNUITY", "DAYS_DECISION", "CNT_PAYMENT"]
    for state in ["Approved", "Refused"]:
        out.append(num_agg(p[p["NAME_CONTRACT_STATUS"] == state], "SK_ID_CURR", f"PREV_{state.upper()}", cols))
    return pd.concat(out, axis=1)


def pos_cash():
    pc = read("POS_CASH_balance")
    return pd.concat([num_agg(pc, "SK_ID_CURR", "POS"), cat_agg(pc, "SK_ID_CURR", "POS", ["NAME_CONTRACT_STATUS"])],
                     axis=1)


def installments():
    ins = read("installments_payments")
    ins["DAYS_LATE"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]  # > 0 = pagou atrasado
    ins["LATE"] = (ins["DAYS_LATE"] > 0).astype(np.float32)
    ins["PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["UNDERPAID"] = (ins["PAYMENT_DIFF"] > 0).astype(np.float32)
    return num_agg(ins, "SK_ID_CURR", "INST")


def credit_card():
    cc = read("credit_card_balance")
    cc["BALANCE_LIMIT"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]
    return pd.concat([num_agg(cc, "SK_ID_CURR", "CC"), cat_agg(cc, "SK_ID_CURR", "CC", ["NAME_CONTRACT_STATUS"])],
                     axis=1)


def build(max_rows=None):
    """DataFrame indexado por "id" (SK_ID_CURR em texto, como em clients.npy) e as features.
    max_rows (testes) limita só os clientes; as tabelas de histórico são lidas inteiras."""
    feats = application()
    if max_rows:
        feats = feats.head(max_rows)
    for part in [bureau, previous, pos_cash, installments, credit_card]:
        agg = part()
        feats = feats.join(agg, how="left")
        log(f"home_credit: {part.__name__} -> {agg.shape[1]} colunas (total {feats.shape[1]})")
    # razões entre o perfil e o histórico
    feats = feats.copy()
    inc = feats["APP_AMT_INCOME_TOTAL"]
    feats["BUREAU_DEBT_INCOME"] = feats["BUREAU_AMT_CREDIT_SUM_DEBT_sum"] / inc
    feats["INST_PAYMENT_INCOME"] = feats["INST_AMT_PAYMENT_mean"] / inc
    feats["PREV_CREDIT_APP_CREDIT"] = feats["PREV_AMT_CREDIT_mean"] / feats["APP_AMT_CREDIT"]
    feats = to_f32(feats.replace([np.inf, -np.inf], np.nan))
    feats.index = feats.index.astype(int).astype(str)
    return feats.rename_axis("id")
