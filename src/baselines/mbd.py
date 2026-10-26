"""Features do MBD (propensão): agregados mensais por cliente em janelas antes do mês do rótulo.

Primeiro cada modalidade vira um tensor denso cliente x mês x agregado (trx: contagem, soma e
média/desvio do log do valor, máximo, dias ativos, contagens e somas pelos event_type principais;
geo: contagem, dias ativos, geohashes distintos; diálogos: contagem, dias ativos). Com somas
prefixadas no eixo dos meses, cada linha (cliente, mês m) recebe as janelas do mês anterior, dos
últimos 3 meses e do histórico todo, mais os meses desde o último evento de cada modalidade.

Sem vazamento: a linha do mês m só usa meses < m, isto é, eventos com event_time antes do 1º dia
de m. É o mesmo corte do modelo fundacional (src/pragma/prepare.py, prepare_mbd). O trans_count
e o diff_trans_date da tabela de alvos descrevem o próprio mês do rótulo e não entram.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from .common import PREPARED, RAW, log

SRC = RAW / "mbd_mini" / "detail"
EPOCH = np.datetime64("2021-01-01", "D")    # origem do cutoff do store
DAY0 = np.datetime64("2020-12-01", "D")     # primeiro dia com eventos (o trx começa em 31/12/2020)
MONTH0 = np.datetime64("2020-12", "M")
N_MONTHS = 25                               # dez/2020 a dez/2022
TOP_TYPES = [1, 46, 40, 49, 11, 56, 41, 50, 18, 48]   # os 10 event_type mais frequentes (~94% das transações)
SUM_TYPES = TOP_TYPES[:5]


def read(modality, columns, value_set):
    """Lê uma modalidade em lotes -> dict de arrays compactos, com o índice do cliente e o dia."""
    out = {k: [] for k in ["cidx", "day", *columns]}
    dataset = ds.dataset(SRC / modality, format="parquet", partitioning="hive")
    for batch in dataset.to_batches(columns=["client_id", "event_time", *columns], batch_size=2_000_000):
        cidx = pc.index_in(batch.column("client_id"), value_set=value_set).fill_null(-1).to_numpy()
        keep = cidx >= 0
        ts = batch.column("event_time").to_numpy().astype("datetime64[D]")
        out["cidx"].append(cidx[keep].astype(np.int32))
        out["day"].append((ts[keep] - DAY0).astype(np.int16))
        for c in columns:
            out[c].append(batch.column(c).to_numpy(zero_copy_only=False)[keep])
    return {k: np.concatenate(v) for k, v in out.items()}


def month_of_day(day):
    return ((DAY0 + day.astype("timedelta64[D]")).astype("datetime64[M]") - MONTH0).astype(np.int64)


class Monthly:
    """Acumula agregados (N, N_MONTHS) por nome."""

    def __init__(self, n):
        self.n, self.cols = n, {}

    def key(self, cidx, day):
        return cidx.astype(np.int64) * N_MONTHS + month_of_day(day)

    def add(self, name, key, weights=None):
        self.cols[name] = np.bincount(key, weights, minlength=self.n * N_MONTHS).reshape(self.n, N_MONTHS)

    def add_days(self, name, cidx, day):
        """Dias distintos com evento no mês."""
        code = np.unique(cidx.astype(np.int64) * 1024 + day)
        self.add(name, self.key(code // 1024, code % 1024))

    def add_distinct(self, name, key, values):
        code = np.unique(key * (1 << 20) + values)
        self.add(name, code >> 20)


def monthly_tensor(n, value_set):
    m = Monthly(n)
    trx = read("trx", ["amount", "event_type"], value_set)
    key = m.key(trx["cidx"], trx["day"])
    la = np.log1p(trx["amount"].astype(np.float64))
    m.add("trx_n", key)
    m.add("trx_amount", key, trx["amount"].astype(np.float64))
    m.add("trx_log", key, la)
    m.add("trx_log2", key, la ** 2)
    mx = np.full(n * N_MONTHS, np.nan)
    np.fmax.at(mx, key, la)
    m.cols["trx_logmax"] = mx.reshape(n, N_MONTHS)
    m.add_days("trx_days", trx["cidx"], trx["day"])
    for t in TOP_TYPES:
        sel = trx["event_type"] == t
        m.add(f"trx_n_type{t}", key[sel])
        if t in SUM_TYPES:
            m.add(f"trx_amount_type{t}", key[sel], trx["amount"][sel].astype(np.float64))
    log(f"mbd: trx {len(key)} eventos")
    del trx, key, la

    geo = read("geo", ["geohash_5"], value_set)
    key = m.key(geo["cidx"], geo["day"])
    m.add("geo_n", key)
    m.add_days("geo_days", geo["cidx"], geo["day"])
    m.add_distinct("geo_cells", key, geo["geohash_5"].astype(np.int64))
    log(f"mbd: geo {len(key)} eventos")
    del geo, key

    dia = read("dialog", [], value_set)
    m.add("dialog_n", m.key(dia["cidx"], dia["day"]))
    m.add_days("dialog_days", dia["cidx"], dia["day"])
    log(f"mbd: dialog {len(dia['cidx'])} eventos")
    return m.cols


def windows(cols, cidx, L):
    """Features das linhas (cidx, mês L) com meses < L."""
    out = {}
    sums = [k for k in cols if k != "trx_logmax"]
    for k in sums:
        x = cols[k].astype(np.float64)
        pre = np.concatenate([np.zeros((len(x), 1)), np.cumsum(x, axis=1)], axis=1)
        out[f"{k}_m1"] = x[cidx, L - 1]
        out[f"{k}_m3"] = pre[cidx, L] - pre[cidx, np.maximum(L - 3, 0)]
        out[f"{k}_all"] = pre[cidx, L]
    mx = cols["trx_logmax"]
    out["trx_logmax_m1"] = mx[cidx, L - 1]
    out["trx_logmax_m3"] = np.fmax(np.fmax(mx[cidx, L - 1], mx[cidx, np.maximum(L - 2, 0)]), mx[cidx, np.maximum(L - 3, 0)])
    cummax = np.concatenate([np.full((len(mx), 1), np.nan), np.fmax.accumulate(mx, axis=1)], axis=1)
    out["trx_logmax_all"] = cummax[cidx, L]

    with np.errstate(divide="ignore", invalid="ignore"):
        for w in ["m1", "m3", "all"]:
            n = out[f"trx_n_{w}"]
            mean = out.pop(f"trx_log_{w}") / n
            out[f"trx_logmean_{w}"] = mean
            out[f"trx_logstd_{w}"] = np.sqrt(np.maximum(out.pop(f"trx_log2_{w}") / n - mean ** 2, 0))
            out[f"trx_amount_mean_{w}"] = out[f"trx_amount_{w}"] / n
        out["trx_n_m1_vs_m3"] = out["trx_n_m1"] / (out["trx_n_m3"] / 3)
        out["trx_amount_m1_vs_m3"] = out["trx_amount_m1"] / (out["trx_amount_m3"] / 3)

    months = np.arange(N_MONTHS)
    for mod in ["trx", "geo", "dialog"]:
        active = cols[f"{mod}_n"] > 0
        # último mês ativo antes de m (-1 = nenhum): prefixo do máximo de "índice do mês se ativo"
        last = np.maximum.accumulate(np.where(active, months, -1), axis=1)
        last = np.concatenate([np.full((len(active), 1), -1), last], axis=1)[cidx, L]
        out[f"{mod}_months_since_last"] = np.where(last >= 0, L - last, np.nan)
        pre = np.concatenate([np.zeros((len(active), 1)), np.cumsum(active, axis=1)], axis=1)
        out[f"{mod}_active_months_all"] = pre[cidx, L]
        out[f"{mod}_active_months_m3"] = pre[cidx, L] - pre[cidx, np.maximum(L - 3, 0)]
    first = np.argmax(cols["trx_n"] > 0, axis=1)
    has = (cols["trx_n"] > 0).any(axis=1)
    out["trx_months_since_first"] = np.where(has[cidx] & (first[cidx] < L), L - first[cidx], np.nan)
    out["label_month"] = ((MONTH0 + L.astype("timedelta64[M]")).astype(int) % 12 + 1).astype(np.float64)
    out["label_month_index"] = L.astype(np.float64)
    return pd.DataFrame({k: v.astype(np.float32) for k, v in out.items()})


def build(max_rows=None):
    """DataFrame indexado por (id, cutoff) das linhas de rótulo do store (max_rows: amostra de clientes, para testes)."""
    d = PREPARED / "mbd_mini"
    clients = np.load(d / "clients.npy")
    sel = np.arange(len(clients))
    if max_rows:
        sel = np.sort(np.random.default_rng(0).choice(len(clients), max_rows, replace=False))
    cols = monthly_tensor(len(sel), pa.array(clients[sel]))
    lab = pd.DataFrame({"cidx": np.load(d / "labels" / "client.npy"),
                        "cutoff": np.load(d / "labels" / "cutoff.npy")}).drop_duplicates()
    lab = lab[np.isin(lab["cidx"], sel)]
    local = np.searchsorted(sel, lab["cidx"].to_numpy())
    start = EPOCH + lab["cutoff"].to_numpy().astype(np.int64).astype("timedelta64[D]")
    if (start.astype("datetime64[M]").astype("datetime64[D]") != start).any():
        raise RuntimeError("cutoff do MBD fora do 1º dia do mês")
    L = (start.astype("datetime64[M]") - MONTH0).astype(np.int64)
    feats = windows(cols, local, L)
    feats.index = pd.MultiIndex.from_arrays([clients[lab["cidx"].to_numpy()], lab["cutoff"].to_numpy(np.float32)],
                                            names=["id", "cutoff"])
    log(f"mbd: {len(feats)} linhas (cliente, mês) x {feats.shape[1]} features")
    return feats
