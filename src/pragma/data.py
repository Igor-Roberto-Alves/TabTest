"""Amostras (cliente, corte no tempo) -> lotes com a sequência de eventos de cada cliente.

Uma amostra lê, de cada fonte, os eventos do cliente com t < corte e fica com os
max_events mais recentes daquela fonte (o limite é por fonte para que uma fonte muito
densa, como bureau_balance ou geo, não expulse as outras). No lote:

    src[fonte]   local (n, C), z (n, C): células dos eventos daquela fonte no lote
                 b (n,): amostra de cada evento; pos (n,): posição na sequência da amostra
    seq_*        (B, L): sequência de cada amostra, eventos de todas as fontes por ordem de t
                 seq_src (índice da fonte), seq_age (dias até a referência), seq_dt (dias desde
                 o evento anterior), seq_cal (B, L, 3), seq_valid (padding = False)
    profile      local (B, Cp), z (B, Cp), se a base tem perfil
    y            (B, k) rótulos, só em amostras rotuladas

A referência de tempo é o corte (se finito) ou o último evento da amostra.
"""

from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .store import Store


def max_events_for(cfg_max_events, source):
    if isinstance(cfg_max_events, int):
        return cfg_max_events
    return cfg_max_events.get(source, cfg_max_events.get("default", 64))


class EventDataset(Dataset):
    """clients (M,), cutoffs (M,) e, opcionalmente, y (M, k)."""

    def __init__(self, store: Store, clients, cutoffs=None, y=None, max_events=64, time_cutoff=None):
        self.store = store
        self.clients = np.asarray(clients)
        self.cutoffs = np.full(len(self.clients), np.inf) if cutoffs is None else np.asarray(cutoffs, dtype=np.float64)
        if time_cutoff is not None:  # ex.: pré-treino só com o passado de uma data
            self.cutoffs = np.minimum(self.cutoffs, time_cutoff)
        self.y = None if y is None else np.asarray(y, dtype=np.float32)
        self.limits = {s: max_events_for(max_events, s) for s in store.sources}

    @classmethod
    def pretrain(cls, store, splits, **kw):
        return cls(store, store.clients_in(splits), **kw)

    @classmethod
    def labelled(cls, store, split, rows=None, **kw):
        rows = store.label_rows(split) if rows is None else rows
        return cls(store, store.label_client[rows], store.label_cutoff[rows], store.label_y[rows], **kw)

    def __len__(self):
        return len(self.clients)

    def __getitem__(self, i):
        c, cut = int(self.clients[i]), self.cutoffs[i]
        item = {"cutoff": cut, "events": {}}
        for name, src in self.store.sources.items():
            a, b = int(src.offsets[c]), int(src.offsets[c + 1])
            if np.isfinite(cut) and b > a:
                b = a + int(np.searchsorted(src.t[a:b], cut, side="left"))  # t ordenado no cliente
            a = max(a, b - self.limits[name])
            item["events"][name] = (np.asarray(src.local[a:b]), np.asarray(src.z[a:b]),
                                    np.asarray(src.t[a:b]), np.asarray(src.cal[a:b]))
        if self.store.profile is not None:
            item["profile"] = (np.asarray(self.store.profile.local[c]), np.asarray(self.store.profile.z[c]))
        if self.y is not None:
            item["y"] = self.y[i]
        return item


def collate(items, sources):
    B = len(items)
    batch = {"src": {}}
    ev_b, ev_t, ev_src, ev_row, ev_cal = [], [], [], [], []
    for k, name in enumerate(sources):
        local = [it["events"][name][0] for it in items]
        n_per = [len(x) for x in local]
        batch["src"][name] = {"local": torch.from_numpy(np.concatenate(local).astype(np.int64)),
                              "z": torch.from_numpy(np.concatenate([it["events"][name][1] for it in items])
                                                    .astype(np.float32))}
        ev_b.append(np.repeat(np.arange(B), n_per))
        ev_t.append(np.concatenate([it["events"][name][2] for it in items]).astype(np.float64))
        ev_cal.append(np.concatenate([it["events"][name][3] for it in items]))
        ev_src.append(np.full(sum(n_per), k))
        ev_row.append(np.arange(sum(n_per)))
    ev_b, ev_t, ev_src = np.concatenate(ev_b), np.concatenate(ev_t), np.concatenate(ev_src)
    ev_row, ev_cal = np.concatenate(ev_row), np.concatenate(ev_cal)

    # ordem na sequência: por amostra, depois por tempo (estável entre fontes)
    order = np.lexsort((ev_t, ev_b))
    counts = np.bincount(ev_b, minlength=B)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order)) - starts[ev_b[order]]
    L = max(1, int(counts.max()) if B else 1)

    cut = np.array([it["cutoff"] for it in items], dtype=np.float64)
    last = np.full(B, -np.inf)
    if len(ev_t):
        np.maximum.at(last, ev_b, ev_t)
    ref = np.where(np.isfinite(cut), cut, np.where(np.isfinite(last), last, 0.0))

    seq_t = np.zeros((B, L))
    seq_src = np.zeros((B, L), dtype=np.int64)
    seq_cal = np.full((B, L, 3), -1, dtype=np.int64)
    valid = np.zeros((B, L), dtype=bool)
    seq_t[ev_b, pos], seq_src[ev_b, pos], seq_cal[ev_b, pos], valid[ev_b, pos] = ev_t, ev_src, ev_cal, True
    age = np.clip(ref[:, None] - seq_t, 0, None)
    dt = np.diff(seq_t, axis=1, prepend=seq_t[:, :1])
    dt = np.where(valid, np.clip(dt, 0, None), 0)

    for k, name in enumerate(sources):
        sel = ev_src == k
        idx = np.empty(sel.sum(), dtype=np.int64)
        idx[ev_row[sel]] = np.nonzero(sel)[0]
        batch["src"][name]["b"] = torch.from_numpy(ev_b[idx])
        batch["src"][name]["pos"] = torch.from_numpy(pos[idx])
    batch.update(seq_src=torch.from_numpy(seq_src), seq_age=torch.from_numpy(age.astype(np.float32)),
                 seq_dt=torch.from_numpy(dt.astype(np.float32)), seq_cal=torch.from_numpy(seq_cal),
                 seq_valid=torch.from_numpy(valid))
    if "profile" in items[0]:
        batch["profile"] = {"local": torch.from_numpy(np.stack([it["profile"][0] for it in items]).astype(np.int64)),
                            "z": torch.from_numpy(np.stack([it["profile"][1] for it in items]).astype(np.float32))}
    if "y" in items[0]:
        batch["y"] = torch.from_numpy(np.stack([it["y"] for it in items]))
    return batch


def make_loader(dataset: EventDataset, batch_size, shuffle=True, infinite=False, num_workers=4, seed=0):
    sampler = None
    if infinite:
        g = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(dataset, replacement=True, num_samples=10**12, generator=g)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      collate_fn=partial(collate, sources=list(dataset.store.sources)),
                      persistent_workers=num_workers > 0 and infinite, pin_memory=torch.cuda.is_available(),
                      drop_last=infinite)


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True) if torch.is_tensor(batch) else batch
