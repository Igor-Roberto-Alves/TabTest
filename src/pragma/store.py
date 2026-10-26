"""Formato comum das bases de eventos, em data/banking/prepared/<base>/.

    meta.json                  nome, tarefa, fontes e encoders (TableEncoder.to_dict)
    clients.npy   (N,)         id original de cada cliente
    split.npy     (N,) int8    0 treino, 1 validação, 2 teste, 3 sem rótulo
    profile/local.npy (N, Cp)  perfil estático codificado (índices locais), se houver
    profile/z.npy     (N, Cp)  valor contínuo padronizado das numéricas do perfil
    sources/<fonte>/
        local.npy   (E, C) int16   células de cada evento
        z.npy       (E, C) float16
        t.npy       (E,)   float32 tempo do evento em dias (a origem depende da base)
        cal.npy     (E, 3) int8    hora, dia da semana, dia do mês (-1 = desconhecido)
        offsets.npy (N+1,) int64   eventos do cliente i em [offsets[i], offsets[i+1]),
                                   ordenados por t
    labels/
        client.npy  (R,) int32     cliente de cada linha de rótulo
        cutoff.npy  (R,) float32   só eventos com t < cutoff entram (inf = histórico todo)
        y.npy       (R, k) float32 rótulos (k > 1 = multirrótulo)

Os arrays grandes são abertos com mmap: o processo de treino lê só os eventos do lote.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.ldm.table import TableEncoder

SPLITS = {"train": 0, "val": 1, "test": 2, "unlabeled": 3}
NO_CAL = np.full(3, -1, dtype=np.int8)


# ------------------------------------------------------------------- escrita
def save_source(out, name, encoder, local, z, cidx, t, cal, n_clients):
    """Ordena os eventos por (cliente, t) e salva. cidx: índice do cliente de cada evento."""
    order = np.lexsort((t, cidx))
    d = Path(out) / "sources" / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "local.npy", local[order].astype(np.int16))
    np.save(d / "z.npy", z[order].astype(np.float16))
    np.save(d / "t.npy", t[order].astype(np.float32))
    np.save(d / "cal.npy", cal[order].astype(np.int8))
    counts = np.bincount(cidx, minlength=n_clients)
    np.save(d / "offsets.npy", np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
    return {"encoder": encoder.to_dict(), "n_events": int(len(order))}


def save_store(out, name, clients, split, sources, task, labels, profile=None, notes=""):
    """sources: {nome: retorno de save_source}; labels: (client, cutoff, y);
    profile: (encoder, local, z) ou None; task: {"name", "labels", "type", "metric"}."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "clients.npy", np.asarray(clients).astype(str))
    np.save(out / "split.npy", np.asarray(split, dtype=np.int8))
    meta = {"name": name, "n_clients": len(clients), "task": task, "sources": sources, "profile": None,
            "notes": notes}
    if profile is not None:
        encoder, local, z = profile
        (out / "profile").mkdir(exist_ok=True)
        np.save(out / "profile" / "local.npy", local.astype(np.int16))
        np.save(out / "profile" / "z.npy", z.astype(np.float16))
        meta["profile"] = {"encoder": encoder.to_dict()}
    client, cutoff, y = labels
    (out / "labels").mkdir(exist_ok=True)
    np.save(out / "labels" / "client.npy", np.asarray(client, dtype=np.int32))
    np.save(out / "labels" / "cutoff.npy", np.asarray(cutoff, dtype=np.float32))
    np.save(out / "labels" / "y.npy", np.asarray(y, dtype=np.float32).reshape(len(client), -1))
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f)


# ------------------------------------------------------------------- leitura
@dataclass
class Source:
    name: str
    encoder: TableEncoder
    local: np.ndarray
    z: np.ndarray
    t: np.ndarray
    cal: np.ndarray
    offsets: np.ndarray

    @property
    def n_columns(self):
        return self.local.shape[1]


class Store:
    def __init__(self, path, mmap=True):
        self._open(Path(path), mmap)

    # os workers do DataLoader (forkserver no Python 3.14) recebem o Store serializado: só o
    # caminho viaja e cada worker reabre os arrays com mmap, em vez de copiar todos os eventos
    def __getstate__(self):
        return {"path": self.path, "mmap": self.mmap}

    def __setstate__(self, state):
        self._open(state["path"], state["mmap"])

    def _open(self, path, mmap):
        self.path, self.mmap = path, mmap
        mode = "r" if mmap else None
        with open(self.path / "meta.json") as f:
            self.meta = json.load(f)
        self.name = self.meta["name"]
        self.task = self.meta["task"]
        self.clients = np.load(self.path / "clients.npy")
        self.split = np.load(self.path / "split.npy")
        self.sources = {}
        for name, info in self.meta["sources"].items():
            d = self.path / "sources" / name
            self.sources[name] = Source(name, TableEncoder.from_dict(info["encoder"]),
                                        np.load(d / "local.npy", mmap_mode=mode), np.load(d / "z.npy", mmap_mode=mode),
                                        np.load(d / "t.npy", mmap_mode=mode), np.load(d / "cal.npy", mmap_mode=mode),
                                        np.load(d / "offsets.npy"))
        self.profile = None
        if self.meta["profile"] is not None:
            self.profile = Source("profile", TableEncoder.from_dict(self.meta["profile"]["encoder"]),
                                  np.load(self.path / "profile" / "local.npy", mmap_mode=mode),
                                  np.load(self.path / "profile" / "z.npy", mmap_mode=mode), None, None, None)
        self.label_client = np.load(self.path / "labels" / "client.npy")
        self.label_cutoff = np.load(self.path / "labels" / "cutoff.npy")
        self.label_y = np.load(self.path / "labels" / "y.npy")

    @property
    def n_clients(self):
        return len(self.clients)

    def clients_in(self, splits):
        codes = [SPLITS[s] for s in splits]
        return np.nonzero(np.isin(self.split, codes))[0]

    def label_rows(self, split):
        """Linhas de rótulo cujos clientes estão no split."""
        return np.nonzero(self.split[self.label_client] == SPLITS[split])[0]

    def summary(self):
        lines = [f"{self.name}: {self.n_clients} clientes, tarefa {self.task['name']} ({self.task['type']}, "
                 f"{len(self.label_client)} linhas de rótulo)"]
        if self.profile is not None:
            lines.append(f"  perfil: {self.profile.n_columns} colunas")
        for s in self.sources.values():
            lines.append(f"  {s.name:24s} {len(s.t):>11} eventos x {s.n_columns} colunas")
        return "\n".join(lines)
