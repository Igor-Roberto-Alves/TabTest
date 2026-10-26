"""Texto -> ids de hash compartilhados entre todas as tabelas (sem vocabulário fixo).

Nomes de colunas e valores categóricos viram um saco de palavras + trigramas de
caracteres, e cada pedaço é mapeado por hash para um de N_BUCKETS ids. Uma tabela
nunca vista cai no mesmo espaço de embedding: é isso que permite pré-treinar um
único modelo em muitas tabelas e aplicá-lo a outras.

    "MedHouseVal" -> ["w:med", "w:house", "w:val", "c:<me", "c:med", "c:ed>", ...]
"""

import re
import zlib

import numpy as np

N_BUCKETS = 1 << 16  # id 0 é reservado para padding
MAX_PIECES = 32

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT = re.compile(r"[^0-9a-z]+")
_SYMBOLS = re.compile(r"[^0-9A-Za-z\s]+")


def pieces(text) -> list[str]:
    words = [w for w in _SPLIT.split(_CAMEL.sub(" ", str(text)).lower()) if w]
    # símbolos também viram pedaços: sem eles "<=50K" e ">50K" (ou "-" e "+") teriam o mesmo hash
    out = [f"w:{w}" for w in words] + [f"s:{s}" for s in _SYMBOLS.findall(str(text))]
    for w in words:
        padded = f"<{w}>"
        out += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
    return out[:MAX_PIECES] or ["w:<empty>"]


def hash_ids(text) -> list[int]:
    # crc32 em vez de hash(): estável entre processos (PYTHONHASHSEED)
    return [1 + zlib.crc32(p.encode()) % (N_BUCKETS - 1) for p in pieces(text)]


def encode_texts(texts) -> np.ndarray:
    """Lista de textos -> (n, MAX_PIECES) int64, preenchido com 0."""
    out = np.zeros((len(texts), MAX_PIECES), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = hash_ids(text)
        out[i, :len(ids)] = ids
    return out
