"""Configs YAML (configs/pragma/*.yaml) com sobrescritas pela linha de comando.

    --config configs/pragma/pretrain.yaml --set optim.lr=1e-3 model.d_model=128 data.datasets=[amex]

O valor depois de "=" é lido como YAML: números, listas e dicionários funcionam.
"""

import copy
from pathlib import Path

import yaml


def set_key(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"sobrescrita sem '=': {item}")
        set_key(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def flatten(cfg, prefix=""):
    """{"a": {"b": 1}} -> {"a.b": 1}; listas e None viram texto (o HParams do TensorBoard só aceita escalares)."""
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (list, tuple)) or v is None:
            out[key] = str(v)
        else:
            out[key] = v
    return out


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False, allow_unicode=True)
