"""Protocolo de datasets: cada arquivo YAML em tasks/ descreve um dataset e suas tarefas.

Um spec diz onde estão os arquivos, quais colunas não são features e, para cada
tarefa, o tipo de problema e a variável alvo (coluna existente ou expressão).
Formato completo e exemplos em tasks/README.md.

    dataset: adult
    files: [data/adult.csv]
    tasks:
      - name: adult_income
        kind: binary
        target: class

Chaves de nível dataset (drop, categorical, datetime, filter, split, max_rows,
features) valem para todas as tarefas; a tarefa pode sobrescrevê-las. `drop` e
`categorical` da tarefa somam às do dataset.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

import yaml

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
KINDS = ("binary", "multiclass", "regression")
SPLIT_METHODS = ("random", "temporal", "group")

Kind = Literal["binary", "multiclass", "regression"]


@dataclass(frozen=True)
class SplitSpec:
    method: str = "random"      # random (estratificado na classificação), temporal ou group
    test_size: float = 0.2
    column: str | None = None   # temporal: coluna de data/ordem; group: coluna de grupo
    cutoff: str | None = None   # temporal: linhas com column >= cutoff vão para teste

    def __post_init__(self):
        if self.method not in SPLIT_METHODS:
            raise ValueError(f"split.method deve ser um de {SPLIT_METHODS}, veio {self.method!r}")
        if self.method != "random" and not self.column:
            raise ValueError(f"split.method={self.method} exige split.column")


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str                   # nome do dataset (também usado na checagem de vazamento do LDM)
    files: tuple[str, ...]         # caminhos/globs relativos à raiz do repo; .csv ou .parquet
    target: str                    # nome da coluna alvo (criada por `expr` se ela existir)
    kind: Kind
    description: str = ""
    expr: str | None = None        # expressão pandas (DataFrame.eval) que cria o alvo
    filter: str | None = None      # DataFrame.query aplicada antes de tudo
    drop: tuple[str, ...] = ()     # colunas fora das features (ids, vazamento)
    features: tuple[str, ...] | None = None  # se dado, usa só estas colunas como features
    categorical: tuple[str, ...] = ()        # força colunas numéricas (códigos) a categóricas
    datetime: tuple[str, ...] = ()           # colunas de data: viram ano/mês/dia/dia-da-semana/hora
    split: SplitSpec = field(default_factory=SplitSpec)
    max_rows: int | None = None    # subamostra datasets grandes
    read: dict = field(default_factory=dict, hash=False, compare=False)  # kwargs de pd.read_csv
    source: str = ""               # arquivo YAML de origem

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind deve ser um de {KINDS}, veio {self.kind!r}")


_DATASET_KEYS = {"dataset", "description", "files", "tasks", "read", "drop", "categorical",
                 "datetime", "filter", "split", "max_rows", "features"}
_TASK_KEYS = {f.name for f in fields(Task)} - {"dataset", "files", "read", "source"}
_ADDITIVE = ("drop", "categorical", "datetime")


def _tuple(value):
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def load_spec(path: str | Path) -> list[Task]:
    """Lê um YAML de dataset e devolve suas tarefas."""
    path = Path(path)
    spec = yaml.safe_load(path.read_text()) or {}
    where = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path

    unknown = set(spec) - _DATASET_KEYS
    if unknown:
        raise ValueError(f"{where}: chaves desconhecidas {sorted(unknown)}")
    for key in ("dataset", "files", "tasks"):
        if not spec.get(key):
            raise ValueError(f"{where}: falta a chave obrigatória '{key}'")

    tasks = []
    for raw in spec["tasks"]:
        unknown = set(raw) - _TASK_KEYS
        if unknown:
            raise ValueError(f"{where}: tarefa {raw.get('name')}: chaves desconhecidas {sorted(unknown)}")
        merged = {k: spec[k] for k in ("filter", "max_rows", "features") if k in spec}
        merged.update({k: v for k, v in raw.items() if k not in _ADDITIVE and k != "split"})
        for key in _ADDITIVE:
            merged[key] = tuple(dict.fromkeys(_tuple(spec.get(key)) + _tuple(raw.get(key))))
        if merged.get("features") is not None:
            merged["features"] = _tuple(merged["features"])
        merged["split"] = SplitSpec(**{**(spec.get("split") or {}), **(raw.get("split") or {})})
        tasks.append(Task(dataset=spec["dataset"], files=_tuple(spec["files"]), read=spec.get("read") or {},
                          source=str(where), **merged))
    return tasks


def discover(extra_specs=(), tasks_dir: Path = TASKS_DIR) -> dict[str, Task]:
    """Todas as tarefas de tasks/**/*.yaml mais as dos specs extras. Nomes devem ser únicos."""
    paths = sorted(tasks_dir.rglob("*.yaml")) if tasks_dir.exists() else []
    paths += [Path(p).resolve() for p in extra_specs if Path(p).resolve() not in paths]
    registry: dict[str, Task] = {}
    for path in paths:
        for task in load_spec(path):
            if task.name in registry and registry[task.name].source != task.source:
                raise ValueError(f"tarefa '{task.name}' definida em {registry[task.name].source} e {task.source}")
            registry[task.name] = task
    return registry
