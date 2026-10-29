"""Descoberta dos modelos: todo módulo de src/models com KINDS e fit_predict é um modelo.

Para criar um modelo novo, copie src/models/_template.py (ou `python -m src.models.new <nome>`) e
preencha. Nada mais precisa ser editado: run.py e o painel leem este registro.

Atributos opcionais do módulo (lidos aqui):
    NAME         nome na linha de comando e no painel (padrão: nome do arquivo)
    LABEL        nome para exibição no painel (padrão: NAME)
    DATA         "tabular" (padrão), "series" (séries temporais) ou "events" (sequência de eventos por cliente)
    INSTALL      pacote pip exigido, mostrado quando o import falha
    CAPABILITIES subconjunto de {"fit_predict", "pretrain", "finetune", "explain_mask"}
"""

import importlib
import pkgutil
from dataclasses import dataclass, field

import src.models as _pkg


@dataclass
class ModelInfo:
    name: str
    label: str
    module: str
    kinds: frozenset
    data: str = "tabular"
    install: str = ""
    capabilities: frozenset = field(default_factory=lambda: frozenset({"fit_predict"}))
    description: str = ""

    def load(self):
        return importlib.import_module(self.module)


def discover():
    models = {}
    for mod in pkgutil.iter_modules(_pkg.__path__):
        if mod.name.startswith("_") or mod.name in ("base", "registry", "new"):
            continue
        module_path = f"{_pkg.__name__}.{mod.name}"
        try:
            m = importlib.import_module(module_path)
        except ImportError:
            continue  # dependência ausente na importação do módulo; o run.py tenta de novo e avisa
        if not (hasattr(m, "KINDS") and hasattr(m, "fit_predict")):
            continue
        name = getattr(m, "NAME", mod.name)
        doc = (m.__doc__ or "").strip().splitlines()
        models[name] = ModelInfo(
            name=name, label=getattr(m, "LABEL", name), module=module_path, kinds=frozenset(m.KINDS),
            data=getattr(m, "DATA", "tabular"), install=getattr(m, "INSTALL", ""),
            capabilities=frozenset(getattr(m, "CAPABILITIES", {"fit_predict"})),
            description=doc[0] if doc else "")
    return dict(sorted(models.items()))


MODELS = discover()


def get(name):
    if name not in MODELS:
        raise KeyError(f"modelo '{name}' não registrado; disponíveis: {', '.join(MODELS)}")
    return MODELS[name]
