"""Large Data Model tabular: um Transformer pré-treinado em muitas tabelas com difusão mascarada."""

from .table import CAT, LOCAL_NULL, NUM, OTHER, ColumnSpec, TableEncoder

# torch é opcional na importação do pacote: o codificador funciona só com pandas/numpy
try:
    from .corpus import EncodedTable, TableCorpus
    from .diffusion import MaskedDiffusion, MaskSchedule
    from .model import LDMConfig, TabularLDM
except ImportError:  # pragma: no cover
    pass

__all__ = ["TableEncoder", "ColumnSpec", "NUM", "CAT", "LOCAL_NULL", "OTHER",
           "TabularLDM", "LDMConfig", "MaskedDiffusion", "MaskSchedule", "TableCorpus", "EncodedTable"]
