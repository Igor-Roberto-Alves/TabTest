"""Registro dos treinos no TensorBoard, todos sob um único diretório raiz (runs/ por padrão).

Layout (cada pasta folha é um "run" no TensorBoard):

    runs/
      <experimento>/<tarefa>/<modelo>/seed<k>/   # python -m src.run
      pretrain/<nome>/                            # python -m src.ldm.pretrain

Abra tudo lado a lado com:  tensorboard --logdir runs
- Scalars: curvas de treino (XGBoost por árvore, LDM por passo) e métricas finais em test/*.
- HParams: uma linha por (tarefa, modelo, seed) com as métricas de teste; filtre e ordene ali.
- Text: descrição da tarefa e do split.
"""

import math
import re
import warnings
from datetime import datetime
from pathlib import Path

from src.tasks import ROOT

DEFAULT_LOGDIR = ROOT / "runs"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _slug(text: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(text)).strip("_") or "run"


def open_writer(logdir: str | Path, *parts: str):
    """SummaryWriter em logdir/parts...; None (com aviso) se o tensorboard não estiver instalado."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:  # torch ou tensorboard ausentes
        warnings.warn(f"TensorBoard desativado ({e}); pip install tensorboard")
        return None
    path = Path(logdir).joinpath(*map(_slug, parts))
    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(path))


def _hparam_value(v):
    if isinstance(v, bool | int | float | str):
        return v
    return str(v)


def log_result(writer, hparams: dict, metrics: dict, text: str | None = None):
    """Grava métricas finais como scalars e a linha da aba HParams no próprio run.

    Escreve as summaries de hparams no writer do run (em vez de writer.add_hparams, que cria
    uma subpasta extra) para que curvas e métricas fiquem na mesma pasta.
    """
    if writer is None:
        return
    from torch.utils.tensorboard.summary import hparams as hparams_summary

    metrics = {k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))}
    exp, ssi, sei = hparams_summary({k: _hparam_value(v) for k, v in hparams.items()}, metrics)
    for summary in (exp, ssi, sei):
        writer.file_writer.add_summary(summary)
    for k, v in metrics.items():
        writer.add_scalar(k, v, 0)
    if text:
        writer.add_text("info", text, 0)
    writer.flush()
