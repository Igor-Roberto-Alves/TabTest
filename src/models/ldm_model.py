"""Large Data Model tabular (src/ldm): pré-treinado em várias tabelas e adaptado à tarefa.

Com checkpoint, carrega o modelo pré-treinado e faz `steps` passos de fine-tuning na
tabela da tarefa (steps=0 -> zero-shot). Sem checkpoint, treina do zero só nesta
tabela: é a ablação que mostra quanto o pré-treino ajuda.
"""

import warnings

import numpy as np

from src.data import Split
from src.models.base import Prediction

KINDS = {"binary", "multiclass", "regression"}


def fit_predict(split: Split, seed=0, checkpoint=None, steps=None, device=None) -> Prediction:
    import torch

    from src.ldm.model import LDMConfig
    from src.ldm.predict import adapt_and_predict
    from src.ldm.pretrain import build_diffusion, load_checkpoint

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if checkpoint:
        diffusion, ckpt = load_checkpoint(checkpoint, device)
        if split.task.dataset in ckpt["tables"]:
            warnings.warn(f"{split.task.dataset} estava no pré-treino: as linhas de teste vazaram")
        steps, lr = (300 if steps is None else steps), 1e-4
    else:
        diffusion = build_diffusion(LDMConfig(d_model=128, n_layers=4, n_heads=4))
        steps, lr = (2000 if steps is None else steps), 1e-3

    target = split.task.target
    y_train = split.y_train if split.is_regression else split.classes[split.y_train]
    train_df = split.X_train.assign(**{target: y_train})
    spec, probs, y_reg = adapt_and_predict(diffusion, train_df, split.X_test, target, split.task.kind,
                                           steps=steps, lr=lr, seed=seed, device=device)
    if split.is_regression:
        return Prediction(y_reg)
    # colunas de probs seguem spec.categories (ordem de frequência); reordena para split.classes
    proba = probs[:, [spec.categories.index(c) for c in split.classes]]
    proba = proba / proba.sum(1, keepdims=True)
    return Prediction(np.argmax(proba, axis=1), proba)
