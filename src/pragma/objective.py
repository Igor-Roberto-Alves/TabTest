"""Pré-treino por modelagem mascarada, com as três máscaras do PRAGMA.

    célula  cada célula de cada evento é mascarada com prob. p_cell
    evento  cada evento é mascarado inteiro com prob. p_event: todas as células e o vetor
            do evento no encoder de histórico (fica só o tempo e a fonte)
    tipo    cada (amostra, fonte, coluna) é mascarada em todos os eventos daquela amostra
            com prob. p_type: o modelo precisa inferir uma variável a partir das outras
    perfil  cada célula do perfil é mascarada com prob. p_profile

A loss é a cross-entropy nas células mascaradas (sobre as opções válidas da coluna,
[NULL] incluído), como no LDM, mais um MSE no valor contínuo das numéricas. Cada tipo de
máscara tem sua própria média e peso (w_*), e as métricas saem separadas para o TensorBoard.
Uma célula atingida por mais de uma máscara conta no tipo mais forte (evento > tipo > célula).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.ldm.table import LOCAL_NULL, NUM

CELL, TYPE, EVENT = 1, 2, 3
REASONS = {CELL: "cell", TYPE: "type", EVENT: "event"}


@dataclass
class MaskConfig:
    p_cell: float = 0.15
    p_event: float = 0.10
    p_type: float = 0.10
    p_profile: float = 0.15
    w_cell: float = 1.0
    w_event: float = 1.0
    w_type: float = 1.0
    w_profile: float = 1.0
    reg_weight: float = 1.0  # peso do MSE em z


def make_masks(batch, cfg: MaskConfig, generator=None):
    dev = batch["seq_valid"].device
    rand = lambda *shape: torch.rand(*shape, device=dev, generator=generator)
    B = batch["seq_valid"].shape[0]
    masks = {"cells": {}, "events": {}, "reason": {}}
    for name, sb in batch["src"].items():
        n, C = sb["local"].shape
        reason = torch.zeros(n, C, dtype=torch.int8, device=dev)
        reason[rand(n, C) < cfg.p_cell] = CELL
        type_mask = (rand(B, C) < cfg.p_type)[sb["b"]]
        reason[type_mask] = TYPE
        events = rand(n) < cfg.p_event
        reason[events] = EVENT
        masks["reason"][name] = reason
        masks["cells"][name] = reason > 0
        masks["events"][name] = events
    if "profile" in batch:
        masks["profile"] = rand(*batch["profile"]["local"].shape) < cfg.p_profile
    return masks


def _cell_terms(logits, z_pred, local, z, masked, kind):
    """CE e acerto por célula; MSE só nas numéricas mascaradas e não nulas."""
    n, C, K = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, K), local.reshape(-1), reduction="none").view(n, C)
    correct = (logits.argmax(-1) == local).float()
    num = (masked & (kind == NUM)[None, :] & (local != LOCAL_NULL)).float()
    sq = (z_pred - z) ** 2 * num
    return ce, correct, sq, num


def mlm_loss(model, batch, metas, cfg: MaskConfig, generator=None):
    masks = make_masks(batch, cfg, generator)
    out = model(batch, metas, masks, decode=True)
    dev = batch["seq_valid"].device
    zero = torch.zeros((), device=dev)
    sums = {r: {"ce": zero, "acc": zero, "n": zero} for r in REASONS}
    sq_sum, sq_n = zero, zero
    for name, (logits, z_pred) in out["cells"].items():
        sb, reason = batch["src"][name], masks["reason"][name]
        ce, correct, sq, num = _cell_terms(logits, z_pred, sb["local"], sb["z"], reason > 0,
                                           metas["sources"][name]["kind"])
        for r in REASONS:
            m = (reason == r).float()
            sums[r]["ce"] = sums[r]["ce"] + (ce * m).sum()
            sums[r]["acc"] = sums[r]["acc"] + (correct * m).sum()
            sums[r]["n"] = sums[r]["n"] + m.sum()
        sq_sum, sq_n = sq_sum + sq.sum(), sq_n + num.sum()

    stats, loss = {}, zero
    weights = {CELL: cfg.w_cell, TYPE: cfg.w_type, EVENT: cfg.w_event}
    for r, label in REASONS.items():
        n = sums[r]["n"].clamp(min=1)
        ce_r = sums[r]["ce"] / n
        loss = loss + weights[r] * ce_r
        stats[f"ce_{label}"] = ce_r.detach()
        stats[f"acc_{label}"] = (sums[r]["acc"] / n).detach()
    if "profile" in out:
        logits, z_pred = out["profile"]
        pm = masks["profile"]
        ce, correct, sq, num = _cell_terms(logits, z_pred, batch["profile"]["local"], batch["profile"]["z"], pm,
                                           metas["profile"]["kind"])
        n = pm.float().sum().clamp(min=1)
        ce_p = (ce * pm).sum() / n
        loss = loss + cfg.w_profile * ce_p
        stats["ce_profile"] = ce_p.detach()
        stats["acc_profile"] = ((correct * pm).sum() / n).detach()
        sq_sum, sq_n = sq_sum + sq.sum(), sq_n + num.sum()
    mse = sq_sum / sq_n.clamp(min=1)
    loss = loss + cfg.reg_weight * mse
    stats["mse"] = mse.detach()
    stats["loss"] = loss
    return stats
