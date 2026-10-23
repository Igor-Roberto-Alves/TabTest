"""Masked diffusion (estado absorvente, estilo MDLM) sobre as células de uma linha.

Forward:  cada célula é mascarada com prob. 1 - alpha(t), t ~ U(t_min, 1).
Loss:     cross-entropy só nas células mascaradas (sobre as opções válidas da coluna,
          incluindo [NULL]). Com weighting="elbo", cada linha é ponderada por
          w(t) = -alpha'(t) / (1 - alpha(t)) (limite contínuo do ELBO do MDLM); com
          "uniform", é a média simples. Numéricas mascaradas somam um MSE em z.
Reverse:  parte de tudo mascarado nas células desconhecidas e, de t=1 até t=0, revela
          cada célula mascarada com prob. (alpha(s) - alpha(t)) / (1 - alpha(t)), s = t - 1/steps.

O modelo é condicionado na fração de máscara observada da linha, não no t sorteado,
para que treino e imputação (com parte da linha conhecida) vejam a mesma grandeza.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .table import LOCAL_NULL, NUM, Z_CLIP


class MaskSchedule:
    """alpha(t) = probabilidade de uma célula continuar visível no instante t."""

    def __init__(self, kind="cosine"):
        if kind not in ("linear", "cosine"):
            raise ValueError(f"schedule desconhecido: {kind}")
        self.kind = kind

    def alpha(self, t):
        return 1 - t if self.kind == "linear" else torch.cos(math.pi / 2 * t)

    def weight(self, t):
        """-alpha'(t) / (1 - alpha(t))."""
        if self.kind == "linear":
            return 1 / t
        return (math.pi / 2) * torch.sin(math.pi / 2 * t) / (1 - torch.cos(math.pi / 2 * t))


class MaskedDiffusion(nn.Module):
    def __init__(self, model, schedule="cosine", weighting="elbo", t_min=1e-3, max_weight=1e3, reg_weight=1.0):
        super().__init__()
        self.model = model
        self.schedule = MaskSchedule(schedule)
        self.weighting = weighting
        self.t_min = t_min
        self.max_weight = max_weight
        self.reg_weight = reg_weight

    # ------------------------------------------------------------- treino
    def loss(self, local, z, meta, force_mask=None, weighting=None):
        """force_mask (N, C) bool: células sempre mascaradas (ex.: o alvo no fine-tuning)."""
        N, C = local.shape
        weighting = weighting or self.weighting
        t = self.t_min + (1 - self.t_min) * torch.rand(N, device=local.device)
        masked = torch.rand(N, C, device=local.device) < (1 - self.schedule.alpha(t))[:, None]
        if force_mask is not None:
            masked = masked | force_mask
        out = self.model(local, z, masked, masked.float().mean(1), meta)

        ce = F.cross_entropy(out["logits"].flatten(0, 1), local.flatten(), reduction="none").view(N, C)
        m = masked.float()
        ce_mean = (ce * m).sum() / m.sum().clamp(min=1)
        if weighting == "elbo":
            w = self.schedule.weight(t).clamp(max=self.max_weight)
            loss = ((ce * m).sum(1) * w / C).mean()
        else:
            loss = ce_mean

        rm = (masked & (meta["kind"] == NUM)[None, :] & (local != LOCAL_NULL)).float()
        mse = ((out["z_pred"] - z) ** 2 * rm).sum() / rm.sum().clamp(min=1)
        loss = loss + self.reg_weight * mse
        return {"loss": loss, "ce_masked": ce_mean.detach(), "mse_masked": mse.detach()}

    # ---------------------------------------------------------- amostragem
    @staticmethod
    def _ban_null(logits, allow_null):
        if allow_null:
            return logits
        logits = logits.clone()
        logits[..., LOCAL_NULL] = float("-inf")
        return logits

    @torch.no_grad()
    def impute(self, local, z, known, meta, steps=None, temperature=1.0, strategy="random", allow_null=False):
        """Completa as células com known=False, desmascarando progressivamente.

        steps:       passos reversos (padrão: C)
        temperature: 0 = argmax; > 0 = amostra da categórica com essa temperatura
        strategy:    "random" (MDLM: revela com a prob. do schedule) ou
                     "confidence" (revela as k células mais confiantes, k pelo schedule)
        Numéricas reveladas recebem como z a mediana padronizada do bin sorteado.
        """
        self.model.eval()
        N, C = local.shape
        masked = ~known
        local = local.masked_fill(masked, 0)
        z = z.masked_fill(masked, 0.0)
        numeric = (meta["kind"] == NUM)[None, :]
        cols = torch.arange(C, device=local.device)[None, :]
        steps = steps or C
        sched = self.schedule

        for i in range(steps, 0, -1):
            if not masked.any():
                break
            t = torch.full((N,), i / steps, device=local.device)
            s = torch.full((N,), (i - 1) / steps, device=local.device)
            out = self.model(local, z, masked, masked.float().mean(1), meta)
            logits = self._ban_null(out["logits"], allow_null)
            if temperature == 0:
                pred = logits.argmax(-1)
            else:
                pred = torch.distributions.Categorical(logits=logits / temperature).sample()
            conf = logits.softmax(-1).gather(-1, pred.unsqueeze(-1)).squeeze(-1)

            p = ((sched.alpha(s) - sched.alpha(t)) / (1 - sched.alpha(t))).clamp(0, 1)
            if strategy == "random":
                reveal = masked & (torch.rand(N, C, device=local.device) < p[:, None])
            else:
                k = torch.ceil(masked.sum(1) * p).long()
                rank = conf.masked_fill(~masked, -1).argsort(1, descending=True).argsort(1)
                reveal = masked & (rank < k[:, None])

            local = torch.where(reveal, pred, local)
            bin_z = meta["opt_z"][cols, (pred - 1).clamp(min=0)]
            z = torch.where(reveal & numeric & (pred != LOCAL_NULL), bin_z, z)
            masked = masked & ~reveal
        return local, z

    @torch.no_grad()
    def generate(self, n, meta, steps=None, temperature=1.0, strategy="random", allow_null=True):
        """Gera n linhas sintéticas de uma tabela partindo de tudo mascarado."""
        C = meta["kind"].shape[0]
        dev = meta["kind"].device
        local = torch.zeros(n, C, dtype=torch.long, device=dev)
        z = torch.zeros(n, C, device=dev)
        known = torch.zeros(n, C, dtype=torch.bool, device=dev)
        return self.impute(local, z, known, meta, steps, temperature, strategy, allow_null)

    @torch.no_grad()
    def predict(self, local, z, meta, target_cols, allow_null=False):
        """Mascara só as colunas alvo: p(alvo | resto) em um forward.

        Retorna (probs (N, T, K+1) sobre índices locais, z_pred (N, T)).
        """
        self.model.eval()
        masked = torch.zeros_like(local, dtype=torch.bool)
        masked[:, target_cols] = True
        out = self.model(local, z, masked, masked.float().mean(1), meta)
        logits = self._ban_null(out["logits"][:, target_cols], allow_null)
        return logits.softmax(-1), out["z_pred"][:, target_cols].clamp(-Z_CLIP, Z_CLIP)
