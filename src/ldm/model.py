"""Large Data Model tabular: um único Transformer para qualquer tabela.

Nenhum parâmetro depende de uma tabela específica:
    coluna c      -> E_col = TextEmb(nome de c) + E_tipo[num/cat]
    opção k de c  -> o_ck = TextEmb(texto da categoria)           se c é categórica
                            MLP(fourier(q_ck), fourier(z_ck))       se c é numérica
                     o_c0 = e_[NULL] (aprendido)

Entrada da célula (n, c) com índice local l:
    h = W_in o_cl (+ W_z fourier(z) se numérica visível)  ou  e_[MASK] se mascarada
        + E_col[c] + MLP(senoidal(r)),   r = fração de células mascaradas na linha
Sem embedding posicional: a ordem das colunas não importa (atenção é equivariante a permutações).

Head de autocomplete: logit_nck = <W_q h_nc, W_out o_ck> / sqrt(d), só sobre as opções válidas
da coluna. Como as opções vêm de texto/quantis, a mesma head serve para qualquer coluna.
Head de regressão: z_nc = w^T h_nc + b (valor contínuo padronizado das numéricas).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .table import CAT, NUM, Z_CLIP
from .text import N_BUCKETS


@dataclass
class LDMConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    n_buckets: int = N_BUCKETS
    n_fourier: int = 8


def timestep_embedding(t, dim, max_period=10_000):
    """Embedding senoidal de t em [0, 1] (escalado para [0, 1000], como no DDPM)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000)[:, None] * freqs[None, :]
    return F.pad(torch.cat([torch.cos(args), torch.sin(args)], dim=-1), (0, dim % 2))


def fourier(x, n):
    """[x, sin(2^k π x), cos(2^k π x)]_{k<n}: escalar em ~[-1, 1] -> 2n+1 features."""
    args = x.unsqueeze(-1) * (2.0 ** torch.arange(n, device=x.device)) * math.pi
    return torch.cat([x.unsqueeze(-1), torch.sin(args), torch.cos(args)], dim=-1)


class HashTextEmbedder(nn.Module):
    """Média dos embeddings dos ids de hash (padding = 0), seguida de LN + Linear."""

    def __init__(self, n_buckets, d):
        super().__init__()
        self.emb = nn.Embedding(n_buckets, d, padding_idx=0)
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, ids):
        mask = (ids > 0).unsqueeze(-1).float()
        return self.proj((self.emb(ids) * mask).sum(-2) / mask.sum(-2).clamp(min=1))


class Block(nn.Module):
    """Bloco pre-LN: x + Attn(LN(x)); x + MLP(LN(x)). Atenção bidirecional (sem máscara causal)."""

    def __init__(self, d, n_heads, mlp_ratio, dropout):
        super().__init__()
        assert d % n_heads == 0, "d_model precisa ser divisível por n_heads"
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class TabularLDM(nn.Module):
    def __init__(self, cfg: LDMConfig):
        super().__init__()
        self.cfg = cfg
        d, nf = cfg.d_model, 2 * cfg.n_fourier + 1

        self.text = HashTextEmbedder(cfg.n_buckets, d)
        self.kind_emb = nn.Embedding(2, d)
        self.num_option = nn.Sequential(nn.Linear(2 * nf, d), nn.GELU(), nn.Linear(d, d))
        self.null_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.z_proj = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.in_proj = nn.Linear(d, d)

        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.query = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.reg_head = nn.Linear(d, 1)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding) and m.padding_idx is not None:
                nn.init.zeros_(m.weight[m.padding_idx])

    def options(self, meta):
        """Embeddings das opções de cada coluna: (C, K+1, d), índice 0 = [NULL]; e a máscara de válidas."""
        nf = self.cfg.n_fourier
        cat = self.text(meta["cand_ids"])
        num = self.num_option(torch.cat([fourier(meta["opt_q"], nf), fourier(meta["opt_z"] / Z_CLIP, nf)], -1))
        opts = torch.where((meta["kind"] == CAT)[:, None, None], cat, num)
        C, K = meta["opt_q"].shape
        opts = torch.cat([self.null_emb.expand(C, 1, -1), opts], dim=1)
        valid = torch.arange(K + 1, device=opts.device)[None, :] <= meta["n_options"][:, None]
        return opts, valid

    def forward(self, local, z, masked, ratio, meta):
        """
        local  (N, C) índice local da célula (ignorado onde masked)
        z      (N, C) valor contínuo padronizado (só usado em numérica visível e não nula)
        masked (N, C) bool
        ratio  (N,)   fração de células mascaradas da linha
        meta   dict de tensores de TableEncoder.meta()
        Retorna {"logits": (N, C, K+1), "z_pred": (N, C)}.
        """
        N, C = local.shape
        opts, valid = self.options(meta)
        col = self.text(meta["name_ids"]) + self.kind_emb(meta["kind"])  # (C, d)

        value = self.in_proj(opts)[torch.arange(C, device=local.device)[None, :], local.clamp(min=0)]
        numeric_visible = (meta["kind"] == NUM)[None, :] & ~masked & (local != 0)
        value = value + numeric_visible.unsqueeze(-1) * self.z_proj(fourier(z / Z_CLIP, self.cfg.n_fourier))
        value = torch.where(masked.unsqueeze(-1), self.mask_emb, value)

        h = value + col[None] + self.time_mlp(timestep_embedding(ratio, self.cfg.d_model))[:, None, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        logits = torch.einsum("ncd,ckd->nck", self.query(h), self.out_proj(opts)) / math.sqrt(self.cfg.d_model)
        logits = logits.masked_fill(~valid[None], float("-inf"))
        return {"logits": logits, "z_pred": self.reg_head(h).squeeze(-1)}
