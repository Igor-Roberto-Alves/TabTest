"""Modelo fundacional de eventos: encoders de evento, perfil e histórico, como no PRAGMA.

    célula (fonte s, coluna c, valor)  -> como no LDM (src/ldm/model.py): opção descrita por
                                          texto/quantis, coluna pelo texto do nome, mais a fonte
    evento   = CellEncoder([EVT] + células)      -> vetor do evento (saída do [EVT])
    perfil   = CellEncoder([EVT] + colunas do perfil), com pesos próprios
    histórico = Transformer([CLS], perfil, evento_1 + tempo_1, ..., evento_L + tempo_L)
    cliente  = LN(saída do [CLS])

Tempo do evento: log(1 + idade em dias até a referência), log(1 + dias desde o evento
anterior) e senoides de hora, dia da semana e dia do mês (zeradas quando desconhecidas).
Sem embedding posicional: a ordem vem do tempo.

Decoder de células (pré-treino): para a célula c do evento e,
    q = LN(saída_da_célula + W_ctx · saída_do_histórico_em_e)
    logit_k = <W_q q, W_out o_ck> / sqrt(d)     (o mesmo autocomplete do LDM)
Assim uma célula mascarada é prevista com o resto do evento e com o histórico, e um evento
inteiro mascarado é previsto só com o histórico.

Nenhum parâmetro depende de uma base: o mesmo modelo é pré-treinado em várias.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ldm.model import Block, HashTextEmbedder, PretrainedTextEmbedder, fourier
from src.ldm.table import CAT, NUM, Z_CLIP
from src.ldm.text import N_BUCKETS, encode_texts


@dataclass
class PragmaConfig:
    d_model: int = 256
    n_heads: int = 8
    cell_layers: int = 2      # encoder de evento (atenção entre as células de um evento)
    profile_layers: int = 2   # encoder do perfil
    history_layers: int = 6   # encoder do histórico (atenção entre eventos)
    mlp_ratio: int = 4
    dropout: float = 0.1
    n_fourier: int = 8
    n_buckets: int = N_BUCKETS
    text_encoder: str = "hash"  # hash | frozen | finetune
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"


def meta_to_torch(meta: dict, device) -> dict:
    """TableEncoder.meta() -> tensores (os textos crus ficam como arrays numpy)."""
    return {k: v if getattr(v, "dtype", None) == object else torch.as_tensor(v, device=device)
            for k, v in meta.items()}


class Vocabulary(nn.Module):
    """Embeddings compartilhados por todas as bases: textos, opções de valor e colunas."""

    def __init__(self, cfg: PragmaConfig):
        super().__init__()
        d, nf = cfg.d_model, 2 * cfg.n_fourier + 1
        self.cfg = cfg
        if cfg.text_encoder == "hash":
            self.text = HashTextEmbedder(cfg.n_buckets, d)
        else:
            self.text = PretrainedTextEmbedder(cfg.text_model, d, frozen=cfg.text_encoder == "frozen")
        self.kind_emb = nn.Embedding(2, d)
        self.num_option = nn.Sequential(nn.Linear(2 * nf, d), nn.GELU(), nn.Linear(d, d))
        self.null_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.z_proj = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        self.in_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def embed_texts(self, texts):
        """Lista de str -> (n, d)."""
        if self.cfg.text_encoder == "hash":
            ids = torch.as_tensor(encode_texts(texts), device=self.null_emb.device)
            return self.text(ids)
        return self.text(np.asarray(texts, dtype=object))

    def embed_meta_text(self, meta, key):
        if self.cfg.text_encoder == "hash":
            return self.text(meta[f"{key}_ids"])
        return self.text(meta[f"{key}_str"])

    def options(self, meta):
        """(C, K+1, d) embeddings das opções (0 = [NULL]) e máscara das válidas, como no LDM."""
        nf = self.cfg.n_fourier
        cat = self.embed_meta_text(meta, "cand")
        num = self.num_option(torch.cat([fourier(meta["opt_q"], nf), fourier(meta["opt_z"] / Z_CLIP, nf)], -1))
        opts = torch.where((meta["kind"] == CAT)[:, None, None], cat, num)
        C, K = meta["opt_q"].shape
        opts = torch.cat([self.null_emb.expand(C, 1, -1), opts], dim=1)
        valid = torch.arange(K + 1, device=opts.device)[None, :] <= meta["n_options"][:, None]
        return opts, valid

    def columns(self, meta):
        return self.embed_meta_text(meta, "name") + self.kind_emb(meta["kind"])


class CellEncoder(nn.Module):
    """[EVT] + células de um evento -> (vetor do evento, saídas das células)."""

    def __init__(self, cfg: PragmaConfig, n_layers):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.evt = nn.Parameter(torch.randn(d) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d)

    def forward(self, vocab: Vocabulary, meta, local, z, masked, opts, context):
        """local, z, masked (n, C); opts (C, K+1, d); context (d,) = fonte + coluna somados abaixo."""
        n, C = local.shape
        value = vocab.in_proj(opts)[torch.arange(C, device=local.device)[None, :], local.clamp(min=0)]
        numeric_visible = (meta["kind"] == NUM)[None, :] & ~masked & (local != 0)
        value = value + numeric_visible.unsqueeze(-1) * vocab.z_proj(fourier(z / Z_CLIP, self.cfg.n_fourier))
        value = torch.where(masked.unsqueeze(-1), self.mask_emb.to(value.dtype), value)
        h = value + vocab.columns(meta)[None] + context
        h = torch.cat([(self.evt + context).expand(n, 1, -1), h], dim=1)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        return h[:, 0], h[:, 1:]


class TimeEncoder(nn.Module):
    def __init__(self, cfg: PragmaConfig):
        super().__init__()
        nf = 2 * cfg.n_fourier + 1
        self.n_fourier = cfg.n_fourier
        self.mlp = nn.Sequential(nn.Linear(2 * nf + 9, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))

    def forward(self, age, dt, cal):
        """age, dt (B, L) em dias; cal (B, L, 3) hora/dia da semana/dia do mês (-1 = desconhecido)."""
        feats = [fourier(torch.log1p(age) / 8, self.n_fourier), fourier(torch.log1p(dt) / 8, self.n_fourier)]
        for j, period in enumerate((24, 7, 31)):
            known = (cal[..., j] >= 0).float()
            ang = 2 * math.pi * cal[..., j].clamp(min=0).float() / period
            feats += [(torch.sin(ang) * known).unsqueeze(-1), (torch.cos(ang) * known).unsqueeze(-1), known.unsqueeze(-1)]
        return self.mlp(torch.cat(feats, -1))


class PragmaModel(nn.Module):
    def __init__(self, cfg: PragmaConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.vocab = Vocabulary(cfg)
        self.event_encoder = CellEncoder(cfg, cfg.cell_layers)
        self.profile_encoder = CellEncoder(cfg, cfg.profile_layers)
        self.time = TimeEncoder(cfg)
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.profile_type = nn.Parameter(torch.randn(d) * 0.02)
        self.event_mask = nn.Parameter(torch.randn(d) * 0.02)
        self.history = nn.ModuleList(Block(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
                                     for _ in range(cfg.history_layers))
        self.ln_h = nn.LayerNorm(d)
        self.ctx_proj = nn.Linear(d, d)
        self.dec_ln = nn.LayerNorm(d)
        self.query = nn.Linear(d, d)
        self.reg_head = nn.Linear(d, 1)

        pretrained = getattr(self.vocab.text, "encoder", None)
        saved = {k: v.clone() for k, v in pretrained.state_dict().items()} if pretrained is not None else None
        self.apply(self._init)
        if saved is not None:
            pretrained.load_state_dict(saved)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding) and m.padding_idx is not None:
                nn.init.zeros_(m.weight[m.padding_idx])

    def forward(self, batch, metas, masks=None, decode=False):
        """metas: {"sources": {fonte: meta}, "profile": meta ou None} (tensores, ver meta_to_torch).
        masks: saída de objective.make_masks (None = nada mascarado).
        Retorna {"client": (B, d)} e, com decode=True, {"cells": {fonte: (logits, z_pred)}, "profile": ...}."""
        B, L = batch["seq_valid"].shape
        d = self.cfg.d_model
        dev = batch["seq_valid"].device
        names = list(batch["src"])
        src_vecs = self.vocab.embed_texts([n.replace("_", " ") for n in names])  # (S, d)

        seq = torch.zeros(B, L, d, device=dev)
        cell_out, opts_cache = {}, {}
        for k, name in enumerate(names):
            sb = batch["src"][name]
            if len(sb["b"]) == 0:
                continue
            meta = metas["sources"][name]
            opts_cache[name] = self.vocab.options(meta)
            masked = masks["cells"][name] if masks else torch.zeros_like(sb["local"], dtype=torch.bool)
            evt, cells = self.event_encoder(self.vocab, meta, sb["local"], sb["z"], masked, opts_cache[name][0],
                                            src_vecs[k])
            if masks:
                evt = torch.where(masks["events"][name][:, None], self.event_mask.to(evt.dtype), evt)
            seq = seq.index_put((sb["b"], sb["pos"]), (evt + src_vecs[k]).to(seq.dtype))
            cell_out[name] = cells
        seq = seq + self.time(batch["seq_age"], batch["seq_dt"], batch["seq_cal"])

        tokens = [self.cls.expand(B, 1, -1)]
        valid = [torch.ones(B, 1, dtype=torch.bool, device=dev)]
        has_profile = "profile" in batch and metas.get("profile") is not None
        if has_profile:
            pmeta = metas["profile"]
            popts = self.vocab.options(pmeta)
            pmask = masks["profile"] if masks else torch.zeros_like(batch["profile"]["local"], dtype=torch.bool)
            pvec, pcells = self.profile_encoder(self.vocab, pmeta, batch["profile"]["local"], batch["profile"]["z"],
                                                pmask, popts[0], self.profile_type)
            tokens.append((pvec + self.profile_type).unsqueeze(1))
            valid.append(torch.ones(B, 1, dtype=torch.bool, device=dev))
        offset = len(tokens)
        h = torch.cat(tokens + [seq.to(tokens[0].dtype)], dim=1)
        key_valid = torch.cat(valid + [batch["seq_valid"]], dim=1)
        attn = key_valid[:, None, None, :]  # (B, 1, 1, T): todos atendem só a posições válidas
        for blk in self.history:
            h = blk(h, attn)
        h = self.ln_h(h)
        out = {"client": h[:, 0]}
        if not decode:
            return out

        out["cells"] = {}
        for name, cells in cell_out.items():
            sb = batch["src"][name]
            ctx = h[sb["b"], offset + sb["pos"]]
            out["cells"][name] = self._decode(cells, ctx, *opts_cache[name])
        if has_profile:
            out["profile"] = self._decode(pcells, h[:, 1], *popts)
        return out

    def _decode(self, cells, ctx, opts, valid):
        q = self.dec_ln(cells + self.ctx_proj(ctx)[:, None])
        logits = torch.einsum("ncd,ckd->nck", self.query(q), self.vocab.out_proj(opts)) / math.sqrt(self.cfg.d_model)
        logits = logits.float().masked_fill(~valid[None], float("-inf"))
        return logits, self.reg_head(q).squeeze(-1).float()


class ClassifierHead(nn.Module):
    """Embedding do cliente -> logits (k rótulos). hidden = 0 dá uma camada linear."""

    def __init__(self, d, n_labels, hidden=0, dropout=0.1):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, n_labels))
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_labels))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) · B A x, com W congelada e B iniciada em zero."""

    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale


def apply_lora(model: nn.Module, r=8, alpha=16, dropout=0.0, targets=("qkv", "proj")):
    """Congela o modelo e põe adaptadores LoRA nas projeções de atenção de todos os blocos."""
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, Block):
            for name in targets:
                setattr(module, name, LoRALinear(getattr(module, name), r, alpha, dropout))
    return model
