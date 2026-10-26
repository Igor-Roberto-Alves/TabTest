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

Com in_context=True (objetivo "icl"), cada camada alterna atenção entre as colunas de uma
linha e atenção entre as linhas de uma coluna, como no TabPFN. As primeiras n_ctx linhas
do lote são o contexto (rotuladas, todas as células visíveis); as demais são consultas, que
leem o contexto mas não umas às outras. Assim o modelo aprende a relação alvo-resto da
tabela nova a partir dos exemplos do contexto, sem fine-tuning.
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
    text_encoder: str = "hash"  # hash | frozen | finetune
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    in_context: bool = False  # atenção também entre linhas (objetivo "icl")


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


class PretrainedTextEmbedder(nn.Module):
    """Mean pooling de um sentence-transformer sobre o texto cru, seguido de LN + Linear.

    frozen:   o encoder não recebe gradiente (só a projeção treina); cada texto é
              codificado uma única vez e fica em cache.
    finetune: o gradiente flui pelo encoder; os textos únicos do lote são
              recodificados a cada passo.
    Texto "" é padding e vira o vetor nulo antes da projeção, como no HashTextEmbedder.
    """

    def __init__(self, model_name, d, frozen, max_length=32):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.frozen = frozen
        self.max_length = max_length
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d))
        if frozen:
            self.encoder.requires_grad_(False)
        self._cache = {}

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()  # sem dropout: o cache precisa ser determinístico
        return self

    def _encode(self, texts):
        dev = self.proj[1].weight.device
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(dev)
        out = self.encoder(**tok).last_hidden_state
        m = tok["attention_mask"].unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, texts):
        """texts: array de str (qualquer shape) -> (*shape, d)."""
        flat = texts.reshape(-1)
        dev = self.proj[1].weight.device
        uniq = sorted({t for t in flat if t})
        hidden = self.encoder.config.hidden_size
        if not uniq:  # ex.: cand_str de um lote só com colunas numéricas
            vecs = torch.zeros(0, hidden, device=dev)
        elif self.frozen:
            missing = [t for t in uniq if t not in self._cache]
            if missing:
                with torch.no_grad():
                    for i in range(0, len(missing), 512):
                        chunk = missing[i:i + 512]
                        self._cache.update(zip(chunk, self._encode(chunk)))
            vecs = torch.stack([self._cache[t] for t in uniq]).to(dev)
        else:
            vecs = self._encode(uniq)
        vecs = torch.cat([torch.zeros(1, hidden, device=dev, dtype=vecs.dtype), vecs])
        index = {t: i + 1 for i, t in enumerate(uniq)}
        idx = torch.tensor([index.get(t, 0) for t in flat], device=dev).view(texts.shape)
        return self.proj(vecs[idx])


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

    def forward(self, x, attn_mask=None):
        """attn_mask (L, L) bool: True onde a posição da linha pode atender a da coluna."""
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class TabularLDM(nn.Module):
    def __init__(self, cfg: LDMConfig):
        super().__init__()
        self.cfg = cfg
        d, nf = cfg.d_model, 2 * cfg.n_fourier + 1

        if cfg.text_encoder == "hash":
            self.text = HashTextEmbedder(cfg.n_buckets, d)
        elif cfg.text_encoder in ("frozen", "finetune"):
            self.text = PretrainedTextEmbedder(cfg.text_model, d, frozen=cfg.text_encoder == "frozen")
        else:
            raise ValueError(f"text_encoder desconhecido: {cfg.text_encoder}")
        self.kind_emb = nn.Embedding(2, d)
        self.num_option = nn.Sequential(nn.Linear(2 * nf, d), nn.GELU(), nn.Linear(d, d))
        self.null_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.z_proj = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.in_proj = nn.Linear(d, d)

        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.n_layers))
        if cfg.in_context:
            self.row_blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
                                            for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.query = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.reg_head = nn.Linear(d, 1)

        pretrained = getattr(self.text, "encoder", None)
        saved = {k: v.clone() for k, v in pretrained.state_dict().items()} if pretrained is not None else None
        self.apply(self._init)
        if saved is not None:
            pretrained.load_state_dict(saved)  # o _init não pode sobrescrever o encoder pré-treinado

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding) and m.padding_idx is not None:
                nn.init.zeros_(m.weight[m.padding_idx])

    def embed_text(self, meta, key):
        """key = "name" | "cand": ids de hash ou texto cru, conforme o encoder."""
        if self.cfg.text_encoder == "hash":
            return self.text(meta[f"{key}_ids"])
        return self.text(meta[f"{key}_str"])

    def options(self, meta):
        """Embeddings das opções de cada coluna: (C, K+1, d), índice 0 = [NULL]; e a máscara de válidas."""
        nf = self.cfg.n_fourier
        cat = self.embed_text(meta, "cand")
        num = self.num_option(torch.cat([fourier(meta["opt_q"], nf), fourier(meta["opt_z"] / Z_CLIP, nf)], -1))
        opts = torch.where((meta["kind"] == CAT)[:, None, None], cat, num)
        C, K = meta["opt_q"].shape
        opts = torch.cat([self.null_emb.expand(C, 1, -1), opts], dim=1)
        valid = torch.arange(K + 1, device=opts.device)[None, :] <= meta["n_options"][:, None]
        return opts, valid

    def forward(self, local, z, masked, ratio, meta, n_ctx=None):
        """
        local  (N, C) índice local da célula (ignorado onde masked)
        z      (N, C) valor contínuo padronizado (só usado em numérica visível e não nula)
        masked (N, C) bool
        ratio  (N,)   fração de células mascaradas da linha
        meta   dict de tensores de TableEncoder.meta()
        n_ctx  só com in_context: as primeiras n_ctx linhas são o contexto
        Retorna {"logits": (N, C, K+1), "z_pred": (N, C)}.
        """
        N, C = local.shape
        opts, valid = self.options(meta)
        col = self.embed_text(meta, "name") + self.kind_emb(meta["kind"])  # (C, d)

        value = self.in_proj(opts)[torch.arange(C, device=local.device)[None, :], local.clamp(min=0)]
        numeric_visible = (meta["kind"] == NUM)[None, :] & ~masked & (local != 0)
        value = value + numeric_visible.unsqueeze(-1) * self.z_proj(fourier(z / Z_CLIP, self.cfg.n_fourier))
        value = torch.where(masked.unsqueeze(-1), self.mask_emb, value)

        h = value + col[None] + self.time_mlp(timestep_embedding(ratio, self.cfg.d_model))[:, None, :]
        if not self.cfg.in_context:
            for blk in self.blocks:
                h = blk(h)
        else:
            # consultas e contexto atendem só ao contexto: uma consulta não vê as outras
            row_mask = torch.zeros(N, N, dtype=torch.bool, device=h.device)
            row_mask[:, :n_ctx] = True
            for col_blk, row_blk in zip(self.blocks, self.row_blocks):
                h = col_blk(h)                                              # (N, C, d): entre colunas
                h = row_blk(h.transpose(0, 1), row_mask).transpose(0, 1)    # (C, N, d): entre linhas
        h = self.ln_f(h)

        logits = torch.einsum("ncd,ckd->nck", self.query(h), self.out_proj(opts)) / math.sqrt(self.cfg.d_model)
        logits = logits.masked_fill(~valid[None], float("-inf"))
        return {"logits": logits, "z_pred": self.reg_head(h).squeeze(-1)}
