"""Adaptação do LDM a uma tabela nova e predição de uma coluna alvo.

O codificador é ajustado no treino da tabela nova (só estatísticas, sem gradiente).
Com steps > 0, uma cópia do modelo recebe fine-tuning nessa tabela, alternando
lotes com a loss de difusão comum e lotes em que o alvo é sempre mascarado.
steps = 0 é zero-shot: só o pré-treino.
"""

import copy

import numpy as np
import torch

from .corpus import EncodedTable, to_torch
from .table import CAT, NUM


def adapt_and_predict(diffusion, train_df, test_df, target, kind, steps=300, batch_size=128, lr=1e-4,
                      target_frac=0.5, seed=0, device="cpu", encoder_kwargs=None, verbose=False):
    """kind: "regression" (alvo numérico) ou "binary"/"multiclass" (alvo categórico, sem cortar classes).

    Retorna (spec do alvo, probs (n_test, n_options) sem [NULL], y_reg (n_test,) na escala original).
    """
    torch.manual_seed(seed)
    kinds = {target: NUM if kind == "regression" else CAT}
    table = EncodedTable.from_frame("target", train_df, encoder_kwargs, kinds, exact=(target,))
    j = table.encoder.column_index(target)
    meta = to_torch(table.meta, device)
    diffusion = copy.deepcopy(diffusion).to(device)

    if steps:
        local_all = torch.as_tensor(table.local, device=device)
        z_all = torch.as_tensor(table.z, device=device)
        force = torch.zeros(local_all.shape[1], dtype=torch.bool, device=device)
        force[j] = True
        opt = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=0.01)
        diffusion.train()
        for step in range(1, steps + 1):
            rows = torch.randint(0, len(local_all), (batch_size,), device=device)
            local, z = local_all[rows], z_all[rows]
            if torch.rand(()) < target_frac:  # objetivo da tarefa: p(alvo | resto)
                stats = diffusion.loss(local, z, meta, force_mask=force.expand_as(local), weighting="uniform")
            else:  # objetivo de difusão: mantém o modelo da tabela inteira
                stats = diffusion.loss(local, z, meta)
            opt.zero_grad(set_to_none=True)
            stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            opt.step()
            if verbose and step % 100 == 0:
                print(f"    fine-tune {step}/{steps}  loss={stats['loss'].item():.4f}")

    test = to_torch(table.encoder.encode(test_df), device)
    probs, z_pred = [], []
    for b in range(0, len(test_df), 1024):
        p, zp = diffusion.predict(test["local"][b:b + 1024], test["z"][b:b + 1024], meta, [j])
        probs.append(p[:, 0, 1:])  # descarta a opção [NULL]
        z_pred.append(zp[:, 0])
    spec = table.encoder.columns[j]
    probs = torch.cat(probs).cpu().numpy()
    y_reg = torch.cat(z_pred).cpu().numpy() * spec.std + spec.mean if spec.kind == NUM else None
    return spec, probs / probs.sum(1, keepdims=True), y_reg


def expected_bin_value(spec, probs):
    """Valor esperado sob a distribuição de bins (alternativa à head de regressão)."""
    return probs @ np.asarray(spec.bin_values)
