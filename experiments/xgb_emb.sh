#!/usr/bin/env bash
# XGBoost com o embedding congelado do base_v1 como features (experimento extra da rodada 2).
# Pré-requisito: python -m src.pragma.export_embeddings runs/pragma/base_v1/ckpt_best.pt --tag base_v1
# Roda em CPU (nthread=12, para sobrar núcleos aos loaders da fila da GPU). Execuções em runs/baselines/.
set -u
cd "$(dirname "$0")/.."
for base in home_credit amex mbd_mini; do
  for frac in 1 0.1 0.01; do
    python -m src.baselines.xgb "$base" --emb base_v1 --label-frac "$frac" --set params.nthread=12
  done
  for frac in 1 0.01; do
    python -m src.baselines.xgb "$base" --emb base_v1 --emb-only --label-frac "$frac" --set params.nthread=12
  done
done
