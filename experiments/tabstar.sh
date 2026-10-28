#!/usr/bin/env bash
# TabSTAR na tabela crua (as colunas do XGB --raw) e com o embedding congelado de um pré-treino, em
# Home Credit e AmEx, com 20 mil linhas de treino. Cada argumento é "raw" ou a tag de um embedding:
#   experiments/tabstar.sh raw base_v2b     # PC da esquerda
#   experiments/tabstar.sh base_v1          # PC da direita
# Pré-requisito por tag: python -m src.pragma.export_embeddings runs/pragma/<run>/ckpt_best.pt --tag <tag> --bases home_credit amex
# GPU (fine-tuning LoRA). Execuções em runs/baselines/tabstar_<base>_raw[_emb_<tag>]_n20k.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
for what in "$@"; do
  for base in home_credit amex; do
    if [ "$what" = raw ]; then
      $PY -m src.baselines.tabstar "$base" --max-train 20000
    else
      $PY -m src.baselines.tabstar "$base" --max-train 20000 --emb "$what"
    fi
  done
done
