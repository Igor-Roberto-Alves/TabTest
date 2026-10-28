#!/usr/bin/env bash
# TabSTAR na tabela crua (as colunas do XGB --raw) e com o embedding congelado de um pré-treino, em
# Home Credit e AmEx. Cada argumento é "raw" ou a tag de um embedding:
#   experiments/tabstar.sh raw base_v2b     # PC da esquerda
#   experiments/tabstar.sh base_v1          # PC da direita
# N = linhas de treino (padrão 20000; 0 = todas, execuções ..._full). BASES = bases, na ordem (padrão "home_credit amex").
#   N=0 BASES=amex experiments/tabstar.sh raw
# Pré-requisito por tag: python -m src.pragma.export_embeddings runs/pragma/<run>/ckpt_best.pt --tag <tag> --bases home_credit amex
# GPU (fine-tuning LoRA). Execuções em runs/baselines/tabstar_<base>_raw[_emb_<tag>]_{n20k,full}.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
N=${N:-20000}
for what in "$@"; do
  for base in ${BASES:-home_credit amex}; do
    if [ "$what" = raw ]; then
      $PY -m src.baselines.tabstar "$base" --max-train "$N"
    else
      # na AmEx, cruas + embedding passam de 200 colunas e o lote de 64 estoura os 20 GB da GPU
      lb=64; [ "$base" = amex ] && lb=32
      $PY -m src.baselines.tabstar "$base" --max-train "$N" --emb "$what" --lora-batch $lb
    fi
  done
done
