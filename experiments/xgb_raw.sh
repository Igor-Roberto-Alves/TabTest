#!/usr/bin/env bash
# XGBoost só com colunas cruas (sem engenharia de features): quanto as features à mão valem?
# Roda em CPU com nthread=8, em paralelo com a fila da GPU. A AmEx espera experiments/xgb_emb.sh
# terminar, para os dois XGBoost da AmEx não dividirem a RAM ao mesmo tempo.
set -u
cd "$(dirname "$0")/.."
for frac in 1 0.1 0.01; do
  python -m src.baselines.xgb home_credit --raw --label-frac "$frac" --set params.nthread=8
done
while pgrep -f "xgb_emb[.]sh" > /dev/null; do sleep 60; done
for frac in 1 0.1 0.01; do
  python -m src.baselines.xgb amex --raw --label-frac "$frac" --set params.nthread=8
done
