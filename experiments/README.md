# Diário dos experimentos

Registro do que foi rodado e do que saiu. Cada execução tem o diretório `runs/<grupo>/<nome>/`, com a config, `status.json` (descrição e métricas), `metrics.jsonl`, os eventos do TensorBoard e o checkpoint ou `results.json`. Os grupos são `pragma` (pré-treino), `pragma_ft` (tarefas) e `baselines` (XGBoost). O painel ao vivo e o `tensorboard --logdir runs` leem esses arquivos.

Todos os números usam o mesmo split por cliente, gravado em `data/banking/prepared/<base>/split.npy`, e as mesmas métricas, de `src/pragma/evaluate.py`. A métrica principal é a AUC no Home Credit, a AUC média dos 4 rótulos no MBD e a métrica oficial na AmEx.

## Baselines XGBoost (26/10/2026)

Código em [src/baselines/](../src/baselines/) e config em [configs/baselines/xgb.yaml](../configs/baselines/xgb.yaml).

- Comando: `python -m src.baselines.xgb <base> [--label-frac f]`.
- Features: agregações por cliente feitas à mão. São 727 no Home Credit, 1096 na AmEx e 95 no MBD.
- Treino: grade pequena escolhida na validação, com early stopping na AUC de validação. Só CPU.

| Base | Fração dos rótulos | Principal val | **Principal teste** | KS teste | Brier teste |
|---|---|---|---|---|---|
| home_credit | 100% | 0,788 | **0,801** | 0,457 | 0,068 |
| home_credit | 10% | 0,771 | 0,776 | 0,415 | 0,069 |
| home_credit | 1% | 0,737 | 0,741 | 0,364 | 0,072 |
| amex | 100% | 0,792 | **0,792** (AUC 0,962) | 0,797 | 0,068 |
| amex | 10% | 0,775 | 0,780 | 0,786 | 0,071 |
| amex | 1% | 0,746 | 0,751 | 0,772 | 0,077 |
| mbd_mini | 100% | 0,824 | **0,802** | 0,481 | 0,0028 |
| mbd_mini | 10% | 0,793 | 0,773 | 0,433 | 0,0028 |
| mbd_mini | 1% | 0,742 | 0,736 | 0,368 | 0,0028 |

Ressalvas:

- **MBD, validação contra teste:** o teste fica cerca de 0,02 abaixo da validação porque são folds oficiais diferentes, com positivos raríssimos (0,05% a 0,45%).
- **MBD, vazamento:** as features de cada linha (cliente, mês m) usam só eventos antes do dia 1 de m. Isso foi conferido em 300 linhas contra o store.
- **AmEx 100%, memória:** o pico de RAM foi de 15,4 GB. Depois disso, a leitura do cache foi trocada por pyarrow direto. A execução não foi refeita, mas as features e o treino são os mesmos.

## Rodada 1 do modelo fundacional (`experiments/round1.yaml`)

### Fase 1: sanidade, 5 mil passos por base

O probe é uma logística sobre o embedding congelado, avaliada na validação.

| Execução | val/loss final | Probe (validação) | XGBoost 100% (validação) |
|---|---|---|---|
| sanity_home_credit | 3,230 | AUC 0,650 | 0,788 |
| sanity_amex | 7,817 | AUC 0,930 · AmEx 0,670 | 0,792 |
| sanity_mbd_mini | 3,068 | AUC 0,704 | 0,824 |

Leitura: as perdas caem e o probe sai do acaso nas três bases, então o pré-treino aprende algo útil. Com só 5 mil passos e um probe linear, o modelo ainda fica bem abaixo do XGBoost. A comparação justa vem com o `base_v1` (100 mil passos) e o fine-tuning da fase 5.

### Fase 3: base_v1

Em andamento.
