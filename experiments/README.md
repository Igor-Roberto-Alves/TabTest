# Diário dos experimentos

Registro do que foi rodado e do que saiu. Cada execução tem o diretório `runs/<grupo>/<nome>/`, com a config, `status.json` (descrição e métricas), `metrics.jsonl`, os eventos do TensorBoard e o checkpoint ou `results.json`. Os grupos são `pragma` (pré-treino), `pragma_ft` (tarefas) e `baselines` (XGBoost). O painel ao vivo e o `tensorboard --logdir runs` leem esses arquivos.

Todos os números usam o mesmo split por cliente, gravado em `data/banking/prepared/<base>/split.npy`, e as mesmas métricas, de `src/pragma/evaluate.py`. A métrica principal é a AUC no Home Credit, a AUC média dos 4 rótulos no MBD e a métrica oficial na AmEx.

## Tarefas usadas na comparação

Uma tarefa supervisionada por base, definida em [src/pragma/prepare.py](../src/pragma/prepare.py) (campo `task` de cada store). O modelo fundacional e o XGBoost veem as mesmas tabelas brutas e os mesmos clientes em cada split. A diferença é que o XGBoost recebe agregações por cliente feitas à mão ([src/baselines/](../src/baselines/)), e o modelo recebe os eventos crus.

| Base | Tarefa | Tipo | Métrica principal | Linhas de rótulo (treino / val / teste) | Taxa de positivos |
|---|---|---|---|---|---|
| Home Credit | `default` (TARGET) | binária, 1 linha por cliente | AUC | 246.008 / 30.751 / 30.752 | 8,0% |
| AmEx | `default` (target) | binária, 1 linha por cliente | métrica AmEx | 367.130 / 45.891 / 45.892 | 25,9% |
| MBD-mini | `propensity` (target_1 a target_4) | 4 binárias, 1 linha por (cliente, mês) | AUC média dos 4 | 721.764 / 240.204 / 240.720 | 0,05% a 0,45% |

### Home Credit: default no empréstimo solicitado

**Definição.** Cada cliente fez um pedido de crédito, que é uma linha de `application_train.csv`. O rótulo oficial é `TARGET = 1` quando o cliente atrasou mais de X dias em pelo menos uma das Y primeiras parcelas desse empréstimo. Em todos os outros casos, `TARGET = 0`. A Home Credit não divulga X nem Y.

**O que entra.** O perfil é o próprio pedido: 120 colunas de renda, valores, idade, emprego e scores externos. O histórico vem de 6 fontes de eventos anteriores ao pedido:
- `bureau`: créditos em outras instituições;
- `bureau_balance`: o saldo mensal desses créditos;
- `previous_application`: pedidos anteriores na Home Credit;
- `pos_cash`, `credit_card` e `installments`: o histórico desses contratos.

O tempo de cada evento é medido em dias relativos à data do pedido. O split é aleatório por cliente, 80/10/10. Os 48.744 clientes do teste do Kaggle não têm rótulo e entram só no pré-treino.

Exemplos do perfil (`application_train.csv`, algumas das 120 colunas):

| SK_ID_CURR | TARGET | Contrato | Sexo | Renda | Crédito | Parcela | DAYS_BIRTH | DAYS_EMPLOYED | Ocupação | EXT_SOURCE_2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 100002 | **1** | Cash loans | M | 202.500 | 406.597 | 24.700 | −9.461 (25 anos) | −637 | Working | 0,263 |
| 100003 | 0 | Cash loans | F | 270.000 | 1.293.502 | 35.698 | −16.765 (45 anos) | −1.188 | State servant | 0,622 |
| 100004 | 0 | Revolving loans | M | 67.500 | 135.000 | 6.750 | −19.046 (52 anos) | −225 | Working | 0,556 |

Eventos do cliente 100002 em `bureau` (4 dos 8 créditos anteriores):

| CREDIT_ACTIVE | CREDIT_TYPE | DAYS_CREDIT | AMT_CREDIT_SUM | AMT_CREDIT_SUM_DEBT | CREDIT_DAY_OVERDUE |
|---|---|---|---|---|---|
| Closed | Credit card | −1.125 | 40.761 | — | 0 |
| Closed | Credit card | −476 | 0 | 0 | 0 |
| Closed | Consumer credit | −1.437 | 135.000 | 0 | 0 |
| Closed | Consumer credit | −1.121 | 19.071 | — | 0 |

Eventos do cliente 100003 em `installments` (parcelas de contratos anteriores; o pagamento foi feito alguns dias antes do vencimento):

| SK_ID_PREV | Parcela nº | DAYS_INSTALMENT | DAYS_ENTRY_PAYMENT | AMT_INSTALMENT | AMT_PAYMENT |
|---|---|---|---|---|---|
| 2396755 | 4 | −2.220 | −2.226 | 6.737 | 6.737 |
| 1810518 | 2 | −686 | −690 | 98.357 | 98.357 |
| 1810518 | 3 | −656 | −661 | 98.357 | 98.357 |

### AmEx: default nos 120 dias após o último extrato

**Definição.** Cada cliente tem até 13 extratos mensais de cartão, de março de 2017 a março de 2018. O rótulo é `target = 1` quando o cliente não pagou o valor devido em até 120 dias após a data do último extrato.

**O que entra.** Os 5,5 milhões de extratos, cada um com 188 colunas anonimizadas e normalizadas. O prefixo indica a família da coluna:
- `D_`: atraso (delinquency);
- `S_`: gasto;
- `P_`: pagamento;
- `B_`: saldo;
- `R_`: risco.

Não há perfil fixo. O tempo de cada extrato é medido em dias relativos ao último (≤ 0). No store, as colunas são renomeadas pela família, por exemplo D_39 → "delinquency 39".

**Métrica.** A métrica oficial da competição é a média entre o Gini normalizado e a fração dos defaults capturada nos 4% de maior score. Na origem, os negativos foram subamostrados a 5%, então a métrica dá peso 20 a cada negativo. Por isso a taxa de positivos de 25,9% na amostra não é a taxa real. O split é aleatório por cliente, 80/10/10.

Últimos 4 extratos de dois clientes (algumas das 188 colunas):

| Cliente | target | S_2 (data) | P_2 | D_39 | B_1 | B_2 | R_1 | S_3 | D_63 | B_30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0000099d6b… | 0 | 2017-12-04 | 0,869 | 0,001 | 0,002 | 1,008 | 0,002 | 0,100 | CR | 0 |
| 0000099d6b… | 0 | 2018-01-11 | 0,910 | 0,006 | 0,003 | 1,005 | 0,008 | 0,099 | CR | 0 |
| 0000099d6b… | 0 | 2018-02-21 | 0,937 | 0,002 | 0,020 | 1,008 | 0,001 | 0,103 | CR | 0 |
| 0000099d6b… | 0 | 2018-03-13 | 0,935 | 0,009 | 0,009 | 1,008 | 0,006 | 0,135 | CR | 0 |
| 0000f99513… | **1** | 2017-12-17 | 0,408 | 0,097 | 0,932 | 0,021 | 0,008 | 0,167 | CO | 1 |
| 0000f99513… | **1** | 2018-01-17 | 0,410 | 0,098 | 0,949 | 0,025 | 0,000 | 0,168 | CO | 1 |
| 0000f99513… | **1** | 2018-02-05 | 0,413 | 0,647 | 0,982 | 0,024 | 0,010 | 0,150 | CO | 1 |
| 0000f99513… | **1** | 2018-03-01 | 0,425 | 0,531 | 0,979 | 0,029 | 0,009 | 0,153 | CO | 1 |

No cliente que deu default, P_2 é baixo, B_1 é alto, B_2 é baixo, e o atraso (D_39) dispara nos últimos meses.

### MBD-mini: propensão a 4 produtos no mês

**Definição.** Cada linha de rótulo é um par (cliente, mês m), de fevereiro de 2022 a janeiro de 2023, com até 12 linhas por cliente. Os rótulos são 4 binários: `target_i = 1` quando o cliente contrata o produto i durante o mês m. O banco não diz quais são os produtos.

**O que entra.** Só os eventos anteriores ao dia 1 de m. Esse ponto de corte é o `cutoff` da linha de rótulo e evita vazamento. Há 3 fontes:
- `trx`: transações, com valor, tipo, subtipo, moeda e códigos de origem e destino;
- `geo`: geolocalização, com geohash em 3 resoluções;
- `dialog`: conversas com o banco, cada uma um embedding reduzido por PCA a 16 colunas.

Não há perfil fixo. O split segue os folds oficiais: folds 0 a 2 para treino, 3 para validação e 4 para teste. Os positivos são raríssimos, por isso a AUC por produto oscila bastante.

Exemplo: cliente 27344f76…, com 335 transações, 622 eventos de geolocalização e 1 diálogo. As linhas de rótulo dos 3 últimos meses:

| mon | target_1 | target_2 | target_3 | target_4 | Eventos visíveis (antes do dia 1) |
|---|---|---|---|---|---|
| 2022-11 | 0 | 0 | 0 | 0 | até 31/10/2022 |
| 2022-12 | 0 | 0 | 0 | 0 | até 30/11/2022 |
| 2023-01 | 0 | **1** | 0 | 0 | até 31/12/2022 |

Primeiras transações do mesmo cliente (`trx`, algumas colunas):

| event_time | amount | event_type | event_subtype | currency | dst_type11 | dst_type12 |
|---|---|---|---|---|---|---|
| 2021-01-03 09:04 | 8.535 | 11 | 32 | 11 | 1.569 | 7.793 |
| 2021-01-03 14:16 | 4.658.568 | 1 | 12 | 11 | 306 | 26.766 |
| 2021-01-04 13:21 | 40.100 | 1 | 12 | 11 | 306 | 26.766 |
| 2021-01-08 09:38 | 244.087 | 1 | 12 | 11 | 780 | 14.702 |

Geolocalização (`geo`):

| event_time | geohash_4 | geohash_5 | geohash_6 |
|---|---|---|---|
| 2021-12-21 11:57 | 48215 | 89615 | 2033453 |
| 2022-01-04 09:37 | 48215 | 89615 | 155870 |
| 2022-06-26 07:56 | 11777 | 44410 | 2380765 |

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

## XGBoost: colunas cruas e embedding do modelo fundacional (27/10/2026)

Dois experimentos extras, rodados em CPU em paralelo com a rodada 2:

- **Cru** ([experiments/xgb_raw.sh](xgb_raw.sh), `--raw`): só colunas brutas, sem agregação nem razões. No Home Credit, as 120 colunas do pedido, sem o histórico. Na AmEx, as 188 colunas do último extrato. O MBD não tem versão crua, porque sem agregar os eventos não sobra coluna.
- **Com embedding** ([experiments/xgb_emb.sh](xgb_emb.sh), `--emb base_v1`): o embedding [CLS] congelado do `base_v1` (256 colunas, de [src/pragma/export_embeddings.py](../src/pragma/export_embeddings.py)), junto com as features feitas à mão ou sozinho (`--emb-only`). O embedding vem do checkpoint de pré-treino, que não viu rótulos nem clientes de validação ou teste, então não vaza o alvo. No MBD, ele respeita o corte de cada linha.

Métrica principal no teste (AUC; métrica AmEx na AmEx):

| Base | Rótulos | Cru | Só embedding | **Feito à mão** (baseline) | Feito à mão + embedding | Fine-tuning completo (rodada 1) |
|---|---|---|---|---|---|---|
| Home Credit | 100% | 0,764 | 0,711 | **0,801** | 0,800 | 0,792 |
| Home Credit | 10% | 0,748 | — | **0,776** | 0,774 | rodada 2 |
| Home Credit | 1% | 0,724 | 0,638 | **0,741** | 0,740 | rodada 2 |
| AmEx | 100% | 0,790 | 0,712 | 0,792 | **0,794** | 0,7915 |
| AmEx | 10% | 0,774 | — | 0,780 | **0,781** | rodada 2 |
| AmEx | 1% | 0,747 | 0,674 | **0,751** | 0,750 | rodada 2 |
| MBD-mini | 100% | — | 0,752 | 0,802 | **0,816** | 0,815 |
| MBD-mini | 10% | — | — | 0,773 | **0,777** | rodada 2 |
| MBD-mini | 1% | — | 0,634 | **0,736** | 0,726 | rodada 2 |

Leitura:

- **O valor das features feitas à mão depende da base.**
  - No Home Credit, elas valem cerca de 0,04 de AUC sobre o pedido cru, porque o histórico (bureau, parcelas, cartão) só entra por meio delas.
  - Na AmEx, valem só 0,002. O último extrato sozinho já carrega quase tudo, já que suas colunas são agregados prontos feitos pela própria AmEx.
- **O embedding congelado ajuda o XGBoost no MBD e quase nada nas outras bases.**
  - No MBD com 100% dos rótulos, ele leva o XGBoost de 0,802 para 0,816, e a validação sobe junto (0,824 → 0,843). É o melhor resultado no MBD até agora, empatado com o fine-tuning.
  - O MBD é a base com menos engenharia (95 features) e com eventos crus mais ricos, então é lá que o resumo aprendido acrescenta algo.
  - No Home Credit e na AmEx, a diferença fica dentro de ±0,002.
- **Com poucos rótulos, o embedding não ajudou, ao contrário do esperado.** Com 1% dos rótulos, as 256 colunas a mais pioram um pouco (MBD: 0,736 → 0,726). Com poucas linhas, o XGBoost sobreajusta nas colunas extras.
- **O embedding sozinho vale o mesmo que o probe linear** (Home Credit 0,711 contra 0,714; AmEx 0,712 contra 0,726; MBD 0,752 contra 0,750). O congelado sabe menos do que as features feitas à mão, e o ganho do pré-treino aparece no fine-tuning.

## Rodada 1 do modelo fundacional (`experiments/round1.yaml`)

### Fase 1: sanidade, 5 mil passos por base

O probe é uma logística sobre o embedding congelado, avaliada na validação.

| Execução | val/loss final | Probe (validação) | XGBoost 100% (validação) |
|---|---|---|---|
| sanity_home_credit | 3,230 | AUC 0,650 | 0,788 |
| sanity_amex | 7,817 | AUC 0,930 · AmEx 0,670 | 0,792 |
| sanity_mbd_mini | 3,068 | AUC 0,704 | 0,824 |

Leitura: as perdas caem e o probe sai do acaso nas três bases, então o pré-treino aprende algo útil. Com só 5 mil passos e um probe linear, o modelo ainda fica bem abaixo do XGBoost. A comparação justa vem com o `base_v1` (100 mil passos) e o fine-tuning da fase 5.

### Fase 3: base_v1 (concluída em 27/10/2026, 00:01)

Pré-treino nas três bases juntas, 100 mil passos, config padrão (25M de parâmetros). Levou cerca de 3h55 na RTX A4500. O checkpoint usado nas tarefas é `runs/pragma/base_v1/ckpt_best.pt`, de menor `val/loss` (3,759).

| Probe (validação) | sanidade, só a base | base_v1, três bases | XGBoost 100% (validação) |
|---|---|---|---|
| Home Credit, AUC | 0,650 | **0,686** | 0,788 |
| AmEx, métrica AmEx (AUC) | 0,670 (0,930) | **0,701** (0,940) | 0,792 (0,961) |
| MBD, AUC média | 0,704 | 0,682 | 0,824 |

Leitura:

- O pré-treino conjunto e mais longo melhora o probe no Home Credit e na AmEx, mas não no MBD, que fica abaixo da sanidade feita só nele.
- O probe do MBD oscilou entre 0,61 e 0,70 ao longo do treino. Os positivos são raríssimos, então parte da oscilação é ruído.
- A perda caiu de forma contínua até o fim (val/loss 6,5 → 3,76), sem sinal de sobreajuste.
- Com embedding congelado e logística, o modelo segue bem abaixo do XGBoost. A fase 5 testa o fine-tuning.

### Fase 5: tarefas a partir do base_v1

Fine-tuning com 6 mil passos e lote 64, usando o checkpoint de menor `val/loss` na validação. O "do zero" usa a mesma arquitetura e o mesmo orçamento, sem pré-treino (lr 3e-4).

**Home Credit, default (AUC):**

| Modo | Val | **Teste** | KS teste | Brier teste |
|---|---|---|---|---|
| Probe (logística sobre o embedding) | 0,705 | 0,714 | 0,319 | 0,073 |
| LoRA r=8 (0,02M parâmetros treináveis) | 0,774 | 0,782 | 0,436 | 0,069 |
| Fine-tuning completo | 0,783 | **0,792** | 0,446 | 0,068 |
| Do zero, sem pré-treino | 0,674 | 0,670 | 0,250 | 0,074 |
| XGBoost 100% | 0,788 | **0,801** | 0,457 | 0,068 |

Leitura:

- **O pré-treino vale cerca de 0,12 de AUC.** Com o mesmo orçamento, o fine-tuning completo chega a 0,792 e o treino do zero, a 0,670. Na validação, a diferença já aparece no passo 1.000 (0,760 contra 0,594).
- **O fine-tuning completo fica a 0,009 do XGBoost**, que tem 727 features feitas à mão, depois de só 6 mil passos.
- O LoRA e o fine-tuning completo atingiram o melhor ponto no fim ou perto dele. Mais passos devem ajudar.

**AmEx, default (métrica AmEx):**

| Modo | Val | **Teste** | AUC teste | KS teste | Brier teste |
|---|---|---|---|---|---|
| Probe | 0,715 | 0,726 | 0,945 | 0,755 | 0,082 |
| LoRA r=8 | 0,777 | 0,784 | 0,959 | 0,790 | 0,070 |
| Fine-tuning completo | 0,784 | **0,7915** | 0,961 | 0,795 | 0,068 |
| Do zero, sem pré-treino | 0,770 | 0,774 | 0,957 | 0,783 | 0,072 |
| XGBoost 100% | 0,792 | **0,7923** | 0,962 | 0,797 | 0,068 |

Leitura:

- **O fine-tuning completo empata com o XGBoost** (0,7915 contra 0,7923), que usa 1096 features feitas à mão.
- **O ganho do pré-treino é menor que no Home Credit** (+0,017 contra o treino do zero). A AmEx é uma sequência curta e regular de extratos, com nomes de coluna sem semântica, e o treino do zero já aprende bem nesse formato.

**MBD-mini, propensão (AUC média dos 4 produtos):**

| Modo | Val | **Teste** | KS teste | Brier teste |
|---|---|---|---|---|
| Probe | 0,778 | 0,750 | 0,393 | 0,0028 |
| LoRA r=8 | 0,756 | 0,768 | 0,412 | 0,0028 |
| Fine-tuning completo | 0,813 | **0,815** | 0,494 | 0,0028 |
| Do zero, sem pré-treino | 0,716 | 0,748 | 0,387 | 0,0028 |
| XGBoost 100% | 0,824 | **0,802** | 0,481 | 0,0028 |

AUC de teste por produto, fine-tuning completo contra XGBoost: target_1 0,780 / 0,793; target_2 **0,828** / 0,799; target_3 0,816 / 0,825; target_4 **0,836** / 0,793.

Leitura:

- **O fine-tuning completo supera o XGBoost no teste** (0,815 contra 0,802), na média e em dois dos quatro produtos. Ele também é mais estável entre os folds: 0,813 na validação e 0,815 no teste, enquanto o XGBoost cai de 0,824 para 0,802.
- **Ressalva:** os positivos são raríssimos (0,05% a 0,45%) e só uma semente foi rodada. É preciso confirmar com mais sementes.
- **O pré-treino vale cerca de 0,07 de AUC** contra o treino do zero (0,815 contra 0,748).
- **O LoRA fica bem abaixo do fine-tuning completo aqui** (0,768). Com tão poucos positivos, 6 mil passos de adaptação de baixa ordem parecem não bastar.

## Resumo da rodada 1

Métrica principal no teste, fine-tuning completo a partir do `base_v1` contra o XGBoost com features feitas à mão:

| Base | Fine-tuning completo | Do zero | XGBoost | Diferença para o XGBoost |
|---|---|---|---|---|
| Home Credit (AUC) | 0,792 | 0,670 | 0,801 | −0,009 |
| AmEx (métrica AmEx) | 0,7915 | 0,774 | 0,7923 | −0,001 |
| MBD-mini (AUC média) | **0,815** | 0,748 | 0,802 | **+0,013** |

Conclusões:

1. **O pré-treino conjunto nas três bases ajuda em todas.** O ganho vai de +0,017 a +0,12 sobre a mesma arquitetura treinada do zero, com o mesmo orçamento.
2. **Um modelo único, sem features feitas à mão, empata ou chega perto do XGBoost** no Home Credit e na AmEx, e o supera no MBD.
3. **O probe linear não basta.** O embedding congelado fica de 0,05 a 0,09 abaixo do fine-tuning. O LoRA r=8 fica no meio do caminho.
4. **Em todas as tarefas, o melhor ponto foi o último ou o penúltimo.** As tarefas estão subtreinadas com 6 mil passos.

Próximos passos sugeridos:

- Fine-tuning mais longo (20 a 30 mil passos) nas três bases.
- Repetir o fine-tuning completo com 3 sementes, para ter intervalos de confiança, sobretudo no MBD.
- Curva de rótulos (1% e 10%) no fine-tuning, para comparar com a do XGBoost. É onde o pré-treino deve mostrar mais vantagem.
- Busca de hiperparâmetros do pré-treino (`configs/pragma/sweep_pretrain.yaml`) e as ablações da fase 4: pré-treino por base contra o conjunto, e máscaras.

## Rodada 2 (`experiments/round2.yaml`)

Começou em 27/10/2026 às 11h15 no PC da direita (c1-03-04-16116, RTX A4500). São 43 execuções em quatro blocos: A) fine-tuning longo com 3 sementes; B) curva de rótulos; C) ablações do pré-treino; D) busca de hiperparâmetros do pré-treino.

Em paralelo, no PC da esquerda (c1-02-03-16117, outra RTX A4500), roda o `base_v2b`: o `base_v1` refeito do zero com 300 mil passos, cosseno ajustado ao novo total e checkpoints fixos em 100 mil e 200 mil (`log.keep_every=100000`). Ele responde se o pré-treino de 100 mil passos estava subtreinado.

### Bloco A: fine-tuning com 20 mil passos e 3 sementes (concluído em 27/10/2026, 20h20)

Métrica principal no teste, com o passo do checkpoint escolhido na validação entre parênteses:

| Base | Semente 0 | Semente 1 | Semente 2 | Média ± dp | Rodada 1 (6 mil) | Do zero, 20 mil | Do zero, 6 mil | XGBoost feito à mão |
|---|---|---|---|---|---|---|---|---|
| Home Credit (AUC) | 0,789 (8k) | 0,788 (10k) | 0,791 (16k) | 0,789 ± 0,002 | 0,792 | 0,691 | 0,670 | **0,801** |
| AmEx (métrica AmEx) | 0,795 (20k) | 0,793 (14k) | 0,792 (20k) | **0,793 ± 0,001** | 0,791 | 0,787 | 0,774 | 0,792 |
| MBD-mini (AUC média) | 0,822 (20k) | 0,827 (16k) | 0,819 (12k) | **0,823 ± 0,004** | 0,815 | 0,772 | 0,748 | 0,802 |

Leitura:

1. **Mais passos ajudam no AmEx (+0,002) e no MBD (+0,008), não no Home Credit (−0,003).** No Home Credit, o melhor passo fica entre 8 mil e 16 mil, e passar disso é sobreajuste. Lá são cerca de 5 épocas em 20 mil passos.
2. **Com 3 sementes, o resultado contra o XGBoost fica claro:** o modelo fundacional perde no Home Credit (−0,012), empata no AmEx (+0,001, dentro de um desvio-padrão) e ganha no MBD (+0,021, cerca de cinco desvios-padrão).
3. **O ganho do pré-treino sobre o "do zero" continua grande com o mesmo orçamento de 20 mil passos:** +0,098 no Home Credit, +0,007 no AmEx e +0,051 no MBD. O "do zero" melhora com mais passos, mas não alcança o fine-tuning.
4. No MBD e no AmEx, o XGBoost com o embedding do `base_v1` como feature (0,816 e 0,794) fica entre o XGBoost feito à mão e o fine-tuning.

### Bloco B: curva de rótulos (concluído em 27/10/2026, 23h30)

Fine-tuning a partir do `base_v1` e do zero com 1% dos rótulos (4 mil passos, avaliação a cada 500) e com 10% (8 mil passos, avaliação a cada 1.000). Os 100% vêm do bloco A. O XGBoost usa as mesmas linhas sorteadas.

| Base | Rótulos | FT completo | Do zero | XGB feito à mão | XGB + embedding | Melhor passo do FT |
|---|---|---|---|---|---|---|
| Home Credit (AUC) | 1% | 0,659 | 0,572 | **0,741** | 0,740 | 500 de 4.000 |
| | 10% | 0,751 | 0,712 | **0,776** | 0,774 | 1.000 de 8.000 |
| | 100% | 0,789 | 0,691 | **0,801** | 0,800 | 8–16 mil |
| AmEx (métrica AmEx) | 1% | 0,691 | 0,695 | **0,751** | 0,750 | 3.000 de 4.000 |
| | 10% | 0,772 | 0,769 | 0,780 | **0,781** | 2.000 de 8.000 |
| | 100% | 0,793 | 0,787 | 0,792 | **0,794** | 14–20 mil |
| MBD-mini (AUC média) | 1% | 0,646 | 0,650 | **0,736** | 0,726 | 500 de 4.000 |
| | 10% | 0,772 | 0,746 | 0,773 | **0,777** | 4.000 de 8.000 |
| | 100% | **0,823** | 0,772 | 0,802 | 0,816 | 12–20 mil |

Leitura:

1. **A hipótese de que o pré-treino ganha mais com poucos rótulos não se confirmou com este protocolo de ajuste.** A diferença para o XGBoost piora quando os rótulos diminuem: −0,01 a +0,02 com 100%, −0,03 a 0,00 com 10% e −0,06 a −0,09 com 1%.
2. **O protocolo é o suspeito principal.** Com 1% e 10%, o melhor ponto na validação costuma ser a primeira ou a segunda avaliação, e depois a nota só cai. Com taxa de aprendizado de 1e-4 na rede inteira, o modelo decora as poucas linhas em centenas de passos (com 1%, são ~100 épocas em 4 mil passos) e apaga o que o pré-treino tinha aprendido. No AmEx e no MBD a 1%, o FT chega a empatar com a rede do zero.
3. **O pré-treino ainda ajuda onde há várias fontes:** no Home Credit, +0,09 com 1%, +0,04 com 10% e +0,10 com 100% sobre a rede do zero. No AmEx (uma fonte só), o ganho é pequeno em qualquer fração.
4. **Próximo passo, bloco B′:** refazer 1% e 10% com taxa de aprendizado menor (3e-5), avaliação a cada 50 a 100 passos e no máximo 1.500 passos a 1%, e incluir probe linear e LoRA, que treinam poucos parâmetros e sobreajustam menos. Só então a curva de rótulos mede o pré-treino, e não o sobreajuste.

### base_v2b: pré-treino com 300 mil passos (PC da esquerda, concluído em 28/10/2026, 04h41)

Mesma configuração do `base_v1`, treinada do zero com 300 mil passos e cosseno ajustado ao novo total. Levou cerca de 11h50 a ~7 passos/s, com pico de 4,7 GB de memória de GPU. Probes (logística sobre o embedding congelado, na validação) nos checkpoints fixos:

| Checkpoint | Perda de validação | Probe HC (AUC) | Probe AmEx (métrica AmEx) | Probe MBD (AUC) |
|---|---|---|---|---|
| `base_v1`, 100 mil passos (fim do cosseno) | 3,762 | 0,686 | 0,701 | 0,682 |
| `base_v2b`, passo 100 mil (taxa ainda alta) | 3,78 | 0,697 | 0,700 | 0,707 |
| `base_v2b`, passo 200 mil | 3,62 | 0,720 | 0,733 | 0,705 |
| `base_v2b`, passo 300 mil (final) | **3,544** | **0,722** | **0,734** | **0,714** |

Leitura: **o `base_v1` estava subtreinado.** Com 3× mais passos, a perda cai 0,22 e os três probes sobem de 0,03 a 0,04, um ganho maior que qualquer diferença entre ablações do bloco C. A maior parte do ganho vem até os 200 mil passos; de 200 para 300 mil, o ganho é pequeno (+0,002 a +0,009). Próximo passo: repetir o fine-tuning do bloco A (e o B′) a partir do `ckpt_step300k.pt` do `base_v2b` para ver quanto disso chega às tarefas.

### Bloco C: ablações do pré-treino, 20 mil passos cada (concluído em 28/10/2026, 04h55)

Cada ablação muda uma coisa em relação ao `abl_ref` (configuração do `base_v1` encurtada para 20 mil passos). Métrica: probe linear na validação ao fim do treino. Nas ablações com uma só base, só existe o probe dessa base.

| Execução | O que muda | Probe HC (AUC) | Probe AmEx (métrica AmEx) | Probe MBD (AUC) |
|---|---|---|---|---|
| `abl_ref` | referência: três bases, todas as máscaras, texto por hash | 0,672 | 0,659 | 0,670 |
| `abl_only_home_credit` | só Home Credit | 0,652 (−0,020) | — | — |
| `abl_only_amex` | só AmEx | — | **0,720 (+0,061)** | — |
| `abl_only_mbd_mini` | só MBD-mini | — | — | 0,627 (−0,043) |
| `abl_no_type` | sem máscara de tipo (`p_type=0`) | 0,607 (−0,065) | 0,656 (−0,003) | 0,679 (+0,009) |
| `abl_no_event` | sem máscara de evento (`p_event=0`) | 0,659 (−0,013) | 0,629 (−0,030) | 0,643 (−0,027) |
| `abl_minilm` | nomes por MiniLM congelado em vez de hash | 0,632 (−0,040) | 0,660 (+0,001) | 0,633 (−0,037) |

Leitura:

1. **O pré-treino conjunto ajuda as bases com várias fontes (Home Credit +0,02, MBD +0,04) e atrapalha o AmEx (−0,06).** O AmEx tem uma fonte só, com 188 colunas de extrato, diferente das outras. No conjunto, ele recebe cerca de um terço dos passos e disputa capacidade com tabelas que não o ajudam. Isso combina com o ganho pequeno do pré-treino no AmEx nos blocos A e B. Vale testar um peso maior para o AmEx na amostragem (`temperature`), ou um modelo maior.
2. **As duas máscaras estruturais importam, e de formas diferentes.** A de tipo é essencial para o Home Credit (−0,065 sem ela), cujas 6 fontes têm colunas relacionadas. A de evento ajuda nas três bases, mais no AmEx e no MBD, onde a sequência temporal pesa.
3. **O texto por hash é melhor que o MiniLM congelado** (−0,04 no HC e no MBD com o MiniLM). Os nomes de colunas são códigos e abreviações ("delinquency 39", "AMT_CREDIT_SUM"), em que um modelo de linguagem genérico ajuda pouco, e o hash treinável se adapta a eles.
4. Nenhuma ablação chega perto do ganho de treinar mais: o `base_v2b` sobe os três probes em ~0,03 sobre o `base_v1`.

### Bloco D: busca de hiperparâmetros do pré-treino, 12 sorteios de 20 mil passos (concluído em 28/10/2026, 13h10)

Sorteios de `configs/pragma/sweep_pretrain.yaml`; probes na validação no passo 20 mil. Referência: `abl_ref` (configuração do `base_v1`: lr 3e-4, wd 0,05, d=256, 6 camadas de histórico, p_cell 0,15, p_event 0,1, p_type 0,1), com probes 0,672 / 0,659 / 0,670 e média 0,667.

| Trial | lr | wd | d | camadas | p_cell | p_event | p_type | HC | AmEx | MBD | Média |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **03** | 5,4e-4 | 0,01 | 256 | 4 | 0,10 | 0,19 | 0,20 | 0,663 | **0,708** | 0,683 | **0,685** |
| 01 | 8,2e-4 | 0,1 | 256 | 6 | 0,25 | 0,19 | 0,10 | 0,656 | 0,703 | 0,674 | 0,678 |
| 09 | 2,1e-4 | 0,05 | 256 | 6 | 0,15 | 0,15 | 0,05 | 0,649 | 0,651 | **0,717** | 0,672 |
| 08 | 3,1e-4 | 0,1 | 256 | 4 | 0,25 | 0,14 | 0,10 | **0,672** | 0,695 | 0,633 | 0,667 |
| 04 | 2,0e-4 | 0,05 | 128 | 6 | 0,10 | 0,08 | 0 | 0,630 | 0,657 | 0,689 | 0,659 |
| 07 | 1,4e-4 | 0,05 | 256 | 8 | 0,25 | 0,13 | 0,10 | 0,627 | 0,683 | 0,666 | 0,659 |
| 02 | 8,6e-4 | 0,01 | 256 | 8 | 0,10 | 0,26 | 0,10 | 0,633 | 0,682 | 0,660 | 0,658 |
| 05 | 4,4e-4 | 0,1 | 128 | 6 | 0,25 | 0,30 | 0,05 | 0,618 | 0,645 | 0,689 | 0,651 |
| 06 | 9,6e-4 | 0,05 | 256 | 8 | 0,15 | 0,22 | 0,10 | 0,657 | 0,686 | 0,577 | 0,640 |
| 10 | 7,8e-4 | 0,01 | 128 | 8 | 0,15 | 0,07 | 0,05 | 0,629 | 0,643 | 0,631 | 0,634 |
| 11 | 6,1e-4 | 0,1 | 128 | 4 | 0,25 | 0,06 | 0,20 | 0,631 | 0,599 | 0,633 | 0,621 |
| 00 | 4,3e-4 | 0,05 | 128 | 4 | 0,10 | 0,05 | 0 | 0,633 | 0,640 | 0,577 | 0,617 |

Leitura:

1. **d_model=256 é melhor que 128:** os 5 sorteios com 128 estão entre os piores (média ≤ 0,659).
2. **Mais camadas de histórico não ajudam em 20 mil passos:** os 4 sorteios com 8 camadas ficam abaixo dos de 4 a 6 camadas com o mesmo d.
3. **Máscaras de evento e de tipo mais fortes ajudam**, confirmando o bloco C: os dois melhores (03 e 01) usam p_event ≈ 0,19, contra 0,10 no `base_v1`.
4. **O AmEx é o que mais varia com os hiperparâmetros** (0,60 a 0,71). O `base_v1` está perto do pior caso para ele (0,659), e os melhores sorteios recuperam o que o AmEx perdia no pré-treino conjunto.
5. **Nenhum sorteio ganha nas três bases ao mesmo tempo:** cada base prefere uma região diferente (09 é o melhor no MBD, 08 no HC, 03 no AmEx).
6. **Candidato para o próximo pré-treino longo:** o trial_03 (lr 5,4e-4, wd 0,01, 4 camadas, p_event 0,19, p_type 0,2), treinado com os 300 mil passos do `base_v2b`. Ganha +0,018 de média sobre a referência no mesmo orçamento.
