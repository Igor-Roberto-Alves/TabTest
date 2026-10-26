# Modelo fundacional de eventos bancários (estilo PRAGMA): tasks possíveis

Esta branch (`pragma-banking`) é para testar um modelo fundacional parecido com o PRAGMA (Revolut, arXiv 2604.08649) com dados públicos. Este documento lista as tasks que dá para atacar com três bases:

- **MBD**: Multimodal Banking Dataset, da Sber AI Lab;
- **Home Credit**: Home Credit Default Risk, no Kaggle;
- **AmEx**: American Express Default Prediction, no Kaggle.

Ainda não há código nesta branch. Os números e esquemas abaixo vêm dos papers e das páginas das competições. Precisam ser conferidos quando os dados forem baixados.

## 1. O que queremos reproduzir do PRAGMA

O PRAGMA representa cada cliente como um **perfil estático** mais um **histórico de eventos**. Cada campo de um evento vira uma tripla (chave, valor, tempo). O modelo tem três encoders: perfil, evento e histórico, este último juntando os dois. O pré-treino é MLM com três tipos de máscara: tokens soltos, eventos inteiros e tipos semânticos. Depois disso, o modelo serve a várias tarefas, seja por embeddings com um modelo linear em cima, seja por fine-tuning LoRA.

Em que nossa proposta difere: o encoder de evento reaproveita a representação de célula do LDM (`src/ldm/`). Nela, as colunas são descritas por texto e quantis, e nenhum parâmetro depende do esquema. Com isso, **um único modelo pode ser pré-treinado nas três bases ao mesmo tempo**, apesar de terem esquemas diferentes. O PRAGMA não faz isso: ele trabalha com um único esquema.

## 2. As bases

| | **MBD** | **Home Credit** | **AmEx** |
|---|---|---|---|
| Entidade | Cliente corporativo | Solicitante de crédito | Titular de cartão |
| Tamanho | Mais de 1,5 milhão de clientes em 2 anos (o paper também cita mais de 2 milhões) | 307 mil solicitações no treino | 459 mil clientes, cerca de 5,5 milhões de extratos |
| Perfil estático | Pouco ou nenhum | `application_*`: 122 colunas, entre renda, bens, família e escores externos | Nenhum |
| Eventos | Transações de cartão, eventos de geolocalização, embeddings de diálogos com o suporte | Históricos com tempo relativo (`DAYS_*`, `MONTHS_BALANCE`): `bureau` + `bureau_balance`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`, `installments_payments` | Até 13 extratos mensais com cerca de 190 features anonimizadas (prefixos `D_` atraso, `S_` gasto, `P_` pagamento, `B_` saldo, `R_` risco) |
| Tempo | Timestamp por evento, irregular | Relativo à data da solicitação, sem data absoluta | Data do extrato (`S_2`), quase mensal |
| Rótulo original | Compra mensal de 4 produtos; casamento entre modalidades | `TARGET`: dificuldade de pagamento | `target`: não pagar em até 120 dias após o último extrato |
| Métrica oficial | ROC-AUC (propensão), métricas de ranking (casamento) | ROC-AUC | Média do Gini normalizado e da taxa de default capturada nos 4% de maior risco |
| Observações | Categóricas codificadas. O texto dos diálogos não vem, só os embeddings | Nomes de colunas legíveis, o que é bom para o encoder de texto | Nomes sem semântica (`D_39`, `B_30`, ...). Os negativos foram subamostrados a 5% |

Em resumo: o **MBD** é a base de escala e multimodalidade, o **Home Credit** tem a estrutura perfil + histórico, e a **AmEx** tem sequências regulares e curtas com features anonimizadas.

## 3. Tasks

Cada task traz o rótulo, a métrica e o baseline de comparação. A prioridade vai de **P1** (fazer primeiro) a **P3**.

### 3.1 Pré-treino auto-supervisionado

Estas tasks são o objetivo de pré-treino, não a avaliação final. Elas entram nas ablações de objetivo.

| Task | Definição | Bases |
|---|---|---|
| **Campo mascarado** | Mascarar células soltas de eventos e prever o valor: categoria, bin de quantil e valor contínuo. É o `mlm` e a difusão do LDM levados para o nível do evento | todas |
| **Evento mascarado** | Mascarar um evento inteiro e reconstruí-lo a partir do contexto | todas |
| **Tipo mascarado** | Mascarar uma coluna em todos os eventos do cliente, como todos os `AMT_PAYMENT` ou todos os `B_*`. Força o modelo a inferir uma variável a partir das outras | todas |
| **Próximo evento** | Prever o tempo até o próximo evento, o tipo e o valor. O alvo é causal, então dá para comparar com um objetivo autoregressivo | MBD, Home Credit |
| **Contraste entre janelas** | Duas subsequências do mesmo cliente devem ter embeddings próximos (CoLES) | todas |
| **Alinhamento multimodal** | Juntar transações, geolocalização e diálogo do mesmo cliente num espaço comum | MBD |

### 3.2 Risco de crédito (foco do estudo)

| Task | Rótulo | Métrica | Baseline | P |
|---|---|---|---|---|
| **Default Home Credit** | `TARGET` | ROC-AUC, KS, Brier | XGBoost/LightGBM com agregações feitas à mão dos históricos (o padrão das soluções do Kaggle) e logística | **P1** |
| **Default AmEx** | `target` | Métrica AmEx, ROC-AUC, KS | LightGBM com agregações por cliente (último, média, desvio, mínimo e máximo) | **P1** |
| **Ganho do histórico** (ablação) | `TARGET` do Home Credit com só `application`, com só os históricos e com os dois | ROC-AUC | Os mesmos modelos com as mesmas entradas | **P1** |
| **Predição antecipada** | Default AmEx usando só os primeiros *k* extratos (k = 1..13) | Métrica AmEx em função de *k* | LightGBM com as mesmas *k* linhas | P2 |
| **Atraso na próxima parcela** | Rótulo derivado do `installments_payments`: `DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT` ou `AMT_PAYMENT < AMT_INSTALMENT` na parcela seguinte, usando só o histórico anterior a ela | ROC-AUC | XGBoost em janelas | P2 |
| **Inadimplência futura no bureau** | Rótulo derivado do `bureau_balance`: o `STATUS` vai a atraso (≥ 1) nos próximos *m* meses | ROC-AUC | XGBoost em janelas | P3 |

Por que essas tasks importam: o Home Credit e a AmEx são duas das competições mais disputadas em crédito, com baselines de árvore muito fortes. A pergunta é se um modelo de sequência pré-treinado bate **agregações feitas à mão + GBDT**. O ganho que o PRAGMA reporta é sobre modelos internos da Revolut e não pode ser verificado. Aqui, a comparação é pública.

### 3.3 Propensão e comportamento

| Task | Rótulo | Métrica | Baseline | P |
|---|---|---|---|---|
| **Propensão de produto MBD** | Compra de cada um dos 4 produtos no mês seguinte (multirrótulo) | ROC-AUC médio | Baselines do paper do MBD (GBDT em agregados, CoLES) | **P1** |
| **Casamento de modalidades MBD** | Dada a sequência de transações de um cliente, achar a sequência de geolocalização ou de diálogo dele | Recall@k, MRR | Baselines do paper do MBD | P2 |
| **Gasto ou saldo futuro** | Regressão do valor das features `S_*` ou `B_*` do próximo extrato (AmEx), ou do volume transacionado no mês seguinte (MBD) | RMSE, MAE | GBDT e "repete o último valor" | P2 |
| **Atividade futura** | O cliente fica inativo, sem eventos, no próximo período? É uma proxy de churn | ROC-AUC | GBDT | P3 |

### 3.4 Qualidade do modelo fundacional

Estas tasks medem o quanto o pré-treino vale por si, e não uma tarefa específica.

| Task | Definição | P |
|---|---|---|
| **Linear probe** | Embedding congelado do cliente + logística, em todas as tasks P1. Compara com o fine-tuning (completo e LoRA) | **P1** |
| **Curva de rótulos** | Treinar as tasks P1 com 1%, 10% e 100% dos rótulos. É onde o pré-treino deve fazer mais diferença | **P1** |
| **Transferência entre bases** | Pré-treinar em MBD + AmEx e avaliar no Home Credit, e as outras combinações. Compara com pré-treinar na própria base e com treinar do zero. Testa o que o PRAGMA não testa: um modelo para vários esquemas | P2 |
| **Escala** | Tamanho do modelo e quantidade de dados de pré-treino contra o desempenho nas tasks P1 | P3 |
| **Ablação do perfil** | Remover o encoder de perfil (Home Credit). Repete a ablação do PRAGMA, em que o perfil importava para crédito e não para engajamento | P2 |

### 3.5 Imputação e geração

Estas usam o objetivo de difusão do LDM, que o PRAGMA não tem.

| Task | Definição | Métrica | P |
|---|---|---|---|
| **Imputação** | A AmEx e o Home Credit têm muitos faltantes. Mascarar valores observados e reconstruí-los | Acurácia e RMSE contra imputação por média e por GBDT | P3 |
| **Sequências sintéticas** | Gerar históricos de clientes | Train-on-synthetic-test-on-real em default; distância entre distribuições; teste de privacidade (*membership inference*) | P3 |

### 3.6 Interpretabilidade

| Task | Definição | P |
|---|---|---|
| **Atribuição por evento** | Quais eventos ou campos pesam na predição de default (atenção, gradiente × entrada, SHAP agrupado por evento). Compara com o SHAP do GBDT | P3 |
| **Calibração** | Curva de calibração e Brier nas tasks de crédito. Importa num contexto regulado | P2 |

## 4. Cuidados de avaliação

- **Vazamento temporal.**
  - No pré-treino, não usar eventos posteriores à data de corte do rótulo do cliente.
  - No MBD, o split deve ser temporal: pré-treino e treino nos meses iniciais, teste nos finais.
  - No Home Credit, os históricos já são anteriores à solicitação.
  - Na AmEx, o rótulo é posterior ao último extrato.
- **Clientes de teste.** Clientes da avaliação não entram no pré-treino, ou entram só com eventos anteriores ao corte. As duas versões devem ser reportadas.
- **Teste do Kaggle.** O Home Credit e a AmEx têm teste privado sem rótulo. Usar um hold-out estratificado do treino, com várias sementes, e submeter ao Kaggle (late submission) só no fim.
- **Subamostragem da AmEx.** A métrica oficial já compensa os negativos subamostrados (peso 20). As métricas próprias precisam fazer o mesmo.
- **Mesmo orçamento.** Os GBDTs entram com tuning e com as agregações conhecidas das soluções públicas. Do contrário, a comparação favorece o modelo fundacional.

## 5. Ordem sugerida

1. **Dados:** baixar as três bases e converter para um formato comum, com uma tabela de perfil e uma tabela longa de eventos `(cliente, tempo, tipo, campos...)`. O MBD é grande, então vale começar por uma amostra de clientes.
2. **Baselines P1:** GBDT + agregações no Home Credit, na AmEx e na propensão do MBD.
3. **Modelo:**
   - encoder de evento = célula do LDM;
   - mais codificação de tempo: log do intervalo e senoides de calendário;
   - mais encoder de histórico;
   - mais encoder de perfil.
4. **Pré-treino** com as máscaras de campo, evento e tipo, primeiro por base e depois nas três juntas.
5. **Tasks P1:** linear probe, fine-tuning e curva de rótulos. Depois as tasks P2.

## Referências

- PRAGMA: Revolut Foundation Model. arXiv 2604.08649.
- Mollaev et al. Multimodal Banking Dataset: Understanding Client Needs through Event Sequences. arXiv 2409.17587. Dados em `ai-lab/MBD` no Hugging Face.
- Home Credit Default Risk: kaggle.com/competitions/home-credit-default-risk.
- American Express Default Prediction: kaggle.com/competitions/amex-default-prediction.
- Babaev et al. CoLES: Contrastive Learning for Event Sequences with Self-Supervision. SIGMOD 2022.
- Padhi et al. Tabular Transformers for Modeling Multivariate Time Series (TabFormer). ICASSP 2021.
