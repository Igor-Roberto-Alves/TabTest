# TabTest

Repositório para testar **modelos fundacionais para dados tabulares** e compará-los com baselines clássicos. É o lado experimental do estudo em `../LDM_study`, onde comparei os papers TabPFN, TabSTAR, nuFormer (Nubank) e PRAGMA (Revolut).

## Objetivo

Descobrir, em datasets reais, quando um modelo fundacional tabular ganha de baselines bem ajustados e quando não ganha. A comparação considera:

- poder preditivo (AUC, log-loss, calibração; RMSE/MAE em regressão);
- custo de treino e de inferência;
- sensibilidade ao tamanho do dataset, a categóricas, a valores faltantes e a campos de texto;
- interpretabilidade, o que importa em crédito e em contextos regulados.

## Modelos fundacionais

### Principais

| Modelo | Ideia | Por que testar |
|---|---|---|
| **TabPFN** (v1 / v2) | Transformer pré-treinado em dados sintéticos (prior SCM/BNN). Faz *in-context learning*: o treino entra no contexto e a predição sai em um forward pass, sem ajuste de hiperparâmetros. | Referência de modelo fundacional tabular. Forte em datasets pequenos. Útil para medir onde a escala limitada começa a pesar. |
| **TabSTAR** | Encoder de texto descongelado, *target-aware tokens* e pré-treino multitarefa em 350 datasets. Fine-tuning com LoRA. | Feito para tabelas com **texto** e categóricas de alta cardinalidade. Aceita classificação e regressão. |

### Candidatos a incluir

Lista inicial, a revisar depois de ler os papers e os repositórios de cada um.

- **TabICL**: ICL tabular desenhado para mais linhas do que o TabPFN original.
- **TabDPT**: pré-treino em dados reais com ICL.
- **Mitra** (AutoGluon): fundacional pré-treinado com priors sintéticos mistos.
- **ConTextTab**: ICL com embeddings semânticos para texto e nomes de colunas.
- **CARTE**: representação baseada em grafo, com atenção a nomes de colunas e valores.
- **TabuLa-8B**: LLM ajustado para tabelas, para ver o custo-benefício de um modelo grande.
- **LLM zero/few-shot com serialização em texto**: ponto de referência para a abordagem "LLM genérico".

### Fora do escopo

**nuFormer** (Nubank) e **PRAGMA** (Revolut) usam dados e pesos proprietários e trabalham com sequências de eventos por cliente. Só entram como inspiração para a tokenização. Se o estudo passar a usar séries temporais, vale revisitar.

## Baselines

| Baseline | Papel |
|---|---|
| **Regressão logística** | Baseline linear padrão e interpretável. Usada com regularização (L1/L2) e com variantes de pré-processamento. |
| **Regressão Probit** | Mesmo papel da logística, com link normal. Comum em econometria e crédito. |
| **XGBoost** | Estado da arte clássico em tabulares. Rodado com e sem tuning, para separar o efeito do ajuste de hiperparâmetros. |

Outros baselines possíveis: LightGBM/CatBoost e uma MLP simples.

## Protocolo de avaliação

1. **Datasets**: variar tamanho (centenas a dezenas de milhares de linhas), proporção de categóricas, faltantes, presença de texto e desbalanceamento. Incluir tarefas de classificação binária (foco do estudo) e algumas de regressão.
2. **Splits**: validação cruzada repetida ou *hold-out* com várias sementes. Para dados com ordem temporal, split temporal.
3. **Pré-processamento**: o mesmo pipeline base para todos os modelos, mais as exceções que cada modelo exige. Cada exceção deve ser documentada.
4. **Tuning**: orçamento igual de busca de hiperparâmetros para os baselines. Os fundacionais entram com os valores padrão e, quando faz sentido, com fine-tuning.
5. **Métricas**: AUC-ROC, log-loss, Brier score e curva de calibração; KS em tarefas de crédito; RMSE/MAE em regressão.
6. **Custo**: tempo de treino e de inferência e uso de memória, na mesma máquina.
7. **Análise**: intervalos de confiança entre sementes, ranking médio entre datasets e curvas de desempenho versus tamanho do treino.

## Estrutura

```
TabTest/
├── tasks/              # protocolo: um YAML por dataset (tipo de problema, alvos, split)
│   ├── README.md       # formato do spec
│   ├── benchmarks/     # datasets públicos (download_datasets.py)
│   └── redistribuicao.yaml
├── data/               # datasets (não versionados)
├── src/
│   ├── tasks.py        # lê e valida os specs
│   ├── data.py         # carrega arquivos, cria alvos derivados, faz o split
│   ├── run.py          # roda modelos x tarefas x seeds
│   ├── tracking.py     # TensorBoard
│   ├── metrics.py, preprocessing.py
│   ├── models/         # wrappers: fit_predict(split, seed, ...) -> Prediction
│   └── ldm/            # Large Data Model tabular (pré-treino próprio)
├── runs/               # saída: TensorBoard + results.csv por experimento (não versionado)
└── checkpoints/
```

## Uso

Qualquer dataset entra no benchmark se estiver no formato padrão (CSV/Parquet, uma linha por observação) e tiver um spec YAML com o tipo de problema e a variável alvo. O formato está em [tasks/README.md](tasks/README.md).

```bash
python download_datasets.py --offline                       # benchmarks públicos -> data/
python -m src.run --list                                    # tarefas registradas
python -m src.run --describe --spec tasks/redistribuicao.yaml
python -m src.run --spec tasks/redistribuicao.yaml --models xgboost logistic --experiment redistrib_v1
python -m src.run --all --models xgboost logistic probit ridge --seeds 0 1 2 --experiment baselines
```

### Modelos fundacionais com datasets grandes

Por padrão, `tabpfn`, `tabpfn_ft` e `tabstar` usam **todo** o treino. `--max-train N` limita o número de linhas.

- **`tabpfn`** (só in-context learning): o contexto de cada passada tem 10k linhas. A V2 foi pré-treinada até esse tamanho, e contextos maiores pioraram nos testes. Para cobrir o treino inteiro, cada membro do ensemble recebe uma fatia diferente, com `n_estimators = max(8, ceil(coverage·N/10k))`. Com classe rara, cada fatia leva todos os positivos (`majority_downsample`) e a probabilidade é corrigida para o prior do treino. Em `redistrib_erro`, isso levou o AUC de 0,70 para 0,94.
- **`tabpfn_ft`**: fine-tuning dos pesos do TabPFN, com a mesma inferência por fatias. Custa cerca de 5x o `tabpfn`.
- **`tabstar`**: fine-tuning LoRA sobre todas as linhas, com custo linear em N por época (~40 s/época com 90k linhas numa A4500). O early stopping usa uma validação estratificada própria (10% do treino, até 20k linhas). A validação interna do TabSTAR tem só 1000 linhas e fica sem positivos em classes raras.

Qualquer parâmetro dos wrappers pode ser ajustado com `--model-arg modelo.param=valor`, por exemplo:

```bash
python -m src.run --spec tasks/redistribuicao.yaml --models tabpfn tabstar --tabpfn-version V2 \
    --model-arg tabpfn.coverage=3 tabpfn.sampling=stratified tabstar.max_epochs=20 tabstar.lora_batch=128
```

### TensorBoard

Todos os treinos gravam no mesmo diretório raiz (`runs/`), com uma subpasta por run:

```
runs/
  <experimento>/                  # --experiment (padrão: data-hora)
    config.json, results.csv
    <tarefa>/<modelo>/seed<k>/    # um run do TensorBoard
  pretrain/<nome>/                # python -m src.ldm.pretrain
```

```bash
tensorboard --logdir runs
```

- **HParams**: uma linha por (experimento, tarefa, modelo, seed) com as métricas de teste. É a tabela para comparar tudo lado a lado, com filtro e ordenação.
- **Scalars**: `test/*` (métricas finais), `time/seconds` e as curvas de treino. Em `curves/*` ficam o XGBoost (por árvore, treino e teste) e o TabSTAR (loss e métrica de validação por época). Em `finetune/*` ficam o LDM e o `tabpfn_ft`. O pré-treino aparece em `pretrain/*`. Para comparar só um subconjunto, filtre os runs por regex, por exemplo `adult_income/.*`.
- **Text**: descrição da tarefa, alvo e split.

Para registrar curvas de um modelo novo, aceite `writer=None` em `fit_predict` e use `writer.add_scalar(...)`. O `run.py` passa o writer automaticamente.

## Referências

Os resumos e comparações dos papers estão em `../LDM_study/comparacao_papers.pdf`, junto com os PDFs de TabPFN, TabSTAR, nuFormer e PRAGMA.
