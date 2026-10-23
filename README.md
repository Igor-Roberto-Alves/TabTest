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

## Estrutura planejada

```
TabTest/
├── README.md
├── data/          # datasets (não versionados)
├── src/           # carregamento, pré-processamento, wrappers de modelos
├── experiments/   # configs e scripts de cada experimento
├── results/       # métricas e tabelas geradas
└── notebooks/     # análise e figuras
```

A estrutura ainda não existe. Ela será criada conforme os experimentos avançarem.

## Referências

Os resumos e comparações dos papers estão em `../LDM_study/comparacao_papers.pdf`, junto com os PDFs de TabPFN, TabSTAR, nuFormer e PRAGMA.
