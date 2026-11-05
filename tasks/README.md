# Protocolo de datasets

Para rodar os modelos em um dataset novo, coloque os dados em um formato padrão e escreva um spec YAML. Nenhuma alteração em código é necessária.

## 1. Formato dos dados

- Um ou mais arquivos `.csv` (com cabeçalho) ou `.parquet`.
- Uma linha por observação e uma coluna por variável. Vários arquivos com as mesmas colunas são concatenados, por exemplo um arquivo por dia.
- As colunas alvo estão no arquivo ou podem ser calculadas a partir dele (veja `expr`).
- Colunas numéricas são features numéricas. Colunas de texto são categóricas. Códigos numéricos (CNPJ, CEP etc.) devem ser marcados em `categorical` ou lidos como texto em `read`.

## 2. Spec

Cada arquivo `tasks/**/*.yaml` descreve um dataset e uma ou mais tarefas sobre ele. Os specs são descobertos automaticamente. Um spec fora de `tasks/` também funciona, passado com `--spec caminho.yaml`.

```yaml
dataset: meu_dataset              # obrigatório
description: texto livre
files: data/meu_dataset/*.csv     # obrigatório; caminho ou glob (lista aceita), relativo à raiz do repo
read: {sep: ";", decimal: ","}    # opcional: kwargs de pandas.read_csv

# Valem para todas as tarefas (a tarefa pode sobrescrever):
drop: [id_cliente, data_cadastro] # fora das features: identificadores e vazamento
categorical: [cep]                # numéricas que são códigos
datetime: [data_evento]           # vira _year/_month/_day/_dow/_hour
features: null                    # lista explícita de features (padrão: todas menos drop)
filter: "status != 'cancelado'"   # DataFrame.query, aplicado antes de tudo
max_rows: 100000                  # subamostra aleatória (null = tudo)
split:
  method: random                  # random | temporal | group
  test_size: 0.2
  column: null                    # temporal: coluna de data/ordem; group: coluna de grupo
  cutoff: null                    # temporal: column >= cutoff vai para teste

tasks:                            # obrigatório
  - name: inadimplencia_90d       # único entre todos os specs
    kind: binary                  # binary | multiclass | regression
    target: inad_90d              # coluna alvo
    description: ...
  - name: atraso_dias
    kind: regression
    target: log_atraso
    expr: log1p(dias_atraso)      # cria o alvo `target` por DataFrame.eval
    filter: dias_atraso.notna()
    drop: [inad_90d]              # soma ao drop do dataset
```

Regras:

- `drop`, `categorical` e `datetime` da tarefa **somam** às do dataset. As demais chaves (`filter`, `split`, `max_rows`, `features`) **sobrescrevem**. Em `split`, a sobrescrita é campo a campo.
- A coluna alvo nunca entra nas features. As colunas usadas para derivá-la em `expr` precisam ir para `drop`, senão vazam o alvo.
- Linhas com alvo nulo são removidas.
- O `split` random é estratificado na classificação. O temporal usa `cutoff` ou as últimas `test_size` linhas em ordem de `column`. O group garante que um mesmo grupo (cliente, contrato) não aparece no treino e no teste.
- `expr` e `filter` aceitam a sintaxe de `DataFrame.eval` e `DataFrame.query` com `engine="python"`: comparações, `and`/`or`, funções como `log1p`, métodos como `.str.startswith(...)`, `.notna()` e `.median()`.

Exemplos: [`benchmarks/`](benchmarks/) (datasets públicos) e [`redistribuicao.yaml`](redistribuicao.yaml) (arquivos diários, alvos derivados, split temporal).

## 3. Checar e rodar

```bash
python -m src.run --spec meu.yaml --describe     # tamanhos, distribuição do alvo, alertas (ids, constantes, nulos)
python -m src.run --spec meu.yaml --models xgboost logistic --seeds 0 1 2 --experiment v1
tensorboard --logdir runs
```
