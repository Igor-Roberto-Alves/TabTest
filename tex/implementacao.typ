#set document(title: "TabTest — o que foi implementado")
#set page(paper: "a4", margin: 2.3cm, numbering: "1")
#set text(lang: "pt", font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show raw.where(block: true): block.with(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%)
#show link: underline

#align(center)[
  #text(size: 18pt, weight: "bold")[TabTest: benchmark tabular e Large Data Model]
  #v(4pt)
  Documentação de tudo que foi implementado no repositório \
  #text(size: 9pt)[23 de outubro de 2026]
]

#outline(depth: 2)
#pagebreak()

= Visão geral

O projeto tem três partes:

+ *Dados* (`download_datasets.py`): baixa datasets tabulares do scikit-learn e do OpenML e salva cada um como CSV em `data/`.
+ *Benchmark* (`src/`): define problemas de classificação e regressão sobre esses CSVs e compara XGBoost, regressão logística, regressão probit, Ridge, TabPFN e TabSTAR com um mesmo split e as mesmas métricas.
+ *Large Data Model (LDM) tabular* (`src/ldm/`): um único Transformer pré-treinado em muitas tabelas, que completa células mascaradas. Nenhum parâmetro depende de uma tabela específica, então ele se aplica a tabelas que nunca viu. O objetivo de treino é de difusão mascarada.

Aqui, *LDM* significa *Large Data Model*: um modelo fundacional para dados tabulares. Não tem relação com _Latent Diffusion Model_, porque não há autoencoder nem espaço latente. A difusão usada é discreta e acontece direto nas células.

```
TabTest/
  download_datasets.py     baixa os datasets -> data/*.csv
  requirements.txt
  src/
    tasks.py               15 problemas sugeridos
    data.py                carrega o CSV, codifica o alvo, faz o split
    preprocessing.py       one-hot/escala, category, ordinal
    metrics.py             métricas por tipo de tarefa
    run.py                 CLI do benchmark
    models/
      base.py              dataclass Prediction
      xgb.py  logistic.py  probit.py  ridge.py
      tabpfn_model.py  tabstar_model.py  ldm_model.py
    ldm/
      text.py              texto -> ids de hash (nomes de colunas e categorias)
      table.py             TableEncoder: bins, categorias, encode/decode, meta
      model.py             TabularLDM: Transformer independente da tabela + heads
      diffusion.py         MaskedDiffusion: loss, impute, generate, predict
      corpus.py            TableCorpus: lotes de várias tabelas
      corpus_download.py   34 tabelas do OpenML -> data/corpus/
      pretrain.py          CLI de pré-treino + checkpoints
      predict.py           adaptação a tabela nova + predição
      README.md
  tex/implementacao.typ    este documento
```

*Notação.* $X in RR^(n times p)$ é a matriz de features, com $n$ linhas e $p$ colunas. $y$ é o vetor alvo e $x_i$ é a $i$-ésima linha. $K$ é o número de classes. No LDM, $C$ é o número de colunas de uma tabela (alvo incluído) e $K_c$ é o número de opções da coluna $c$.

= Dados: `download_datasets.py`

Para cada dataset, o script chama o loader do scikit-learn com `as_frame=True` e salva o `frame` (features e alvo) em `data/<nome>.csv`.

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.4pt, inset: 5pt,
  table.header[*Origem*][*Datasets*][*Observação*],
  [Embutidos (sem internet)], [iris, wine, breast\_cancer, diabetes, linnerud], [vêm com o pacote],
  [`fetch_*` / OpenML], [california\_housing, adult, titanic, house\_prices, credit\_g, bike\_sharing, kc\_house, covertype], [cache em `~/scikit_learn_data`],
)

- `--offline` baixa só os embutidos, `--list` lista os disponíveis, e também é possível passar nomes avulsos.
- Se um download falha, o script mostra o erro e segue para o próximo.
- *Correção aplicada:* o parser `pandas` de ARFF do scikit-learn quebra no titanic, no bike\_sharing e no kc\_house com `TypeError` em `strip_single_quotes`. A função `openml()` tenta o parser padrão e, se der `TypeError`, repete com `parser="liac-arff"`.

= Problemas sugeridos: `src/tasks.py`

Cada `Task` (dataclass imutável) informa o CSV, a coluna alvo e o tipo (`binary`, `multiclass` ou `regression`). Também pode trazer colunas a descartar (`drop`), uma função que cria um alvo derivado (`derive`) e um limite de linhas (`max_rows`).

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.4pt, inset: 4pt,
  table.header[*Tarefa*][*Tipo*][*Alvo*][*Comentário*],
  [adult\_income], [binária], [`class`], [renda > 50K; mistura num/cat, nulos],
  [titanic\_survival], [binária], [`survived`], [descarta `boat` e `body`, que entregam a resposta, e também `name`, `ticket`, `home.dest`],
  [credit\_risk], [binária], [`class`], [German Credit, quase tudo categórico],
  [breast\_cancer], [binária], [`target`], [30 features numéricas, classes quase separáveis],
  [california\_expensive], [binária], [`expensive`], [derivado: $"MedHouseVal" > "mediana"$],
  [iris, wine], [multiclasse], [`target`], [testes de sanidade, 3 classes],
  [covertype], [multiclasse], [`Cover_Type`], [7 classes, subamostrado para 20 000 linhas],
  [california\_price], [regressão], [`MedHouseVal`], [preço mediano por bairro],
  [house\_prices], [regressão], [`SalePrice`], [Ames, 79 features, muitos nulos],
  [kc\_house\_price], [regressão], [`price`], [King County],
  [bike\_demand], [regressão], [`count`], [aluguéis por hora, efeitos de calendário/clima],
  [diabetes], [regressão], [`target`], [só 442 linhas, um regime de poucos dados],
  [adult\_hours], [regressão], [`hours-per-week`], [alvo alternativo do mesmo dataset],
  [titanic\_fare], [regressão], [`fare`], [preço da passagem],
)

= Pipeline comum

== Carregamento e split: `src/data.py`

+ `load_frame` lê o CSV, aplica `derive` e remove as colunas de `drop`.
+ `load_task` remove as linhas com alvo nulo e subamostra se houver `max_rows`.
+ Na classificação, o alvo passa a texto e é codificado com `LabelEncoder` para inteiros $0, dots, K-1$. Os rótulos originais ficam em `Split.classes`, na mesma ordem.
+ O split treino/teste é 80/20 e estratificado na classificação. A semente padrão é 0.
+ `subsample(X, y, n)` limita o treino a $n$ linhas sem reposição. TabPFN e TabSTAR usam essa função.

Uma coluna é categórica quando seu dtype não é numérico. Essa é a regra das propriedades `Split.cat_cols` e `Split.num_cols`.

== Pré-processamentos: `src/preprocessing.py`

- `linear_preprocessor` (lineares):
  - numéricas: imputação pela mediana, depois padronização $z = (x - mu) \/ sigma$;
  - categóricas: imputação pela moda, depois one-hot. Categorias com menos de 10 ocorrências viram "infrequentes".
  - Com `drop_first=True`, usado no probit, a primeira dummy de cada variável é descartada para evitar a armadilha das dummies: com intercepto, as $m$ dummies de uma variável somam 1 e a matriz fica singular.
- `to_category` (XGBoost): converte colunas de texto para `category`, usando as categorias vistas no treino. Categoria nova no teste vira NaN.
- `ordinal_encode` (TabPFN): devolve uma matriz numérica com as categóricas como inteiros, mantém os NaN e retorna os índices das colunas categóricas.

== Métricas: `src/metrics.py`

- *Classificação:* acurácia, acurácia balanceada (média do recall por classe), log-loss $-1/n sum_i log hat(p)_(i, y_i)$ e ROC-AUC (one-vs-rest no multiclasse).
- *Regressão:* $"RMSE" = sqrt(1/n sum_i (y_i - hat(y)_i)^2)$, MAE e $R^2 = 1 - (sum_i (y_i - hat(y)_i)^2) / (sum_i (y_i - macron(y))^2)$.

== Interface dos modelos

Cada módulo em `src/models/` expõe duas coisas:
- `KINDS`: os tipos de tarefa que o modelo suporta;
- `fit_predict(split, seed, **kw)`: devolve um `Prediction(y_pred, proba, model)`.

`run.py` importa cada módulo sob demanda, então um pacote ausente só faz aquele modelo ser pulado. Modelos incompatíveis com a tarefa também são pulados, e qualquer exceção vira "FALHOU" sem parar o benchmark.

```
python -m src.run --list
python -m src.run --task adult_income titanic_survival
python -m src.run --all --models xgboost logistic probit ridge --out results.csv
python -m src.run --task iris --models tabpfn --tabpfn-version V2
```

= Modelos

== XGBoost: `models/xgb.py`

Gradient boosting de árvores. O modelo é uma soma de $M$ árvores, $hat(y)_i = sum_(m=1)^M f_m (x_i)$. Cada nova árvore $f_m$ minimiza uma aproximação de segunda ordem da perda regularizada:
$ cal(L)^((m)) approx sum_i [g_i f_m (x_i) + 1/2 h_i f_m (x_i)^2] + gamma T + 1/2 lambda norm(w)^2 , $
- $g_i$ e $h_i$ são o gradiente e a hessiana da perda em relação à predição atual;
- $T$ é o número de folhas e $w$ são os pesos das folhas.

Configuração usada:
- 500 árvores, `learning_rate` 0.05, profundidade 6, `subsample` e `colsample_bytree` 0.8, `tree_method="hist"`;
- `enable_categorical=True`: as categóricas são divididas nativamente, sem one-hot;
- os NaN seguem uma direção padrão aprendida em cada split.

Não há pesos pré-treinados: o modelo é treinado do zero.

== Regressão logística: `models/logistic.py`

No caso binário, $P(y = 1 | x) = sigma(beta_0 + x^top beta)$, com $sigma(z) = 1 \/ (1 + e^(-z))$. No multiclasse, usa softmax:
$ P(y = k | x) = exp(beta_(0k) + x^top beta_k) / (sum_(j=1)^K exp(beta_(0j) + x^top beta_j)) . $
O scikit-learn minimiza a log-verossimilhança negativa mais uma penalidade $1/(2C) norm(beta)^2$, com $C = 1$. Por isso a logística segue estável mesmo quando as classes são separáveis.

== Regressão probit: `models/probit.py`

A ligação passa a ser a CDF da normal padrão, $Phi$:
$ P(y = 1 | x) = Phi(x^top beta) , quad Phi(z) = integral_(-oo)^z 1/sqrt(2 pi) e^(-u^2 \/ 2) dif u . $
Uma forma equivalente usa uma variável latente: $y^* = x^top beta + epsilon$, com $epsilon tilde cal(N)(0, 1)$ e $y = bb(1)[y^* > 0]$. O estimador é o de máxima verossimilhança:
$ ell(beta) = sum_(i=1)^n [y_i log Phi(x_i^top beta) + (1 - y_i) log(1 - Phi(x_i^top beta))] . $

Detalhes da implementação (`ProbitClassifier`, compatível com o scikit-learn):
- usa `statsmodels.Probit` com intercepto (`add_constant`), ajustado por *L-BFGS*. O Newton padrão falha com Hessiana singular quando há colunas colineares, como `education` e `education-num` no adult;
- o probit é binário por natureza. No multiclasse, ajusto um modelo one-vs-rest por classe, $s_k = Phi(x^top beta_k)$, e renormalizo: $hat(p)_k = s_k \/ sum_j s_j$;
- `pred.model[-1].results_[k].summary()` dá coeficientes, erros-padrão e p-valores. É a vantagem inferencial do probit sobre os modelos de caixa-preta;
- não há regularização. Quando as classes são (quase) separáveis, como no breast\_cancer, $norm(beta) -> oo$ e as probabilidades vão a 0 ou 1, o que explode o log-loss.

A relação entre os dois modelos: $Phi(z) approx sigma(1.6 z)$. Por isso as acurácias de logística e probit costumam ficar próximas, com coeficientes cerca de 1.6 vezes menores no probit.

== Ridge: `models/ridge.py`

Logística e probit não servem para alvos contínuos, então a regressão tem um baseline linear próprio:
$ hat(beta) = arg min_beta norm(y - X beta)^2 + alpha norm(beta)^2 = (X^top X + alpha I)^(-1) X^top y , $
com $alpha in {10^(-3), dots, 10^3}$ escolhido por validação cruzada (`RidgeCV`). Usa o mesmo pré-processamento linear.

== TabPFN: `models/tabpfn_model.py`

O TabPFN é um Transformer pré-treinado em milhões de datasets sintéticos amostrados de um _prior_ sobre processos geradores (modelos causais estruturais). Ele aprende a aproximar a distribuição preditiva posterior:
$ p(y_* | x_*, cal(D)_"treino") = integral p(y_* | x_*, phi) thin p(phi | cal(D)_"treino") dif phi , $
em que $phi$ indexa o processo gerador dos dados. O treino entra como *contexto* da atenção, então `fit` não altera nenhum peso: é _in-context learning_ em um único forward.

- *Pesos:* ficam no Hugging Face, na org `Prior-Labs` (`tabpfn_3`, `tabpfn_3_5`, `tabpfn_2_5`, ...), e são baixados no primeiro uso.
  - Da versão 2.5 em diante são _gated_ e não comerciais: é preciso aceitar a licença na página e fazer `hf auth login` (ou definir `HF_TOKEN`).
  - A v2 é Apache 2.0 com atribuição e pode ser escolhida com `--tabpfn-version V2`.
- *Entrada:* matriz `ordinal_encode` com NaN preservado e `categorical_features_indices`. Também passo `ignore_pretraining_limits=True`.
- *Limites:* o custo da atenção cresce com o tamanho do contexto, por isso o treino fica limitado a 10 000 linhas (`MAX_TRAIN`). Em CPU, acima de cerca de 1000 linhas, é preciso definir `TABPFN_ALLOW_CPU_LARGE_DATASET=1`.

== TabSTAR: `models/tabstar_model.py`

Modelo fundacional tabular (Arazi, Shapira e Reichart, NeurIPS 2025) feito para tabelas com texto.

- *Como funciona:* cada célula é _verbalizada_, ou seja, nome da coluna e valor viram texto. Um encoder de texto pré-treinado gera as representações, e os rótulos do alvo também entram como texto ("target-aware"). A arquitetura não tem parâmetros específicos de dataset.
- *Pesos:* `alana89/TabSTAR`, baixados automaticamente no `fit()`.
- *Treino:* ao contrário do TabPFN, o `fit` faz *fine-tuning LoRA*: até 50 épocas, paciência 5, $r = 32$, `lr` $10^(-3)$. Por isso a GPU é praticamente obrigatória, e o treino fica limitado a 5 000 linhas (`MAX_TRAIN`).
- *Entrada:* um `DataFrame` bruto e uma `Series`. Passo os rótulos originais (`split.classes[y]`), não os inteiros, porque o modelo usa o significado do texto.
- *Probabilidades:* assumo que as colunas de `predict_proba` seguem a ordem ordenada dos rótulos. Se o formato não bater, só as predições duras são avaliadas.
- *Ambiente:* requer Python 3.10–3.12.

= Resultados obtidos

Execução real em CPU, com Python 3.12 e semente 0. TabPFN e TabSTAR ainda não rodaram porque `torch`, `tabpfn` e `tabstar` não estavam instalados.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.4pt, inset: 5pt, align: (left, left, right, right, right, right),
  table.header[*Tarefa*][*Métrica*][*XGBoost*][*Logística*][*Probit*][*Ridge*],
  [iris], [acurácia], [0.933], [0.967], [0.967], [–],
  [breast\_cancer], [ROC-AUC], [0.986], [0.996], [0.948], [–],
  [titanic\_survival], [ROC-AUC], [0.864], [0.863], [0.860], [–],
  [adult\_income], [ROC-AUC], [0.927], [0.904], [0.904], [–],
  [diabetes], [$R^2$], [0.236], [–], [–], [0.339],
  [california\_price], [$R^2$], [0.854], [–], [–], [0.594],
  [titanic\_fare], [$R^2$], [0.467], [–], [–], [0.303],
)

Leitura dos resultados:
- O XGBoost ganha onde há não-linearidade e interações (california, adult).
- Os lineares empatam ou vencem com poucos dados e sinal aproximadamente linear (diabetes, iris, breast\_cancer).
- No breast\_cancer, o probit teve log-loss de 1.90, contra 0.08 da logística. É o efeito da separação sem regularização descrito acima.

= Large Data Model tabular: `src/ldm/`

== Ideia

Um *Large Data Model* é o análogo tabular de um modelo de linguagem fundacional: *um único modelo, pré-treinado em muitas tabelas*, que depois se aplica a uma tabela nova com pouco ou nenhum treino. O desafio é que cada tabela tem colunas diferentes, em número, nome, tipo e escala. Por isso o requisito central da arquitetura é:

#align(center)[_nenhum parâmetro pode depender de uma tabela específica._]

Na primeira versão do backbone, as colunas e os valores eram ids aprendidos por tabela (um vocabulário com _offset_ por coluna e uma head por coluna). Um modelo assim só funciona na tabela em que foi treinado. A versão atual troca todos esses ids por *descrições* que fazem sentido em qualquer tabela:

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt, inset: 5pt,
  table.header[*Objeto*][*Antes (específico da tabela)*][*Agora (independente da tabela)*],
  [coluna $c$], [$E_"coluna" [c]$, um id], [embedding do *texto do nome* da coluna],
  [categoria], [id com _offset_ por coluna], [embedding do *texto da categoria*],
  [bin numérico $b$], [id com _offset_ por coluna], [posição do bin no CDF ($q$) e mediana padronizada ($z$)],
  [head de saída], [uma matriz $W_c$ por coluna], [*produto interno* com os embeddings das opções],
)

Cada linha é uma sequência de $C$ células, e o modelo aprende a completar células mascaradas com uma taxa de máscara que varia de 0 a 1 (difusão discreta com estado absorvente, MDLM). O mesmo modelo serve para:
- *predição:* mascarar só o alvo dá $p("alvo" | "resto")$;
- *imputação:* mascarar as células faltantes;
- *geração sintética:* começar com tudo mascarado.

== Texto sem vocabulário fixo: `text.py`

Nomes de colunas e categorias viram um saco de *palavras* e *trigramas de caracteres*. O camelCase e o snake\_case são separados antes:
$ "MedHouseVal" -> {"w:med", "w:house", "w:val", "c:<me", "c:med", "c:ed>", dots} $

Cada pedaço é mapeado por hash (`crc32`, estável entre processos) para um de $B = 2^16$ ids, e o id 0 fica reservado para padding. O embedding de um texto $s$ com pedaços $u_1, dots, u_m$ é
$ "TextEmb"(s) = W ("LN"(1/m sum_(i=1)^m E_"hash" [h(u_i)])) + b . $
Um nome nunca visto cai no mesmo espaço: "hours-per-week" compartilha pedaços com "hours", e "MedHouseVal" com "house\_value". Trigramas dão robustez a abreviações e variações de grafia.

== Codificador por tabela: `table.py`

O `TableEncoder` é ajustado em cada tabela. Ele só calcula estatísticas, sem nenhum parâmetro treinável.

- *Tipo:* numérica com mais de 10 valores distintos é `NUM`; o resto, incluindo texto e booleanos, é `CAT`.
- *Numérica:* até 32 bins por quantis do treino ($q_1 < dots < q_(K_c - 1)$, colapsando quantis repetidos). Cada bin $b$ é descrito por:
  - sua posição no CDF, $q_b = (N_(< b) + N_b \/ 2) \/ N$, onde $N_(< b)$ é o número de valores nos bins anteriores e $N_b$ o número no próprio bin;
  - sua mediana padronizada, $z_b = ("mediana"_b - mu) \/ sigma$, cortada em $[-5, 5]$.

  A célula guarda o índice do bin e o valor contínuo $z = (x - mu) \/ sigma$.
- *Categórica:* mantém as 63 categorias mais frequentes com pelo menos 2 ocorrências. As demais viram `<other>`. A coluna alvo de uma classificação nunca é cortada (`exact`).
- *Índice local:* $0 = $ `[NULL]` (valor faltante) e $1, dots, K_c$ = opção da coluna.
- `meta()` devolve a descrição da tabela que o modelo consome: tipo, número de opções, ids do nome, ids das categorias, $q$ e $z$ de cada bin.

== Embeddings das opções e da entrada: `model.py`

Para cada coluna $c$ e opção $k$, o embedding da opção é
$ o_(c k) = cases(
  "TextEmb"("categoria"_k) & "se" c "é categórica",
  "MLP"(phi(q_(c k)) ⊕ phi(z_(c k) \/ 5)) & "se" c "é numérica",
) , quad o_(c 0) = e_"[NULL]" , $
com features de Fourier $phi(x) = [x, sin(2^j pi x), cos(2^j pi x)]_(j < 8)$ e $⊕$ a concatenação. Como $q$ é a posição no CDF, "o bin mais alto" tem o mesmo significado em qualquer coluna numérica de qualquer tabela.

A entrada da célula $(n, c)$, com índice local $l_(n c)$, é
$ h_(n c)^((0)) = v_(n c) + underbrace("TextEmb"("nome"_c) + E_"tipo" [c], "coluna") + "MLP"("senoidal"(r_n)) , $
$ v_(n c) = cases(
  e_"[MASK]" & "se a célula está mascarada",
  W_"in" o_(c, l_(n c)) + bb(1)_"num. visível" dot "MLP"(phi(z_(n c) \/ 5)) & "caso contrário",
) $
- $r_n$ é a fração de células mascaradas da linha $n$;
- o termo $"MLP"(phi(z))$ injeta o valor numérico exato, além do bin.

Não há embedding posicional: a atenção é equivariante a permutações das colunas, e as colunas de uma tabela não têm ordem.

*Encoder.* $L$ blocos *pre-LN* bidirecionais, sem máscara causal: $h <- h + "MHA"("LN"(h))$, seguido de $h <- h + "MLP"("LN"(h))$. Os padrões do pré-treino são $d = 256$, $L = 6$ e 8 cabeças; nos testes em CPU usei $d = 128$, $L = 4$, cerca de 9 M de parâmetros, dos quais 8.4 M estão na tabela de hash.

== Head de autocomplete de máscaras

A head não tem pesos por coluna. Ela pontua o estado da célula contra os embeddings das opções válidas da coluna:
$ "logit"_(n c k) = (chevron.l W_q h_(n c)^((L)), thin W_"out" o_(c k) chevron.r) / sqrt(d) , quad k in {0, dots, K_c} , $
- opções inexistentes ($k > K_c$) recebem $-oo$;
- a head se aplica a qualquer coluna de qualquer tabela, inclusive com categorias nunca vistas no pré-treino, desde que o texto seja informativo;
- o mesmo $o_(c k)$ entra na entrada (via $W_"in"$) e na saída (via $W_"out"$), um _weight tying_ análogo ao dos modelos de linguagem.

A *head de regressão*, $hat(z)_(n c) = w^top h_(n c)^((L)) + b$, prevê o valor contínuo padronizado das numéricas.

== Objetivo de difusão: `diffusion.py`

*Forward.* Com o schedule $alpha(t)$ (probabilidade de a célula continuar visível),
$ alpha_"cos" (t) = cos(pi t \/ 2) quad ("padrão") , quad alpha_"linear" (t) = 1 - t , $
cada célula é mascarada de forma independente:
$ q(x_t^c | x_0^c) = alpha(t) thin delta_(x_0^c) + (1 - alpha(t)) thin delta_"[MASK]" . $

*Loss.* Sorteio $t tilde cal(U)(10^(-3), 1)$ e uso a cota contínua do ELBO do MDLM, mais o MSE da head de regressão:
$ cal(L)(theta) = EE [ w(t) dot 1/C sum_(c in cal(M)) - log p_theta (x_0^c | x_t) ] + lambda thin EE [ 1/(|cal(M)_"num"|) sum_(c in cal(M)_"num") (hat(z)_c - z_c)^2 ] , $
- $cal(M)$ é o conjunto das células mascaradas e $cal(M)_"num"$ o das numéricas mascaradas não nulas;
- $w(t) = -alpha'(t) \/ (1 - alpha(t))$, que vale $1 \/ t$ no schedule linear e $(pi \/ 2) sin(pi t \/ 2) \/ (1 - cos(pi t \/ 2))$ no cosseno, limitado a $10^3$;
- $lambda = 1$;
- a cross-entropy é sobre as $K_c + 1$ opções da coluna, incluindo `[NULL]`. Assim o modelo também aprende *quando* um valor costuma faltar.

*Condicionamento.* O modelo recebe a fração de máscara *observada* $r$, e não o $t$ sorteado. Na imputação, parte da linha é conhecida e $t$ perde o sentido, mas $r$ continua bem definido.

*Reverse.* As células desconhecidas começam mascaradas. Para $t = 1, (S-1)\/S, dots, 1\/S$ e $s = t - 1\/S$:
+ calcula $p_theta (x_0^c | x_t)$;
+ amostra um candidato (com temperatura) ou toma o argmax;
+ revela cada célula mascarada com probabilidade
  $ P("revelar") = (alpha(s) - alpha(t)) / (1 - alpha(t)) , $
  que é o posterior exato do processo absorvente (estratégia `random`). A alternativa `confidence` revela as mais confiantes.

Numéricas reveladas recebem como $z$ a mediana padronizada do bin escolhido. Células reveladas não voltam a ser mascaradas.

*Predição.* `predict` mascara só as colunas alvo e roda um forward, com `[NULL]` proibido:
- *classificação:* a probabilidade de cada classe é o softmax sobre as opções do alvo;
- *regressão:* $hat(y) = hat(z) sigma + mu$, a head de regressão na escala original. A alternativa é o valor esperado dos bins, $sum_b p_b "mediana"_b$ (`expected_bin_value`).

== Pré-treino em várias tabelas: `corpus.py`, `pretrain.py`

- *Corpus:* `corpus_download.py` baixa 34 tabelas do OpenML para `data/corpus/`, com prefixo `openml_`. Entre elas: phoneme, spambase, bank-marketing, electricity, MagicTelescope, mushroom, letter, wine\_quality, pol, elevators, nursery e outras.
  - Tabelas de avaliação ficam de fora.
  - O `houses` do OpenML foi removido, porque é o California Housing e vazaria as tarefas `california_*`.
- *Amostragem:* cada passo sorteia uma tabela com probabilidade $prop n_"linhas"^(0.5)$, o que dá mais peso relativo às pequenas. Depois sorteia um lote de linhas e um subconjunto de até 64 colunas.
  - Um lote tem uma tabela só, então todas as linhas compartilham o mesmo `meta`.
  - Cortar colunas também é aumento de dados: o modelo aprende a prever a partir de contextos parciais.
- *Otimização:* AdamW, warmup linear seguido de cosseno, _clipping_ em 1.0.
- *Checkpoint:* guarda a configuração, os pesos, o schedule e a *lista das tabelas vistas*. O `ldm_model` usa essa lista para avisar sobre vazamento.

== Adaptação a uma tabela nova: `predict.py`, `models/ldm_model.py`

+ Ajusta o `TableEncoder` no treino da tabela nova, forçando o tipo do alvo (`CAT` na classificação, `NUM` na regressão).
+ *Zero-shot* (`--ldm-steps 0`): prediz direto com o modelo pré-treinado.
+ *Fine-tuning* (padrão de 300 passos, `lr` $10^(-4)$): uma cópia do modelo treina na tabela nova, alternando duas losses:
  - com probabilidade 0.5, a loss de difusão comum, que preserva o modelo da tabela inteira;
  - no resto, lotes em que o alvo é sempre mascarado (a loss da tarefa, com média uniforme).
+ *Sem checkpoint:* o mesmo procedimento treina do zero (2000 passos, `lr` $10^(-3)$). É a ablação que mede o ganho do pré-treino.

```
python -m src.ldm.corpus_download
python -m src.ldm.pretrain --dirs data/corpus --steps 20000 --save checkpoints/ldm.pt
python -m src.run --task adult_income --models ldm --ldm-checkpoint checkpoints/ldm.pt --ldm-steps 0
python -m src.run --task adult_income --models ldm xgboost --ldm-checkpoint checkpoints/ldm.pt
python -m src.run --task adult_income --models ldm        # do zero
```

== Resultados do LDM

*Configuração:*
- *Pré-treino:* checkpoint `checkpoints/ldm_small.pt`. Modelo com $d = 128$, $L = 4$ e 9.3 M de parâmetros, treinado em CPU por 3000 passos (lote 128, cerca de 10 min) nas 34 tabelas de `data/corpus` (256 mil linhas). A cross-entropy nas células mascaradas caiu de 2.44 para cerca de 2.05.
- *Sem vazamento:* nenhuma tabela de avaliação está no corpus.
- *Comparação justa:* fine-tuning e treino do zero usam o *mesmo número de passos* (300) e a mesma arquitetura, então a diferença entre eles mede só o efeito do pré-treino.

#text(size: 9.5pt)[#table(
  columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr),
  stroke: 0.4pt, inset: 4pt, align: (left, left, right, right, right, right, right),
  table.header[*Tarefa*][*Métrica*][*LDM \ zero-shot*][*LDM \ pré-tr.+FT*][*LDM \ do zero*][*XGBoost*][*Linear*],
  [titanic\_survival], [ROC-AUC], [0.429], [*0.825*], [0.737], [0.864], [0.863],
  [credit\_risk], [ROC-AUC], [0.532], [*0.654*], [0.464], [0.811], [0.830],
  [breast\_cancer], [ROC-AUC], [0.089], [0.987], [*0.990*], [0.986], [0.996],
  [diabetes], [$R^2$], [0.094], [0.244], [*0.325*], [0.236], [0.339],
  [california\_price], [$R^2$], [0.054], [*0.540*], [0.052], [0.854], [0.594],
)]

"Linear" é a logística na classificação e o Ridge na regressão. Em negrito, o melhor entre as três versões do LDM.

*Leitura dos resultados:*
- *O pré-treino ajuda.* Com o mesmo orçamento de 300 passos, o modelo pré-treinado supera o treinado do zero em 3 de 5 tarefas, com destaque para o california (R² 0.54 contra 0.05) e o titanic (AUC 0.83 contra 0.74). O modelo do zero ainda nem saiu do lugar no california, enquanto o pré-treinado já chega perto do Ridge.
  - No diabetes (442 linhas, 10 features numéricas), o modelo do zero foi melhor. Os bins e o espaço numérico aprendidos no corpus não se transferiram bem para essa tabela.
- *O zero-shot ainda não funciona.* Fica no nível do acaso, e no breast\_cancer o AUC de 0.089 está *invertido*. É o esperado neste estágio:
  - o alvo é o texto "0"/"1", sem semântica;
  - os embeddings de hash não têm significado prévio;
  - 34 tabelas não bastam para o modelo aprender relações gerais.

  O zero-shot real exige um corpus muito maior, um encoder de texto semântico e, de preferência, linhas de treino como contexto (TabPFN).
- *O LDM ainda perde para os modelos clássicos.* Ele fica abaixo do XGBoost e dos lineares, exceto no breast\_cancer e no diabetes, onde empata. O pré-treino foi pequeno (10 min de CPU, $d = 128$) e o fine-tuning curto, sem ajuste de hiperparâmetros. O resultado valida o pipeline, mas ainda não demonstra um modelo competitivo. Os próximos passos estão na seção de limitações.

= Ambiente e instalação

O Python do sistema (3.14) não tem `python3-venv`, e o TabSTAR exige Python 3.10–3.12. O caminho recomendado é o `uv`:

```
uv venv -p 3.12 .venv312
uv pip install -p .venv312/bin/python -r requirements.txt
.venv312/bin/python download_datasets.py
.venv312/bin/python -m src.ldm.corpus_download
.venv312/bin/python -m src.run --all --out results.csv
```

Para o LDM, o pré-treino deve rodar em GPU. Em CPU, cada passo atualiza densamente toda a tabela de hash ($2^16 times d$) no AdamW e fica lento.

= Limitações e próximos passos

- *Benchmark:*
  - não há busca de hiperparâmetros nem early stopping no XGBoost;
  - avaliação com um único split: o ideal é repetir com várias sementes ou usar CV e reportar média ± desvio;
  - o probit não tem regularização, então em dados separáveis o ideal é penalizar ou usar Firth.
- *TabPFN e TabSTAR:* ainda não foram executados. Faltam GPU, `torch` e o aceite da licença dos pesos do TabPFN v2.5+.
- *LDM:*
  - *corpus pequeno:* 34 tabelas é pouco para um modelo fundacional. O próximo passo é escalar para milhares de tabelas (OpenML, Kaggle) e/ou tabelas sintéticas geradas por modelos causais estruturais, como faz o TabPFN;
  - *texto:* o embedding é um saco de n-gramas com hash, sem semântica real. Um encoder de frases pré-treinado faria "salary" ficar perto de "income", como no CARTE e no TabSTAR;
  - *contexto entre linhas:* cada linha é processada isoladamente. Atender também a outras linhas da tabela (atenção entre linhas, no estilo TabPFN) daria aprendizado em contexto;
  - não há validação nem early stopping no fine-tuning, e a qualidade das amostras sintéticas não foi avaliada;
  - colunas de texto livre de alta cardinalidade, como `name` no titanic, viram quase tudo `<other>`.

= Referências

- Chen & Guestrin (2016). _XGBoost: A Scalable Tree Boosting System_. KDD.
- Hollmann et al. (2025). _Accurate predictions on small data with a tabular foundation model_. Nature. Pesos em #link("https://huggingface.co/Prior-Labs")[huggingface.co/Prior-Labs].
- Arazi, Shapira & Reichart (2025). _TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields_. NeurIPS. #link("https://github.com/alanarazi7/TabSTAR")[github.com/alanarazi7/TabSTAR].
- Sahoo et al. (2024). _Simple and Effective Masked Diffusion Language Models_ (MDLM). NeurIPS.
- Austin et al. (2021). _Structured Denoising Diffusion Models in Discrete State-Spaces_ (D3PM). NeurIPS.
- Kim, Grinsztajn & Varoquaux (2024). _CARTE: Pretraining and Transfer for Tabular Learning_. ICML. (Colunas e valores descritos por texto para transferência entre tabelas.)
- Zhu et al. (2023). _XTab: Cross-table Pretraining for Tabular Transformers_. ICML.
