# SHAP, explicado do zero

Este guia explica o que é SHAP, de onde vem a ideia, como o TreeSHAP calcula os valores dentro das árvores do XGBoost e como ler a saída do script `explain_shap_xgb.py`. Não é preciso saber nada de teoria dos jogos nem de explicabilidade para acompanhar. A ideia é chegar ao fim entendendo não só o que os números significam, mas também por que eles são calculados desse jeito.

---

## 1. O problema: por que o modelo deu essa resposta?

O XGBoost do AmEx recebe as 188 colunas do último extrato de um cliente e devolve uma probabilidade de default. Para um cliente, ele diz 0,80 e o crédito é negado. A pergunta natural é: **quais colunas levaram o modelo a 0,80?**

Num modelo linear, isso é fácil. Cada coluna é multiplicada por um peso, e a contribuição dela é esse produto. Num conjunto de centenas de árvores, não existe um peso por coluna. A mesma coluna aparece em várias árvores, em profundidades diferentes, e o efeito dela depende de quais outras colunas vieram antes no caminho. O saldo pode só importar quando o atraso é alto, por exemplo.

O SHAP resolve isso com uma ideia emprestada da economia: tratar a previsão como um prêmio que precisa ser dividido de forma justa entre as colunas que o produziram.

---

## 2. A ideia de Shapley: dividir um prêmio de forma justa

Em 1953, o economista Lloyd Shapley estudou o seguinte problema. Um grupo de pessoas trabalha junto e ganha um prêmio. Quanto cada uma merece receber?

Dividir igualmente parece injusto se alguém contribuiu muito mais. Medir "quanto cada um fez" também não é simples, porque as pessoas se complementam. Imagine três pessoas que ganham R$ 90 juntas:

- Ana sozinha conseguiria R$ 10.
- Bruno sozinho conseguiria R$ 10.
- Ana e Bruno juntos conseguem R$ 60, porque as habilidades deles se completam.
- Carla acrescenta R$ 30 a qualquer grupo em que entrar.

Quanto vale a Ana? Se ela entra primeiro, acrescenta R$ 10. Se entra depois do Bruno, acrescenta R$ 50 (de 10 para 60). A contribuição dela **depende da ordem**.

A solução de Shapley é olhar **todas as ordens possíveis** em que as pessoas poderiam entrar no grupo, medir quanto cada uma acrescenta no momento em que entra e tirar a média. Com três pessoas há 6 ordens. Fazendo a conta:

- Ana e Bruno recebem R$ 30 cada (às vezes acrescentam 10, às vezes 50; na média, 30).
- Carla recebe R$ 30 (acrescenta sempre 30).
- Soma: R$ 90, exatamente o prêmio.

Shapley provou que essa é a **única** forma de dividir que satisfaz quatro exigências razoáveis de justiça:

1. **Eficiência:** as partes somam o prêmio inteiro, nem mais nem menos.
2. **Simetria:** quem contribui igual em todas as situações recebe igual.
3. **Jogador nulo:** quem nunca acrescenta nada recebe zero.
4. **Aditividade:** se o prêmio vem de dois jogos somados, a parte de cada um é a soma das partes nos dois jogos.

A quarta propriedade é a que torna o método perfeito para árvores: um XGBoost é literalmente uma soma de árvores, então dá para calcular a explicação de cada árvore e somar.

### A fórmula, lida em português

Para quem quiser ver a fórmula, o valor de Shapley de um jogador *j* é:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(M-|S|-1)!}{M!}\,\big[f(S \cup \{j\}) - f(S)\big]$$

Traduzindo:

- *F* é o conjunto de todos os jogadores, e *M* é quantos são.
- *S* é um grupo que **não** contém *j*.
- *f(S)* é o prêmio que o grupo *S* consegue sozinho.
- *f(S ∪ {j}) − f(S)* é quanto *j* acrescenta quando entra num grupo que já tem *S*.
- A fração na frente é a chance de, numa ordem sorteada ao acaso, *j* entrar exatamente depois do grupo *S*.

Ou seja, a fórmula é só "a média do que *j* acrescenta, sobre todas as ordens de entrada", escrita de forma compacta.

---

## 3. Do prêmio para o modelo: o que é SHAP

Em 2017, Scott Lundberg e Su-In Lee mostraram que essa ideia serve para explicar qualquer modelo de aprendizado de máquina. Eles chamaram o método de **SHAP** (*SHapley Additive exPlanations*). A tradução é direta:

| Teoria dos jogos | Modelo |
|---|---|
| Jogadores | Colunas (D_59, B_3, P_2…) |
| Jogo | A previsão para **um** cliente |
| Prêmio | Previsão do cliente − previsão média do modelo |
| Parte de cada jogador | Valor SHAP de cada coluna |

O valor SHAP de uma coluna diz **quanto ela empurrou a previsão deste cliente para cima ou para baixo, em relação à média**. Um valor positivo aumenta o risco; um negativo diminui.

### Um exemplo

Suponha que o modelo, em média, dê risco 0,25, e que para um cliente dê 0,80. O SHAP divide os 0,55 de diferença entre as colunas:

| Coluna | Contribuição |
|---|---|
| B_3 (saldo) | +0,30 |
| D_59 (atraso) | +0,20 |
| P_2 (pagamento) | +0,10 |
| S_3 (gasto) | −0,05 |
| **Soma** | **+0,55** |

A média mais a soma das contribuições dá exatamente a previsão: 0,25 + 0,55 = 0,80. Essa é a propriedade de **eficiência** da seção anterior, e é o que torna o SHAP tão fácil de ler. Nada fica sem explicação, e nada é contado duas vezes.

### Atenção: log-odds, não probabilidade

O exemplo acima usa probabilidade para ficar intuitivo, mas **o XGBoost faz as contas em log-odds**. Um XGBoost de classificação soma as saídas das árvores para obter um número *z*, que pode ir de −∞ a +∞, e só no final aplica a sigmoide para virar probabilidade:

$$p = \frac{1}{1 + e^{-z}}$$

O SHAP divide o *z*, não o *p*. A soma é exata em log-odds:

$$\text{base} + \sum_j \phi_j = z$$

Por que não em probabilidade? Porque a sigmoide é curva. Um empurrão de +1 em log-odds muda p de 0,50 para 0,73 se o cliente estiver no meio, mas só de 0,95 para 0,98 se ele já estiver lá em cima. A mesma coluna, com o mesmo efeito no modelo, pareceria importar muito menos para um cliente já muito arriscado. Em log-odds isso não acontece.

Para ler em log-odds, uma referência útil:

| Log-odds | Probabilidade |
|---|---|
| −2 | 0,12 |
| −1 | 0,27 |
| 0 | 0,50 |
| +1 | 0,73 |
| +2 | 0,88 |

E uma regra prática: **somar +0,7 em log-odds é aproximadamente dobrar a chance** (a razão p / (1 − p)).

O script guarda tudo em log-odds e, na explicação de cada cliente, também mostra a probabilidade passo a passo (veja a seção 8).

---

## 4. A parte difícil: o que é "não conhecer" uma coluna?

Para aplicar Shapley, é preciso saber o "prêmio" de qualquer grupo de colunas, ou seja, qual seria a previsão **se o modelo só conhecesse algumas colunas**. Mas o modelo foi treinado para receber as 188. O que significa ele não conhecer B_3?

Aqui está o ponto central, e a ligação com o estudo de máscaras do dashboard. "Não conhecer" **não é** pôr NULL. NULL é um valor que o XGBoost conhece: ele aprendeu, no treino, para que lado mandar os clientes com dado faltante. Colocar NULL responde a pergunta "e se esse dado estivesse faltando?", que é outra coisa. Em colunas onde faltar tem significado (como em várias de atraso, D), o NULL mudaria a previsão por um motivo que não tem a ver com o valor real do cliente.

O SHAP define "não conhecer" como **calcular a previsão esperada quando aquele valor é desconhecido**: a média das previsões sobre os valores que a coluna poderia ter. É a mesma pergunta que o [MASK] responde no PRAGMA. A diferença é que no PRAGMA o modelo aprendeu no pré-treino a imaginar o valor escondido, e no XGBoost a média precisa ser calculada de fora.

Há duas formas de fazer essa média, que dão os dois modos do script.

### Modo *path-dependent* (o padrão)

Usa as **próprias árvores**. Cada nó de uma árvore do XGBoost guarda quantas amostras de treino passaram por ele (a *cobertura*, ou *cover*). Quando o caminho chega a um nó que divide por uma coluna desconhecida, a previsão segue **pelos dois ramos ao mesmo tempo**, e cada ramo pesa conforme a fração do treino que foi para ele.

Se 600 clientes do treino foram para a esquerda e 400 para a direita, a esquerda pesa 60% e a direita 40%. É a média das previsões que se teria sem saber o valor, usando a distribuição que a árvore viu no treino.

Vantagens: é exato, rápido e não precisa de nenhum dado extra. É o que o XGBoost calcula com `pred_contribs=True`.

Desvantagem: a média em cada nó usa só os clientes que **chegaram até ali**. Isso faz o valor depender de correlações entre colunas, e às vezes uma coluna que o modelo nem usa ganha um pouco de crédito por ser correlacionada com outra que ele usa.

### Modo *interventional* (`--interventional`)

Usa um **conjunto de clientes de referência** (o *fundo*, por padrão 200 clientes sorteados da validação). Para "não conhecer" B_3, o método troca o B_3 do cliente pelo B_3 de cada cliente do fundo, mantendo as outras colunas do cliente original, e tira a média das previsões.

O nome vem de "intervenção": é como se mexêssemos só naquela coluna, sem deixar as outras mudarem junto. Isso responde melhor à pergunta "quanto **o modelo** usa essa coluna", porque uma coluna que o modelo nunca consulta recebe exatamente zero.

Desvantagens: é mais lento e depende de qual fundo se escolhe. Por isso o script só calcula esse modo para os clientes do estudo, não para a amostra inteira.

Quando há colunas categóricas no modelo (no AmEx, por exemplo, D_63 e D_64), a biblioteca `shap` não consegue ler as divisões categóricas das árvores do XGBoost. Nesse caso, o script calcula o mesmo valor *interventional* por **permutação**: sorteia ordens de entrada das colunas e mede o acréscimo de cada uma, exatamente como na definição de Shapley. É uma estimativa, mas a soma continua exata. O campo `meta.interventional` da saída diz qual dos dois métodos foi usado.

### Qual usar?

Para a maioria das análises, os dois modos concordam bastante. No teste com dados sintéticos, a correlação entre eles ficou acima de 0,99. Quando discordam, a diferença costuma estar em colunas correlacionadas:

- **path-dependent** é mais "fiel aos dados": divide o crédito entre colunas que andam juntas.
- **interventional** é mais "fiel ao modelo": dá crédito só às colunas que o modelo de fato consulta.

Para comparar com o estudo de máscaras do PRAGMA, os dois servem. O *interventional* é o que mais se aproxima da ideia de esconder uma coluna e ver o que acontece.

---

## 5. O que o TreeSHAP resolve

Calcular Shapley "na força bruta" exige avaliar o modelo para todos os grupos possíveis de colunas. Com 188 colunas, são 2¹⁸⁸ grupos, um número com 57 dígitos. Impossível.

O **TreeSHAP** (Lundberg e colegas, 2018 e 2020) é um algoritmo que aproveita a estrutura das árvores para calcular os valores de Shapley **exatos** em tempo polinomial. Ele não percorre os 2¹⁸⁸ grupos. Em vez disso, desce cada caminho da árvore uma única vez e, a cada nó, mantém uma conta de "em quantos grupos, e com que peso, o caminho passaria por aqui". Para um XGBoost de algumas centenas de árvores, isso leva milissegundos por cliente.

Como o XGBoost é uma soma de árvores, e o Shapley de uma soma é a soma dos Shapleys (a propriedade de aditividade), basta calcular árvore por árvore e somar.

### Um exemplo feito à mão

Para ver o que acontece dentro de uma árvore, vamos calcular o SHAP numa árvore minúscula, com só duas colunas: B_3 (saldo) e P_2 (pagamento).

```
                  B_3 < 0 ?
              (1000 clientes no treino)
             /                     \
          sim                       não
      (600 clientes)            (400 clientes)
        P_2 < 0 ?                folha: +1,5
       /        \
     sim        não
    (300)      (300)
  folha: −1,0  folha: +0,2
```

Os números nas folhas são as saídas da árvore, em log-odds. O cliente que queremos explicar tem **B_3 = 1** e **P_2 = −1**.

**Passo 1: a previsão do cliente.** B_3 = 1 não é menor que 0, então o cliente vai para a direita e cai na folha +1,5. Previsão: **1,5**.

**Passo 2: a previsão quando não se conhece nada** (a base). Sem saber nenhuma coluna, a previsão desce pelos dois ramos em todos os nós, pesando pela cobertura:

- Na raiz, 60% vão para a esquerda e 40% para a direita.
- À esquerda, 50% vão para cada folha: 0,5 × (−1,0) + 0,5 × 0,2 = −0,4.
- Base = 0,6 × (−0,4) + 0,4 × 1,5 = −0,24 + 0,60 = **0,36**.

A diferença a explicar é 1,5 − 0,36 = **1,14**.

**Passo 3: a previsão conhecendo só uma coluna.**

- **Só B_3:** o cliente vai para a direita, e lá não tem mais divisão. Previsão: **1,5**.
- **Só P_2:** na raiz, B_3 é desconhecido, então desce pelos dois lados. À esquerda (60%), P_2 = −1 é conhecido e leva à folha −1,0. À direita (40%), a folha é +1,5. Previsão: 0,6 × (−1,0) + 0,4 × 1,5 = **0,0**.

**Passo 4: as duas ordens de entrada.**

| Ordem | Acréscimo de B_3 | Acréscimo de P_2 |
|---|---|---|
| B_3 primeiro, depois P_2 | 1,5 − 0,36 = 1,14 | 1,5 − 1,5 = 0 |
| P_2 primeiro, depois B_3 | 1,5 − 0,0 = 1,50 | 0,0 − 0,36 = −0,36 |
| **Média (valor SHAP)** | **1,32** | **−0,18** |

Conferindo: 1,32 + (−0,18) = 1,14, exatamente a diferença a explicar.

**O que esse exemplo ensina.** O caminho do cliente **nem passou** pelo nó de P_2: ele foi direto para a direita. Mesmo assim, P_2 recebeu −0,18. Isso acontece porque, quando o modelo não sabe o saldo, saber que o pagamento é negativo empurra a previsão para baixo. O Shapley leva em conta todas as situações, não só o caminho que aconteceu. Por isso o SHAP não é a mesma coisa que "quais colunas o cliente usou no caminho".

O TreeSHAP faz exatamente essa conta, só que de forma eficiente, para centenas de árvores e 188 colunas ao mesmo tempo.

---

## 6. Por que isso é melhor do que mascarar uma coluna por vez

O estudo de máscaras do dashboard esconde **uma** coluna e mede quanto p muda. Isso é a primeira linha da tabela do passo 4: o acréscimo de uma coluna quando **todas as outras estão presentes**. No exemplo, P_2 teria efeito zero, porque com B_3 conhecido ela não importa.

O problema aparece com colunas que carregam a mesma informação. Imagine B_3 e B_9, dois saldos quase iguais. Esconder B_3 sozinho não muda nada, porque o modelo ainda tem B_9. Esconder B_9 sozinho também não muda nada. As duas parecem inúteis, mas juntas podem ser decisivas.

O Shapley resolve isso porque faz a média sobre todas as ordens. Nas ordens em que B_3 entra antes de B_9, é B_3 que traz a informação. Nas outras, é B_9. Cada uma fica com a sua parte do efeito que dividem.

A busca gulosa do dashboard ("quantas colunas é preciso esconder para aprovar") também é sensível a isso, porque o resultado depende da ordem em que ela escolhe as colunas. O SHAP não depende de ordem nenhuma.

---

## 7. Interações: quando duas colunas agem juntas

O SHAP de uma coluna mistura dois tipos de efeito: o que ela faz **sozinha** e o que ela faz **em conjunto** com outras. Os **valores de interação SHAP** separam essas duas partes. Para cada par de colunas, eles dizem quanto da previsão vem de as duas estarem juntas, além do que cada uma faria sozinha.

No teste com dados sintéticos, o risco foi construído com um termo do tipo "atraso × saldo" (D_59 × B_3). O script encontrou exatamente esse par como a maior interação, com valor muito acima dos outros pares.

Para o AmEx, isso responde perguntas como "o saldo alto só pesa porque o atraso também é alto?". Use `--interactions 10` para salvar os 10 pares mais fortes de cada cliente do estudo.

---

## 8. Como usar o script

### Rodar

```bash
python explain_shap_xgb.py \
    --model runs/baselines/xgb_amex_raw/model.json \
    --data data/amex/val.parquet \
    --clients 254131 316386 62456 132266 167843 150691 \
    --interactions 10 --interventional \
    --out-json runs/explain/amex_shap_xgb.json \
    --out-js runs/explain/mask_data_shap.js
```

Os clientes acima são os seis do estudo de máscaras do dashboard. Os números ali são **posições na validação** (`--clients-are rows`, o padrão). Se você quiser passar o `customer_ID` do AmEx, use `--clients-are ids`.

Se o arquivo de dados tiver vários extratos por cliente, o script fica só com o último, ordenando pela coluna de data (`--date-col`, padrão `S_2`), que é o que o XGBoost lê. As colunas são postas na mesma ordem em que o modelo foi treinado, e as categóricas são convertidas como o modelo espera. Os caminhos do exemplo são sugestões; ajuste para onde estão o seu modelo e a sua validação.

Precisa de `xgboost`, `pandas` e `numpy`. O modo `--interventional` precisa também de `shap` (`pip install shap`).

### Opções principais

| Opção | O que faz |
|---|---|
| `--sample` | Quantos clientes da validação entram na importância global (padrão 5000) |
| `--interactions K` | Salva os K pares de colunas que mais interagem em cada cliente do estudo |
| `--interventional` | Calcula também o modo *interventional* para os clientes do estudo |
| `--background` | Tamanho do fundo do modo *interventional* (padrão 200) |
| `--waterfall-top` | Quantas colunas mostrar no passo a passo de cada cliente (padrão 15) |

### O que o script confere

Ao rodar, ele imprime o **erro máximo de aditividade**: a maior diferença entre "base + soma dos SHAP" e a saída real do modelo. Deve ser praticamente zero (algo como 10⁻⁶). Se não for, há algum problema na leitura dos dados ou do modelo.

### A saída

O mesmo conteúdo é salvo em JSON (para notebook) e em `.js` (`window.SHAPX = {...}`, no mesmo formato dos `mask_data_*.js` do dashboard).

**`meta`**: informações gerais.
- `base` e `p_base`: a previsão média do modelo, em log-odds e em probabilidade. É o ponto de partida de toda explicação.
- `units`: sempre `log-odds`.
- `additivity_max_err`: o erro de aditividade descrito acima.
- `interventional`: o método usado no modo *interventional*, se foi pedido.
- `cols`: as colunas, com nome e grupo, no mesmo formato do dashboard.

**`global`**: o que importa para o modelo **em geral**, na amostra da validação.
- `cols`: cada coluna com `mean_abs` (média de |SHAP|, a importância), `mean` (média com sinal: se positiva, a coluna costuma aumentar o risco) e `pos_share` (fração dos clientes em que ela aumentou o risco). Ordenadas da mais para a menos importante.
- `groups`: a mesma coisa por grupo (atraso, gasto, pagamento, saldo, risco). Graças à aditividade, o SHAP de um grupo é só a soma do SHAP das suas colunas.
- `dependence`: para as 6 colunas mais importantes, até 500 pares (valor da coluna, SHAP). Com isso se desenha um **gráfico de dependência**, que mostra como o efeito muda conforme o valor: por exemplo, se o risco sobe de forma contínua com o saldo ou só a partir de um certo patamar.

**`clients`**: a explicação de cada cliente do estudo.
- `p`, `margin` e `base`: a previsão do cliente (em probabilidade e log-odds) e o ponto de partida.
- `shap`: o valor SHAP de cada uma das 188 colunas, na ordem de `meta.cols`.
- `groups`: o SHAP somado por grupo.
- `waterfall`: o **passo a passo** da previsão. Parte da base e vai somando as colunas da maior para a menor |SHAP|, mostrando o valor da coluna para o cliente, o SHAP e a probabilidade depois de somar (`p_after`). As colunas restantes ficam juntas em "outras". É o formato do gráfico em cascata, o mais usado para explicar uma decisão individual.
- `interactions` (com `--interactions`): os pares que mais interagem, e `main`, o efeito de cada coluna sozinha.
- `shap_interventional` e `base_interventional` (com `--interventional`): os mesmos valores no outro modo.

### Exemplo de leitura

Suponha que o passo a passo de um cliente comece assim:

| Coluna | Valor | SHAP | p depois |
|---|---|---|---|
| (base) | | | 0,26 |
| B_3 | 0,41 | +0,90 | 0,46 |
| D_59 | 2,0 | +0,55 | 0,60 |
| P_2 | 0,32 | +0,40 | 0,69 |
| S_3 | 0,18 | −0,20 | 0,65 |
| … | | | |

A leitura é: um cliente médio teria 26% de risco. O saldo B_3 deste cliente leva a 46%, o atraso D_59 sobe para 60%, o pagamento P_2 para 69%, e o gasto S_3 puxa um pouco para baixo. O risco final vem principalmente do saldo e do atraso.

---

## 9. Cuidados ao interpretar

**Explica o modelo, não o mundo.** Um SHAP alto em D_59 quer dizer que o modelo se apoia muito em D_59 para este cliente. Não quer dizer que, se o cliente reduzisse o atraso, o default dele mudaria. As colunas são correlacionadas, e o modelo pode estar usando D_59 como sinal de algo que ele não vê diretamente. É a mesma ressalva do estudo de máscaras: para causalidade é preciso um modelo causal.

**Colunas correlacionadas dividem o crédito.** Se B_3 e B_9 são quase iguais, o SHAP pode dar metade para cada uma, e as duas parecerem moderadas quando, juntas, são "o saldo" e decidem tudo. Por isso vale olhar também o SHAP por grupo.

**A soma só é exata em log-odds.** Convertendo cada contribuição separadamente para probabilidade, as partes não somam mais o total. O passo a passo (`p_after`) contorna isso mostrando a probabilidade acumulada, e não a contribuição isolada.

**A importância global e a individual respondem perguntas diferentes.** Uma coluna pode ser pouco importante na média e decisiva para um cliente específico, ou o contrário. Para "por que este crédito foi negado", use a explicação do cliente. Para "em que o modelo se apoia em geral", use a global.

**O XGBoost do AmEx lê só o último extrato.** Então o SHAP dele nunca vai dizer nada sobre a trajetória ao longo dos meses. Essa é uma diferença real em relação ao PRAGMA, que lê os 13 extratos, e não uma limitação do SHAP.

---

## 10. Como isso conversa com o estudo de máscaras

| | PRAGMA com [MASK] | XGBoost com NULL (hoje) | XGBoost com SHAP |
|---|---|---|---|
| Pergunta respondida | Previsão sem saber o valor | Previsão com o dado faltando | Previsão sem saber o valor |
| Como "não saber" é feito | Token aprendido no pré-treino | Direção padrão dos nulos | Média pelos ramos ou pelo fundo |
| Considera as outras colunas | Uma configuração de máscara por vez | Uma configuração por vez | Média sobre todas as ordens |
| Soma das partes = total | Não | Não | Sim |

Para uma comparação justa entre os dois modelos, o mais natural é colocar lado a lado o efeito de esconder uma coluna no PRAGMA e o SHAP da mesma coluna no XGBoost, deixando claro que cada um usa o mecanismo próprio do seu modelo para responder à mesma pergunta.

---

## Referências

- Shapley, L. S. *A Value for n-Person Games*. Contributions to the Theory of Games II, 1953. A origem da ideia de dividir o prêmio pela média das ordens.
- Lundberg, S. M.; Lee, S.-I. *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017. Introduz o SHAP.
- Lundberg, S. M.; Erion, G. G.; Lee, S.-I. *Consistent Individualized Feature Attribution for Tree Ensembles*. arXiv, 2018. O algoritmo TreeSHAP e os valores de interação.
- Lundberg, S. M. et al. *From Local Explanations to Global Understanding with Explainable AI for Trees*. Nature Machine Intelligence, 2020. TreeSHAP exato, modo *interventional* e explicações globais a partir das locais.
- Janzing, D.; Minorics, L.; Blöbaum, P. *Feature Relevance Quantification in Explainable AI: A Causal Problem*. AISTATS 2020. Por que o modo *interventional* responde melhor "quanto o modelo usa a coluna".
- Chen, H.; Janizek, J. D.; Lundberg, S.; Lee, S.-I. *True to the Model or True to the Data?* arXiv, 2020. A diferença entre os dois modos, com exemplos.
