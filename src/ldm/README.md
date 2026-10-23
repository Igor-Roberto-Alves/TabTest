# Large Data Model tabular

Um único Transformer pré-treinado em muitas tabelas, que completa células mascaradas.
O objetivo é de difusão mascarada (MDLM). O mesmo modelo serve para:

- **predição**: mascara a coluna alvo e lê `p(alvo | resto)`;
- **imputação**: mascara as células faltantes;
- **geração sintética**: começa com tudo mascarado.

O que o torna um *large data model*, e não um modelo por tabela, é que **nenhum parâmetro
depende de uma tabela específica**. Colunas e valores são descritos por texto e quantis,
então o modelo pré-treinado se aplica a tabelas que nunca viu.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `text.py` | texto → ids de hash (palavras + trigramas de caracteres, 2^16 buckets), sem vocabulário fixo |
| `table.py` | `TableEncoder`: tipo de cada coluna, bins de quantil, categorias, `encode`/`decode`, `meta()` |
| `model.py` | `TabularLDM`: embeddings independentes da tabela, Transformer bidirecional e heads |
| `diffusion.py` | `MaskedDiffusion`: loss ELBO, `impute`, `generate`, `predict` |
| `corpus.py` | `TableCorpus`: carrega várias tabelas e sorteia lotes (tabela, linhas, colunas) |
| `corpus_download.py` | baixa cerca de 34 tabelas do OpenML para `data/corpus/` |
| `pretrain.py` | CLI de pré-treino e checkpoints (`load_checkpoint`) |
| `predict.py` | adaptação a uma tabela nova (fine-tuning opcional) e predição do alvo |

A integração com o benchmark está em `src/models/ldm_model.py` (modelo `ldm` do `src/run.py`).

## Arquitetura

**Representação de uma célula.** Cada célula tem um índice local:
`0 = [NULL]`, `1..K_c` = opção da coluna (categoria ou bin de quantil).
As opções não são ids aprendidos; cada uma é *descrita*:

- **categórica**: `TextEmb("texto da categoria")`;
- **numérica**: `MLP(fourier(q), fourier(z))`, com `q` = posição do bin no CDF do treino e
  `z` = mediana padronizada do bin. Assim "bin 3 de 32" significa a mesma coisa em qualquer tabela.

**Entrada** da célula `(n, c)`:

```
h = W_in · o_{c, l}  (+ W_z · fourier(z_nc) se numérica visível)   ou   e_[MASK]
    + TextEmb(nome da coluna c) + E_tipo
    + MLP(senoidal(r))        r = fração de células mascaradas da linha
```

Não há embedding posicional: o conjunto de colunas não tem ordem.

**Encoder:** blocos pre-LN com atenção bidirecional entre as colunas da linha.

**Head de autocomplete:** `logit_{n,c,k} = <W_q h_nc, W_out o_ck> / sqrt(d)`. É uma
pontuação contra os embeddings das opções válidas daquela coluna; como as opções vêm de
texto e quantis, a mesma head serve para qualquer coluna de qualquer tabela.
**Head de regressão:** `z_nc = w·h_nc + b`, o valor contínuo padronizado das numéricas.

**Difusão:** cada célula é mascarada com probabilidade `1 - α(t)`, com schedule cosseno ou linear. A loss é
a cross-entropy nas mascaradas, ponderada por `w(t) = -α'(t) / (1 - α(t))`, mais um MSE em `z`.
No reverse, cada célula mascarada é revelada com probabilidade `(α(s) - α(t)) / (1 - α(t))`.

**Pré-treino:** a cada passo sorteia uma tabela (probabilidade ∝ `n_linhas^0.5`), um lote de
linhas e até 64 colunas.

**Adaptação:** o `TableEncoder` é ajustado no treino da tabela nova. Com `steps > 0`, uma
cópia do modelo faz fine-tuning, alternando a loss de difusão com lotes em que o alvo é sempre
mascarado. `steps = 0` é zero-shot.

## Uso

```bash
pip install torch pandas numpy scikit-learn
python download_datasets.py                     # tabelas de avaliação -> data/
python -m src.ldm.corpus_download               # tabelas de pré-treino -> data/corpus/
python -m src.ldm.pretrain --dirs data/corpus --steps 20000 --save checkpoints/ldm.pt

# zero-shot, fine-tuning e treino do zero (ablação) no benchmark
python -m src.run --task adult_income --models ldm --ldm-checkpoint checkpoints/ldm.pt --ldm-steps 0
python -m src.run --task adult_income --models ldm xgboost --ldm-checkpoint checkpoints/ldm.pt
python -m src.run --task adult_income --models ldm
```

Uso direto:

```python
from src.ldm.corpus import EncodedTable, to_torch
from src.ldm.pretrain import load_checkpoint

diffusion, _ = load_checkpoint("checkpoints/ldm.pt")
table = EncodedTable.from_frame("minha_tabela", df)
meta = to_torch(table.meta, "cpu")
local, z = diffusion.generate(100, meta)        # 100 linhas sintéticas
synthetic = table.encoder.decode(local.numpy(), z.numpy())
```

## Vazamento

O pré-treino vê **todas** as linhas das tabelas do corpus. Para avaliar numa tabela, ela
precisa ficar fora (`--dirs data/corpus` ou `--exclude nome`). O `ldm_model` avisa se o dataset
da tarefa está no checkpoint. Não coloque no corpus cópias dos datasets de avaliação; o
`houses` do OpenML, por exemplo, é o California Housing.

## Limitações e próximos passos

- O embedding de texto é um saco de n-gramas com hash. Trocar por um encoder de frases
  pré-treinado dá semântica real aos nomes ("salary" ≈ "income").
- Ainda é um modelo de *linha*: cada linha é processada isoladamente. Atender também a outras
  linhas da tabela, como contexto no estilo TabPFN, é o próximo passo natural.
- O corpus tem cerca de 34 tabelas. Um LDM de verdade precisa de milhares (OpenML, Kaggle,
  tabelas sintéticas geradas por SCMs).
- A tabela de hash (2^16 × d) é atualizada densamente pelo AdamW, o que é lento em CPU.
