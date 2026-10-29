# Como adicionar um modelo

1. `python -m src.models.new meu_modelo` cria `src/models/meu_modelo.py` a partir do molde `_template.py`.
2. Preencha `KINDS` (binary, multiclass, regression), `DATA` (tabular, series, events) e `fit_predict(split, seed, **kw) -> Prediction`.
3. Teste: `python -m src.run --task adult_income --models meu_modelo` (`--model-arg meu_modelo.chave=valor` passa argumentos).
4. Nada mais a editar: `python -m src.run --list-models` e o painel leem o registro (`src/models/registry.py`).

## Execuções longas (treino com passos, fila, painel)

Use `RunStatus(run_dir, name, group, model="meu_modelo", ...)` de `src/pragma/runlog.py` e escreva em `runs/<grupo>/<nome>/`.
O campo `model` vira a coluna `model` das linhas do painel; sem ele, o exportador deduz (PRAGMA, TabSTAR ou XGBoost).

## Capacidades opcionais

`CAPABILITIES` declara o que o modelo faz além de `fit_predict`: `pretrain`, `finetune`, `explain_mask`.
A aba "Estudo de máscaras" só deve listar modelos com `explain_mask`. Para séries e eventos, a máscara é uma célula, um evento ou uma coluna; para tabelas, uma coluna vira nula.
