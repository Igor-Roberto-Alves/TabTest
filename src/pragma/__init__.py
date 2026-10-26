"""Modelo fundacional de eventos bancários, no estilo PRAGMA (Revolut).

Cada cliente é um perfil estático mais um histórico de eventos vindos de várias fontes
(transações, extratos, parcelas...). Células de eventos e do perfil são representadas como
no LDM (src/ldm): coluna descrita pelo texto do nome, valor por texto da categoria ou bin
de quantil. Assim um único modelo é pré-treinado em bases com esquemas diferentes.

    prepare.py    bases brutas -> formato comum em data/banking/prepared/<base>/
    store.py      leitura e escrita desse formato
    data.py       amostras (cliente, corte no tempo) -> lotes com sequências de eventos
    model.py      encoders de evento, perfil e histórico; decoder de células; LoRA
    objective.py  máscaras de célula, evento e tipo; loss de pré-treino
    evaluate.py   embeddings, linear probe e métricas (AUC, KS, Brier, métrica AmEx)
    train.py      pré-treino com TensorBoard
    finetune.py   tarefas supervisionadas (probe, LoRA, fine-tuning completo, do zero)
    sweep.py      busca de hiperparâmetros (grade ou aleatória)
    config.py     configs YAML com sobrescritas pela linha de comando

Visão geral e plano de treino: docs/pragma_sketch.md.
"""
