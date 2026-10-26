"""Baselines GBDT (XGBoost em agregações feitas à mão) para as tarefas P1.

São a referência que o modelo fundacional (src/pragma) precisa bater. Usam exatamente o split
por cliente, as linhas de rótulo e as métricas do modelo fundacional, e registram cada execução
em runs/baselines/<run>/ no formato de src/pragma/runlog.py, para o painel ao vivo.

    common.py       rótulos e split do store, subamostra de rótulos, treino com registro
    home_credit.py  application_* + agregações das 6 tabelas de histórico
    amex.py         agregações por cliente dos extratos (last, mean, std, min, max, last - mean)
    mbd.py          agregados mensais de trx, geo e diálogos em janelas antes do mês do rótulo
    xgb.py          linha de comando: python -m src.baselines.xgb home_credit|amex|mbd_mini

As features ficam em cache em data/banking/features/<base>.parquet.
"""
