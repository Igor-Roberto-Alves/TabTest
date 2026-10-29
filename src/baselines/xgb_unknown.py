"""Previsão do XGBoost quando um conjunto de colunas é desconhecido (a mesma definição do TreeSHAP path-dependent).

Num nó que divide por uma coluna desconhecida, a previsão segue pelos dois ramos e pesa cada um pela cobertura (cover)
do treino. É a média das previsões sobre os valores que a coluna poderia ter, e não um NULL: o XGBoost aprendeu para
onde mandar NULL, então NULL responde "e se o dado faltasse?", outra pergunta (ver docs/SHAP_explicado.md).
Com todas as colunas desconhecidas dá a base do SHAP; com nenhuma, a margem do modelo.
"""

import json

import numpy as np


class Trees:
    def __init__(self, booster):
        self.names = list(booster.feature_names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.trees = [self._compile(json.loads(t)) for t in booster.get_dump(dump_format="json", with_stats=True)]
        x0 = np.zeros((1, len(self.names)), dtype=np.float32)
        import xgboost as xgb
        self.offset = float(booster.predict(xgb.DMatrix(x0, feature_names=self.names), output_margin=True)[0]) \
            - sum(self.value(t, x0[0], np.zeros(len(self.names), bool)) for t in self.trees)

    def _compile(self, node):
        if "leaf" in node:
            return float(node["leaf"])
        kids = {c["nodeid"]: c for c in node["children"]}
        yes, no, miss = kids[node["yes"]], kids[node["no"]], kids[node["missing"]]
        return (self.index[node["split"]], float(node["split_condition"]), self._compile(yes), self._compile(no),
                self._compile(miss), float(yes["cover"]), float(no["cover"]))

    def value(self, t, x, unknown):
        if not isinstance(t, tuple):
            return t
        f, thr, yes, no, miss, cy, cn = t
        if unknown[f]:
            return (cy * self.value(yes, x, unknown) + cn * self.value(no, x, unknown)) / (cy + cn)
        v = x[f]
        if v != v:
            return self.value(miss, x, unknown)
        return self.value(yes if v < thr else no, x, unknown)

    def margin(self, x, unknown):
        """Margem (log-odds) com as colunas marcadas em `unknown` (bool, tamanho = nº de colunas) desconhecidas."""
        return self.offset + sum(self.value(t, x, unknown) for t in self.trees)

    def proba(self, x, unknown):
        return 1.0 / (1.0 + np.exp(-self.margin(x, unknown)))
