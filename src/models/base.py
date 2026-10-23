from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    y_pred: np.ndarray  # classe inteira ou valor contínuo
    proba: np.ndarray | None = None  # (n, n_classes) para classificação
    model: object = None  # estimador treinado, para inspeção
