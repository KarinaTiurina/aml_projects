import numpy as np
from typing import List, Tuple


def prepare_x(X, interactions: List[Tuple[int, int]] = None):
    if interactions is not None and interactions:
        X = _apply_interactions(X, interactions)
    ones = np.ones(X.shape[0]).reshape((X.shape[0], 1))
    return np.concatenate([ones, X], 1)


def _apply_interactions(X, interactions: List[Tuple[int, int]]):
    inter_cols = []
    for v1, v2 in interactions:
        inter_col = X[:, v1] * X[:, v2]
        inter_cols.append(inter_col.reshape((inter_col.shape[0], 1)))
    Xint = np.concatenate(inter_cols, axis=1)
    X = np.concatenate([X, Xint], axis=1)
    return X
