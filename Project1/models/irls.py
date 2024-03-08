from typing import List, Tuple
import numpy as np


class IRLS:
    def __init__(self, p: int = 2, iter_limit: int = 500, delta: float = 1E-4):
        self._p = p
        self._iter_limit = iter_limit
        self._delta = delta

        self._weights = None
        self._beta = None
        self._n_iter = 0

    def _update_beta(self, X, y):
        m1 = np.linalg.multi_dot([X.T, self._weights, X])
        m2 = np.linalg.multi_dot([X.T, self._weights, y])
        inv_m1 = np.linalg.inv(m1)
        self._beta = np.dot(inv_m1, m2)

    def _update_weights(self, X, y):
        residuals = np.abs(y - self.predict_proba(X))
        if self._p == 1:
            residuals[residuals < self._delta] = self._delta
        weights_diag = np.power(residuals, self._p - 2)
        self._weights = np.diag(weights_diag)

    def _iteration(self, X, y):
        if self._n_iter >= self._iter_limit:
            raise StopIteration()

        # TODO: Also some stop-condition, we have to decide
        self._update_beta(X, y)
        self._update_weights(X, y)
        self._n_iter += 1

    def fit(self, X, y, interactions: List[Tuple[int, int]] = None):
        classes = list(np.unique(y))
        if len(classes) != 2 or -1 not in classes or 1 not in classes:
            raise ValueError("y.classes != [-1, 1]")

        if self._weights is not None or self._beta is not None or self._n_iter != 0:
            raise ValueError("Model already fitted or corrupted")

        if interactions is not None and len(interactions) > 0:
            inter_cols = []
            for v1, v2 in interactions:
                inter_col = X[:, v1] * X[:, v2]
                inter_cols.append(inter_col)
            Xint = np.concatenate(inter_cols, axis=0)
            X = np.concatenate([X, Xint], axis=0)

        if len(y.shape) == 1:
            y = y.reshape((y.shape[0], 1))

        ncol = X.shape[1]
        self._weights = np.diag(np.ones(ncol))

        while True:
            try:
                self._iteration(X, y)
            except StopIteration:
                break

    def predict_proba(self, X):
        if self._weights is None or self._beta is None:
            raise ValueError("Start with fitting the model")

        return 1/(1 + np.exp(-np.dot(X, self._beta)))

    def predict(self, X):
        probs = self.predict_proba(X)
        probs[probs < 0] = -1
        probs[probs >= 0] = 1
        return probs

    def get_params(self):
        params = {
            "p-regularization": self._p,
            "iteration_limit": self._iter_limit,
            "weights": self._weights,
            "coefficients": self._beta,
            "iterations_run": self._n_iter
        }

        if self._p == 1:
            params["delta"] = self._delta
        return params
