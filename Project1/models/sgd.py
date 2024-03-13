from typing import List, Tuple
import numpy as np
import math

from models.util import ClassMapper


def _sigmoid(z):
    if isinstance(z, float):
        if z < -100:
            return 0

        return 1/(1+math.exp(-z))
    else:
        exp_z = z.copy()
        exp_z[z < -100] = 0
        exp_z[z >= -100] = 1/(1+np.exp(-z[z >= -100]))
        return exp_z


class SGD:
    def __init__(self,
                 iter_limit: int = 500,
                 rate: float = 1E-4,
                 rng: np.random.Generator = None):
        self._iter_limit = iter_limit
        self._learning_rate = rate

        if rng is None:
            self._rng = np.random.default_rng(0)
        else:
            self._rng = rng

        self._beta = None
        self._n_iter = 0
        self._mapper = ClassMapper([-1, 1])

    def _gradient(self, x_sample, y_sample):
        pred = _sigmoid(self.log_odds(x_sample, prepare=False))
        return x_sample * (pred - (y_sample + 1) / 2)

    def _update_beta(self, x_sample, y_sample):
        grad = self._gradient(x_sample, y_sample)
        self._beta -= self._learning_rate * grad

    def _iteration(self, X, y):
        if self._n_iter >= self._iter_limit:
            raise StopIteration()

        # TODO: Also some stop-condition, we have to decide
        #       Based on my struggles, it would be good to see if the gradient is
        #       close to 0 for both classes for some number of samples
        #       (e.g. 10 samples from class 0, 10 from 1, nothing changed, ergo nothing should change in the future)

        combined_data = np.concatenate([X, y.reshape((len(y), 1))], axis=1)
        self._rng.shuffle(combined_data)

        np.apply_along_axis(lambda r: self._update_beta(r[:-1], r[-1]), 1, combined_data)
        self._n_iter += 1

    def _prepare_x(self, X):
        ones = np.ones(X.shape[0]).reshape((X.shape[0], 1))
        return np.concatenate([ones, X], 1)

    def fit(self, X, y, interactions: List[Tuple[int, int]] = None):
        yy = self._mapper.map_to_target(y)
        classes = list(np.unique(yy))
        if len(classes) != 2:
            raise ValueError("y is not a binary vector")

        if self._n_iter != 0:
            raise ValueError("Model already fitted or corrupted")

        if interactions is not None and len(interactions) > 0:
            inter_cols = []
            for v1, v2 in interactions:
                inter_col = X[:, v1] * X[:, v2]
                inter_cols.append(inter_col)
            Xint = np.concatenate(inter_cols, axis=1)
            X = np.concatenate([X, Xint], axis=1)

        if len(yy.shape) != 1:
            yy = yy.flatten()

        X = self._prepare_x(X)

        ncol = X.shape[1]
        self._beta = np.zeros(ncol)

        X_copy = np.copy(X)
        y_copy = np.copy(yy)

        while True:
            try:
                self._iteration(X_copy, y_copy)
            except StopIteration:
                break

    def log_odds(self, X, prepare=True):
        if self._beta is None:
            raise ValueError("Start with fitting the model")
        if prepare:
            X = self._prepare_x(X)
        return np.dot(X, self._beta)

    def predict_proba(self, X, prepare=True):
        log_odds = self.log_odds(X, prepare)
        return _sigmoid(log_odds)

    def predict(self, X):
        probs = self.predict_proba(X)
        probs[probs < 0.5] = -1
        probs[probs >= 0.5] = 1
        return self._mapper.map_from_target(probs)

    def get_params(self):
        params = {
            "learning_rate": self._learning_rate,
            "iteration_limit": self._iter_limit,
            "coefficients": self._beta,
            "iterations_run": self._n_iter
        }

        return params
