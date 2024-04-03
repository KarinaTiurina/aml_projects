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


class ADAM:
    # Default values taken from https://arxiv.org/pdf/1412.6980.pdf
    def __init__(self,
                 iter_limit: int = 500,
                 stepsize: float = 1E-3,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1E-8,
                 tol: float = 1E-6,
                 rng: np.random.Generator = None):
        self._iter_limit = iter_limit
        self._stepsize = stepsize

        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._tol = tol

        self._t = 0
        self._beta1t = 1
        self._beta2t = 1

        if rng is None:
            self._rng = np.random.default_rng(0)
        else:
            self._rng = rng

        self._theta = None
        self._moment_m = 0
        self._moment_v = 0
        self._n_iter = 0
        self._prev_loss = float('inf')
        self._loss_history = []
        self._mapper = ClassMapper([-1, 1])

    def _gradient(self, x_sample, y_sample):
        pred = _sigmoid(np.dot(x_sample, self._theta))
        y_scaled = (y_sample + 1) / 2
        return (pred - y_scaled) * x_sample

    def _update(self, x_sample, y_sample):
        self._t += 1
        self._beta1t *= self._beta1
        self._beta2t *= self._beta2

        grad = self._gradient(x_sample, y_sample)
        self._moment_m = self._beta1 * self._moment_m + (1 - self._beta1) * grad
        self._moment_v = self._beta2 * self._moment_v + (1 - self._beta2) * np.square(grad)

        m_unbias = self._moment_m / (1 - self._beta1t)
        v_unbias = self._moment_v / (1 - self._beta2t)

        self._theta -= self._stepsize * m_unbias / (np.sqrt(v_unbias) + self._epsilon)

    def _nll_loss(self, y_true, y_pred):
        # Stabilize the log computation
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # Compute the Negative Log Likelihood loss and normalize by the number of samples
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / len(y_true)
        return loss

    def _iteration(self, X, y):
        if self._n_iter >= self._iter_limit:
            raise StopIteration()

        # TODO: Also some stop-condition, we have to decide
        #       Based on my struggles, it would be good to see if the gradient is
        #       close to 0 for both classes for some number of samples
        #       (e.g. 10 samples from class 0, 10 from 1, nothing changed, ergo nothing should change in the future)

        combined_data = np.concatenate([X, y.reshape((len(y), 1))], axis=1)
        self._rng.shuffle(combined_data)

        theta = np.copy(self._theta)
        np.apply_along_axis(lambda r: self._update(r[:-1], r[-1]), 1, combined_data)

        loss = self._nll_loss(y, self.predict_proba(X, prepare=False))

        if abs(self._prev_loss - loss) < self._tol:
            raise StopIteration()

        self._prev_loss = loss
        self._loss_history.append(loss)

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
                inter_col = (X[:, v1] * X[:, v2]).reshape((X.shape[0], 1))
                inter_cols.append(inter_col)
            Xint = np.concatenate(inter_cols, axis=1)
            X = np.concatenate([X, Xint], axis=1)

        if len(yy.shape) != 1:
            yy = yy.flatten()

        X = self._prepare_x(X)

        ncol = X.shape[1]
        self._theta = np.zeros(ncol)

        X_copy = np.copy(X)
        y_copy = np.copy(yy)

        while True:
            try:
                self._iteration(X_copy, y_copy)
            except StopIteration:
                break

    def log_odds(self, X, prepare=True):
        if self._theta is None:
            raise ValueError("Start with fitting the model")
        if prepare:
            X = self._prepare_x(X)
        return np.dot(X, self._theta)

    def predict_proba(self, X, prepare=True):
        log_odds = self.log_odds(X, prepare)
        return _sigmoid(log_odds)

    def predict(self, X, interactions: List[Tuple[int, int]] = None):
        if interactions is not None and len(interactions) > 0:
            inter_cols = []
            for v1, v2 in interactions:
                inter_col = (X[:, v1] * X[:, v2]).reshape((X.shape[0], 1))
                inter_cols.append(inter_col)
            Xint = np.concatenate(inter_cols, axis=1)
            X = np.concatenate([X, Xint], axis=1)

        probs = self.predict_proba(X)
        probs[probs < 0.5] = -1
        probs[probs >= 0.5] = 1
        return self._mapper.map_from_target(probs)

    def get_params(self):
        params = {
            "stepsize": self._stepsize,
            "beta1": self._beta1,
            "beta2": self._beta2,
            "epsilon": self._epsilon,
            "iteration_limit": self._iter_limit,
            "coefficients": self._theta,
            "iterations_run": self._n_iter
        }

        return params
