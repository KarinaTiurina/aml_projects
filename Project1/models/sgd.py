from typing import List, Tuple
import numpy as np
import math

from models.util import ClassMapper
from sklearn.model_selection import train_test_split


def _sigmoid(z):
    cutoff = -110
    if isinstance(z, float):
        if z < cutoff:
            return 0

        return 1 / (1 + math.exp(-z))
    else:
        exp_z = z.copy()
        exp_z[z < cutoff] = 0
        exp_z[z >= cutoff] = 1 / (1 + np.exp(-z[z >= cutoff]))
        return exp_z


class SGD:
    def __init__(self,
                 iter_limit: int = 500,
                 rate: float = 1E-4,
                 rng: np.random.Generator = None,
                 early_stopping=False,
                 n_iter_no_change=5,
                 mov_avg_window=10,
                 tol=1e-3,
                 validation_fraction=0.2,
                 diagnostics=False):
        self._iter_limit = iter_limit
        self._learning_rate = rate

        if rng is None:
            self._rng = np.random.default_rng(0)
        else:
            self._rng = rng

        self._beta = None
        self._n_iter = 0
        self._mapper = ClassMapper([-1, 1])

        self._diagnostics = diagnostics
        self._training_loss = []

        self._early_stopping = early_stopping
        self._n_iter_no_change = n_iter_no_change
        self._tol = tol
        self._validation_fraction = validation_fraction

        self._validation_loss = []
        self._validation_avg = []
        self._n_val_loss_worsens = 0

        self._mov_avg_window = mov_avg_window
        self._mov_avg = 0
        self._best_beta = None

    def _gradient(self, x_sample, y_sample):
        pred = _sigmoid(np.dot(x_sample, self._beta))
        y_scaled = (y_sample + 1) / 2
        return (pred - y_scaled) * x_sample

    def _update_beta(self, x_sample, y_sample):
        grad = self._gradient(x_sample, y_sample)
        self._beta -= self._learning_rate * grad

    def _check_early_stopping(self, current_loss):
        if self._diagnostics:
            self._validation_loss.append(current_loss)

        if self._mov_avg_window >= self._n_iter:
            new_mov_avg = (self._mov_avg * (self._n_iter - 1) + current_loss) / self._n_iter
            self._mov_avg = new_mov_avg
            if self._diagnostics:
                self._validation_avg.append(self._mov_avg)
            return

        new_mov_avg = (self._mov_avg * (self._mov_avg_window - 1) + current_loss) / self._mov_avg_window
        delta = new_mov_avg - self._mov_avg
        self._mov_avg = new_mov_avg
        if self._diagnostics:
            self._validation_avg.append(self._mov_avg)

        if delta <= -self._tol:
            self._n_val_loss_worsens = 1
            self._best_beta = np.copy(self._beta)
        else:
            self._n_val_loss_worsens += 1

        if self._n_val_loss_worsens >= self._n_iter_no_change:
            print(f"Stopping after {self._n_iter} iterations")
            self._beta = self._best_beta
            raise StopIteration()

    def _iteration(self, X, y, X_val=None, y_val=None):
        if self._n_iter >= self._iter_limit:
            raise StopIteration()

        combined_data = np.concatenate([X, y.reshape((len(y), 1))], axis=1)
        self._rng.shuffle(combined_data)

        np.apply_along_axis(lambda r: self._update_beta(r[:-1], r[-1]), 1, combined_data)

        if self._diagnostics:
            loss = self.logloss(X, y, prepare=False)
            self._training_loss.append(loss)

        self._n_iter += 1

        if self._early_stopping:
            validation_loss = self.logloss(X_val, y_val, prepare=False)
            val_score = validation_loss / len(y_val)
            self._check_early_stopping(val_score)

    def _prepare_x(self, X):
        ones = np.ones(X.shape[0]).reshape((X.shape[0], 1))
        return np.concatenate([ones, X], 1)

    def logloss(self, X, y, prepare=True):
        a = self.predict_proba(X, prepare=prepare)
        a[y==-1] = 1-a[y==-1]
        a[a == 0] = 1e-300
        loss = -np.log(a).sum() / len(y)
        return loss

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
        self._beta = np.zeros(ncol)

        if self._early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(X, yy,
                                                              test_size=self._validation_fraction,
                                                              random_state=np.random.RandomState(self._rng.bit_generator))
        else:
            X_train = np.copy(X)
            y_train = np.copy(yy)
            X_val, y_val = None, None

        while True:
            try:
                self._iteration(X_train, y_train, X_val=X_val, y_val=y_val)
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
            "learning_rate": self._learning_rate,
            "iteration_limit": self._iter_limit,
            "coefficients": self._beta,
            "iterations_run": self._n_iter
        }

        return params

    def get_diagnostics(self):
        if not self._diagnostics:
            raise RuntimeError("Use `diagnostics=True` parameter in constructor")
        return {
            'training_loss': np.asarray(self._training_loss),
            'validation_loss': np.asarray(self._validation_loss),
            'mov_avg': np.asarray(self._validation_avg),
        }
