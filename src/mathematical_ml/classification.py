import numpy as np
from qpsolvers import solve_qp


class HardMarginSVM:
    """Hard-margin linear SVM solved through the dual quadratic program.

    Adapted from the original coursework implementation. The model solves

        min 1/2 alpha^T Q alpha - 1^T alpha
        s.t. alpha >= 0, y^T alpha = 0

    where Q_ij = y_i y_j <x_i, x_j>.
    """

    def __init__(self, support_tol: float = 1e-7, solver: str = "quadprog"):
        self.support_tol = support_tol
        self.solver = solver
        self.alpha_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: float | None = None
        self.support_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HardMarginSVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("HardMarginSVM expects labels in {-1, 1}")

        n_samples = X.shape[0]

        Q = (y[:, None] * y[None, :]) * (X @ X.T)

        P = Q + 1e-8 * np.eye(n_samples)
        q = -np.ones(n_samples)

        G = -np.eye(n_samples)
        h = np.zeros(n_samples)

        A = y.reshape(1, -1)
        b = np.array([0.0])

        alpha = solve_qp(P, q, G, h, A, b, solver=self.solver)

        if alpha is None:
            raise RuntimeError("quadratic program did not return a solution")

        support_indices = np.where(alpha > self.support_tol)[0]
        weights = np.sum((alpha * y)[:, None] * X, axis=0)

        bias_candidates = []
        for i in support_indices:
            bias_candidates.append(y[i] - weights @ X[i])

        bias = float(np.mean(bias_candidates))

        self.alpha_ = alpha
        self.weights_ = weights
        self.bias_ = bias
        self.support_indices_ = support_indices

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("SVM must be fit before prediction")

        X = np.asarray(X, dtype=float)
        return X @ self.weights_ + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X))
