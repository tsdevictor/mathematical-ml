import numpy as np


class LogisticRegressionGD:
    """Binary logistic regression trained with gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        num_iters: int = 1000,
        l2: float = 0.0,
    ):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.l2 = l2
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.num_iters):
            logits = X @ self.weights_ + self.bias_
            probs = self._sigmoid(logits)

            eps = 1e-12
            loss = -np.mean(
                y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)
            )
            loss += 0.5 * self.l2 * np.sum(self.weights_**2)
            self.loss_history_.append(float(loss))

            error = probs - y
            grad_w = (X.T @ error) / n_samples + self.l2 * self.weights_
            grad_b = float(np.mean(error))

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("model must be fit before predict_proba")

        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


class LinearSVM:
    """Linear SVM trained with subgradient descent on hinge loss."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        num_iters: int = 1000,
        C: float = 1.0,
    ):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.C = C
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        y_signed = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        for _ in range(self.num_iters):
            margins = y_signed * (X @ self.weights_ + self.bias_)
            misclassified = margins < 1

            grad_w = self.weights_ - self.C * (X[misclassified].T @ y_signed[misclassified])
            grad_b = -self.C * np.sum(y_signed[misclassified])

            self.weights_ -= self.learning_rate * grad_w / n_samples
            self.bias_ -= self.learning_rate * grad_b / n_samples

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("model must be fit before decision_function")

        X = np.asarray(X, dtype=float)
        return X @ self.weights_ + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)
