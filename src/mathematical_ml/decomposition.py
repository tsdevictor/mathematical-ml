import numpy as np


class PCA:
    """PCA via the eigendecomposition of X.T @ X.

    Adapted from the original coursework implementation, which explicitly
    constructs principal directions, singular values, and low-dimensional scores.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)

        if self.n_components > min(X.shape):
            raise ValueError("n_components cannot exceed min(n_samples, n_features)")

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        eigenvalues, eigenvectors = np.linalg.eigh(X_centered.T @ X_centered)
        idx = np.argsort(eigenvalues)[::-1]

        V = eigenvectors[:, idx]
        V = V[:, : self.n_components]

        singular_values = np.array(
            [np.linalg.norm(X_centered @ V[:, i]) for i in range(V.shape[1])]
        )

        self.components_ = V.T
        self.singular_values_ = singular_values

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA must be fit before transform")

        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA must be fit before inverse_transform")

        return scores @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
