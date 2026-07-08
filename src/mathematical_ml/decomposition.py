import numpy as np


class PCA:
    """Principal component analysis implemented from first principles."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)

        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed the number of features")

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        _, singular_values, vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components_ = vt[: self.n_components]
        self.explained_variance_ = (
            singular_values[: self.n_components] ** 2 / (X.shape[0] - 1)
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA must be fit before transform")

        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA must be fit before inverse_transform")

        return Z @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
