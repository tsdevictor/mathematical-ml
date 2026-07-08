import numpy as np

from mathematical_ml.kernels import rbf_kernel


class KMeans:
    """k-means clustering implemented in NumPy."""

    def __init__(
        self,
        n_clusters: int,
        max_iters: int = 100,
        tol: float = 1e-6,
        seed: int = 0,
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.centroids_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)

        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot exceed number of samples")

        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        centroids = X[indices]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [
                    X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                    for k in range(self.n_clusters)
                ]
            )

            if np.linalg.norm(new_centroids - centroids) < self.tol:
                centroids = new_centroids
                break

            centroids = new_centroids

        self.centroids_ = centroids
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise ValueError("KMeans must be fit before predict")

        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)


def spectral_clustering(
    X: np.ndarray,
    n_clusters: int,
    gamma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Cluster data using an RBF similarity graph and the graph Laplacian."""
    X = np.asarray(X, dtype=float)

    W = rbf_kernel(X, gamma=gamma)
    np.fill_diagonal(W, 0.0)

    D = np.diag(W.sum(axis=1))
    L = D - W

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    embedding = eigenvectors[:, 1 : n_clusters + 1]

    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    embedding = embedding / row_norms

    return KMeans(n_clusters=n_clusters, seed=seed).fit_predict(embedding)
