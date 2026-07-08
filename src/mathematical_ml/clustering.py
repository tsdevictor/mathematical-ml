import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph


def make_knn_graph(X: np.ndarray, n_neighbors: int = 30):
    """Build a symmetric k-nearest-neighbor graph."""
    A = kneighbors_graph(X, n_neighbors=n_neighbors)
    A = A + A.T - A.multiply(A.T)
    return A


def graph_laplacian(A):
    """Compute the unnormalized graph Laplacian L = D - A."""
    degrees = np.asarray(A.sum(axis=1)).ravel()
    D = diags(degrees)
    return D - A


def spectral_clustering(
    X: np.ndarray,
    n_clusters: int,
    n_embedding_dims: int = 3,
    n_neighbors: int = 30,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral clustering using a kNN graph and graph Laplacian.

    Adapted from the original coursework implementation.
    """
    X = np.asarray(X, dtype=float)

    A = make_knn_graph(X, n_neighbors=n_neighbors)
    L = graph_laplacian(A)

    _, eigenvectors = eigsh(L, k=n_embedding_dims)
    embedding = eigenvectors[:, 1:n_embedding_dims]

    labels = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(
        embedding
    )

    return embedding, labels


def cluster_homogeneity(y_true: np.ndarray, labels: np.ndarray) -> float:
    """Compute average cluster purity/homogeneity."""
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)

    scores = []

    for cluster in np.unique(labels):
        cluster_indices = np.where(labels == cluster)[0]
        true_labels = y_true[cluster_indices]

        if len(true_labels) == 0:
            scores.append(0.0)
            continue

        counts = np.bincount(true_labels.astype(int))
        scores.append(np.max(counts) / len(true_labels))

    return float(np.mean(scores))
