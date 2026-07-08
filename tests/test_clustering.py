import numpy as np

from mathematical_ml.clustering import KMeans, spectral_clustering


def test_kmeans_returns_labels():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ]
    )

    labels = KMeans(n_clusters=2, seed=0).fit_predict(X)

    assert labels.shape == (4,)
    assert len(set(labels)) == 2


def test_spectral_clustering_returns_labels():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ]
    )

    labels = spectral_clustering(X, n_clusters=2, gamma=1.0)

    assert labels.shape == (4,)
