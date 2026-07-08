import numpy as np

from mathematical_ml.clustering import cluster_homogeneity, spectral_clustering


def test_spectral_clustering_shapes():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ]
    )

    embedding, labels = spectral_clustering(
        X,
        n_clusters=2,
        n_embedding_dims=3,
        n_neighbors=2,
    )

    assert embedding.shape[0] == X.shape[0]
    assert labels.shape == (X.shape[0],)


def test_cluster_homogeneity_perfect_labels():
    y = np.array([0, 0, 1, 1])
    labels = np.array([1, 1, 0, 0])

    assert cluster_homogeneity(y, labels) == 1.0
