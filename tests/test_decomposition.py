import numpy as np

from mathematical_ml.decomposition import PCA


def test_pca_reduces_dimension():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))

    Z = PCA(n_components=2).fit_transform(X)

    assert Z.shape == (30, 2)


def test_pca_reconstruction_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    X_hat = pca.inverse_transform(Z)

    assert X_hat.shape == X.shape
