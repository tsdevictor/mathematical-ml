import numpy as np

from mathematical_ml.linear_algebra import (
    block_power_iteration_xxt,
    householder_qr,
)


def test_householder_qr_reconstructs_matrix():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 4))

    Q, R = householder_qr(A)

    assert np.allclose(Q @ R, A)
    assert np.allclose(Q.T @ Q, np.eye(8))


def test_block_power_iteration_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))

    eigenvalues, eigenvectors = block_power_iteration_xxt(
        X,
        n_components=2,
        num_iters=20,
    )

    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (20, 2)
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(2), atol=1e-6)
