import numpy as np

from mathematical_ml.linear_algebra import power_iteration, qr_decomposition


def test_qr_decomposition_reconstructs_matrix():
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
    Q, R = qr_decomposition(A)

    assert np.allclose(Q @ R, A)
    assert np.allclose(Q.T @ Q, np.eye(2))


def test_power_iteration_dominant_eigenvalue():
    A = np.array([[3.0, 0.0], [0.0, 1.0]])
    eigenvalue, eigenvector = power_iteration(A)

    assert np.isclose(eigenvalue, 3.0, atol=1e-3)
    assert np.isclose(np.linalg.norm(eigenvector), 1.0)
