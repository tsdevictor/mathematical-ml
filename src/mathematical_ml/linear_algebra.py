import numpy as np


def householder_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the QR decomposition using Householder reflections.

    Adapted from the Householder QR implementation in the original coursework.
    """
    A = np.asarray(A, dtype=float)
    n, p = A.shape

    Q = np.eye(n)
    R = A.copy()

    for k in range(p):
        x = R[k:, k]

        if np.allclose(x, 0):
            continue

        e1 = np.zeros_like(x)
        e1[0] = 1.0

        sign = 1.0 if x[0] >= 0 else -1.0
        v = x + sign * np.linalg.norm(x) * e1
        v_norm = np.linalg.norm(v)

        if np.isclose(v_norm, 0):
            continue

        v = v / v_norm

        R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2 * Q[:, k:] @ np.outer(v, v)

    return Q, R


def gram_schmidt(X: np.ndarray) -> np.ndarray:
    """Orthonormalize the columns of X using modified Gram-Schmidt."""
    X = np.asarray(X, dtype=float)
    Q = np.zeros_like(X, dtype=float)

    for i in range(X.shape[1]):
        v = X[:, i].copy()

        for j in range(i):
            v -= np.dot(Q[:, j], v) * Q[:, j]

        norm = np.linalg.norm(v)

        if np.isclose(norm, 0):
            raise ValueError("columns are linearly dependent")

        Q[:, i] = v / norm

    return Q


def block_power_iteration_xxt(
    X: np.ndarray,
    n_components: int = 2,
    num_iters: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate top eigenpairs of X X^T using block power iteration.

    This is adapted from the original coursework implementation, which iterates

        U <- X (X^T U)

    and re-orthonormalizes the columns after each iteration.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, n_components))
    U = gram_schmidt(U)

    for _ in range(num_iters):
        U = X @ (X.T @ U)
        U = gram_schmidt(U)

    eigenvalues = np.array(
        [
            float(U[:, i] @ ((X @ X.T) @ U[:, i]) / (U[:, i] @ U[:, i]))
            for i in range(n_components)
        ]
    )

    return eigenvalues, U


# backwards-compatible alias for README/tests
qr_decomposition = householder_qr
