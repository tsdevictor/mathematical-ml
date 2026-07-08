import numpy as np


def qr_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the QR decomposition of A using classical Gram-Schmidt."""
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)

        if np.isclose(R[j, j], 0):
            raise ValueError("matrix has linearly dependent columns")

        Q[:, j] = v / R[j, j]

    return Q, R


def power_iteration(
    A: np.ndarray,
    num_iters: int = 1000,
    tol: float = 1e-10,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Estimate the dominant eigenvalue/eigenvector of a square matrix."""
    if A.shape[0] != A.shape[1]:
        raise ValueError("power iteration requires a square matrix")

    rng = np.random.default_rng(seed)
    v = rng.normal(size=A.shape[0])
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        v_next = A @ v
        norm = np.linalg.norm(v_next)

        if np.isclose(norm, 0):
            raise ValueError("power iteration encountered a zero vector")

        v_next = v_next / norm

        if np.linalg.norm(v_next - v) < tol or np.linalg.norm(v_next + v) < tol:
            v = v_next
            break

        v = v_next

    eigenvalue = float(v @ A @ v)
    return eigenvalue, v
