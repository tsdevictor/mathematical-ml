import numpy as np


def linear_kernel(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    if Y is None:
        Y = X
    return X @ Y.T


def polynomial_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    if Y is None:
        Y = X
    return (X @ Y.T + coef0) ** degree


def rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    if Y is None:
        Y = X

    X_norm = np.sum(X**2, axis=1)[:, None]
    Y_norm = np.sum(Y**2, axis=1)[None, :]
    sq_dists = X_norm + Y_norm - 2 * X @ Y.T

    return np.exp(-gamma * sq_dists)
