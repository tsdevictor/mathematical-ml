import numpy as np
import scipy.io as sio
import scipy.sparse.linalg


def gram_schmidt(X):
    Q = np.zeros_like(X, dtype=float)
    for i in range(X.shape[1]):
        v = X[:, i]
        for j in range(i):
            v -= np.dot(Q[:, j], X[:, i]) * Q[:, j]
        v /= np.linalg.norm(v)
        Q[:, i] = v
    return Q


def power_iteration_XtX(X, k=2, num_iters=10, verbose=False):
    n = X.shape[0]
    U = np.random.normal(size=(n, k))

    if verbose: print('Computing iteration: ', end='')
    for i in range(num_iters):
        if verbose: print(i, end=',' if i < num_iters - 1 else '\n')
        U = X @ (X.T @ U)  # U = [Av1   Av2]
        U = gram_schmidt(U)

    evecs = U
    # Rayleigh quotient below
    evals = np.array([np.dot(U[:, i], (X @ X.T) @ U[:, i]) / np.dot(U[:, i], U[:, i]) for i in range(k)])

    return evals, evecs


# (a)
X = sio.mmread('X.mtx')
k = 2  # number of eigenvectors/values to compute
evals, evecs = power_iteration_XtX(X, k,100)
linalg_evals, linalg_evecs = scipy.sparse.linalg.eigsh(X @ X.T, k=k)
linalg_evals, linalg_evecs = linalg_evals[::-1], linalg_evecs[:,::-1]  # linalg computes them in reverse order

print(f'Computed eigenvalues: {evals}')
print(f'Linalg\'s eigenvalues: {linalg_evals}')
for i in range(k):
    print(f'Error in eigenvalue {i}: {linalg_evals[i] - evals[i]}')

print(f'Computed eigenvectors:\n {evecs}')
print(f'Linalg\'s eigenvectors:\n {linalg_evecs}')
for i in range(k):
    print(f'Error in eigenvector {i}: {np.linalg.norm(evecs[i] - linalg_evecs[i])}')

# (b)
X_dense = X.toarray()
evals, evecs = power_iteration_XtX(X_dense, k,100, True)
linalg_evals, linalg_evecs = scipy.sparse.linalg.eigsh(X_dense @ X.T, k=k)
linalg_evals, linalg_evecs = linalg_evals[::-1], linalg_evecs[:,::-1]  # linalg computes them in reverse order

print(f'Computed eigenvalues: {evals}')
print(f'Linalg\'s eigenvalues: {linalg_evals}')
for i in range(k): print(f'Error in eigenvalue {i}: {linalg_evals[i] - evals[i]}')

print(f'Computed eigenvectors:\n {evecs}')
print(f'Linalg\'s eigenvectors:\n {linalg_evecs}')
for i in range(k): print(f'Error in eigenvector {i}: {np.linalg.norm(evecs[i] - linalg_evecs[i])}')
