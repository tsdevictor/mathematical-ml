import numpy as np
import pandas as pd


def QR_householder(A):
    n, p = A.shape

    # initially, let Q = I and R = A
    Q = np.eye(n)
    R = A.copy()

    # householder algorithm from Trefethen and Bau
    for k in range(p):
        x = R[k:, k]

        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x + np.sign(x[0]) * np.linalg.norm(x) * e1
        v = v / np.linalg.norm(v)

        R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])   # (n-k x n-k) = (n-k x 1)(1 x n-k)(n-k x n-k)
        # as described in Trefethen and Bau, construct Q by computing its columns Qe1, Qe2,...
        Q[:, k:] -= 2 * Q[:, k:] @ np.outer(v, v)    # (n x n-k) = (n x n-k)(n-k x 1)(1 x n-k)

    return Q, R


# generate a random matrix
n = np.random.randint(10, 20)  # random n
p = np.random.randint(5, n)  # random p, with n > p
A = np.random.normal(size=(n, p))

# testing the function
Q, R = QR_householder(A)
print(np.allclose(Q @ R, A))

# make sure R is an upper triangular matrix
df = pd.DataFrame(R)
print(df.to_string(index=False, header=False, float_format=" {:.3f}".format))