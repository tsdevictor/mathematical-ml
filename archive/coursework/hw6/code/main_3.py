import scipy.sparse.linalg
import scipy.io as sio
from numpy.random import randn
from scipy import linalg
import numpy as np


def random_projection(X, k, q):
    G = randn(X.shape[1], k)
    Y = X @ G
    for i in range(q):
        Y = X @ (X.T @ Y)
    Q, R = linalg.qr(Y, mode="economic")

    X_tilde = Q.T @ X
    u, S, V = linalg.svd(X_tilde, full_matrices=False)
    U = Q @ u
    return U, S, V.T


X = sio.mmread('X.mtx')
k = 2
q = 10

# computing additional more than 2 singular values
# gives a more accurate estimate of the first 2
fake_k = 5

Up, Sp, Vp = random_projection(X, fake_k, q=q)
Ul, Sl, Vl = scipy.sparse.linalg.svds(X)
Sl = Sl[::-1]  # scipy returns them in the reverse order

print(f'Computed singular values: {Sp[:k]}')
print(f'Linalg\'s singular values: {Sl[:k]}')
for i in range(k): print(f'Error in eigenvalue {i+1}: {100*(Sl[i] - Sp[i])/Sl[i]:.2f} %')



