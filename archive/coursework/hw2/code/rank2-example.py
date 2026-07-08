import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import stats

def make_X(n=5000, p=800):
    # make two random p dim vectors called v1 and v2 that are normalized
    v1 = np.random.randn(p)
    v1 = v1 / np.sqrt(np.sum(v1**2))
    v2 = np.random.randn(p)
    v2 = v2 / np.sqrt(np.sum(v2**2))
    V = np.column_stack([v1, v2])

    # make two random vectors alpha1 and alph2 of dimension n, normalized
    alpha1 = np.random.randn(n)
    alpha1 = alpha1 / np.sqrt(np.sum(alpha1**2))
    alpha2 = np.random.randn(n)
    alpha2 = alpha2 / np.sqrt(np.sum(alpha2**2))
    alpha = np.column_stack([alpha1, alpha2])

    s = np.diag([np.sqrt(6.0), np.sqrt(4.0)])

    X = alpha @ s @V.T
    # add noise to X
    X = X + 2*np.random.randn(n, p)/np.sqrt(n)

    return X, v1, v2


X, v1, v2 = make_X()
n,p = X.shape

Sigma = (X.T @ X)
evals, evecs = linalg.eigh(Sigma)
evals = evals[::-1]
evecs = evecs[:,::-1]
plt.hist(evals, bins=50)
plt.show()

v1_pred = evecs[:,0]
v2_pred = evecs[:,1]    
plt.scatter(v1, v1_pred)
plt.show()
plt.scatter(v2, v2_pred)
plt.show()


