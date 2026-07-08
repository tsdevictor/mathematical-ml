import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import stats

def load_MNIST(N=None, s=1):
    print("Loading MNIST dataset...")
    data = pd.read_csv('mnist.csv', header=None)
    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()
    X = X/255

    # subsample N
    if N is not None:
        idx = np.random.choice(X.shape[0], N, replace=False)
        X = X[idx]
        y = y[idx]

    mu = np.mean(X, axis=0)
    X = X - mu

    X_noise = X + s*stats.norm.rvs(size=X.shape)
    return X, X_noise, y


def encode(X, Q, k):
    return X @ Q[:, :k]

def decode(scores, Q, k):
    return scores @ Q[:, :k].T

def show_image(x):
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()

X, X_noise, y = load_MNIST(5000, s=1)
n,p = X.shape

Sigma = (X_noise.T @ X_noise)/n
evals, evecs = linalg.eigh(Sigma)
evals = evals[::-1]
evecs = evecs[:,::-1]
plt.hist(evals, bins=50)
plt.show()



# I think 10 might be good
k = 60

# Randomly select k columns
random_indices = np.random.choice(p, k, replace=False)
Q = evecs[:, random_indices]

# encode
scores = encode(X_noise, Q, k)
X_rec = decode(scores, Q, k)

var_frac = np.sum(evals[:k])/np.sum(evals)
print(f"Variance retained: {var_frac:.2f}")
i = 1
show_image(X[i])
show_image(X_noise[i])
show_image(X_rec[i])




