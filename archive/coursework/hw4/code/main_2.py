import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg


def load_dataset(f):
    print(f"Loading {f} dataset...")
    data = pd.read_csv(f, header=None)

    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()
    X = X / 255

    mu = np.mean(X, axis=0)
    X = X - mu

    return X, y


# project the data onto the top-k eigenvectors
def encode(X, Q, k):
    return X @ Q[:, :k]  # transpose of Q.T @ X which we had in class


# reconstruct the data from the top-k eigenvectors
def decode(scores, Q, k):
    return scores @ Q[:, :k].T  # transpose of Q @ scores.T which we had in class


# display an mnist image
def show_image(x, k):
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'./images_k_dim_PCA/k_{k}.png')
    plt.show()


# load dataset
X, y = load_dataset('mnist.csv')
n, p = X.shape

# compute the covariance matrix and perform PCA
Sigma = (X.T @ X) / n
evals, evecs = linalg.eigh(Sigma)
evals, evecs = evals[::-1], evecs[:,::-1]

test_k_vals = [10, 20, 40, 60, 80, 100, 200, 300] # for different k, find frac of dataset variance captured by k-dim PCA
frac_vars = []
total_var = np.sum(evals ** 2)
for k in test_k_vals:
    print(f'Computing {k}-dimensional PCA...')
    scores = encode(X, evecs, k)
    X_rec = decode(scores, evecs, k)
    captured_var = np.sum(evals[:k]**2)
    frac_vars.append(captured_var / total_var)
    show_image(X_rec[0], k)

# visualize the fraction of variance captured by different k values
plt.plot(test_k_vals, frac_vars)
plt.xlabel('Number of Principal Components (k)')
plt.ylabel('Fraction of Variance Captured')
plt.title('Fraction of Variance Captured by k-Dimensional PCA')
plt.grid(True)
plt.savefig('2_Frac_var')
plt.show()
