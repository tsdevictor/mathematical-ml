import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import stats

def load_dataset():
    print("Loading senators.csv dataset...")
    data = pd.read_csv('senators.csv', header=None)
    y = data.iloc[1:, 0].to_numpy()

    # convert y to 0 for republican, 1 for democrat
    y = np.where([name[-1] == 'R' for name in y], 0, 1)

    X = data.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy()
    # X = X/255

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(X)

    mu = np.mean(X, axis=0)
    X = X - mu

    return X, y



# project the data onto the top-k eigenvectors
def encode(X, Q, k):
    return X @ Q[:, :k]  # transpose of Q.T @ X which we had in class

# reconstruct the data from the top-k eigenvectors
def decode(scores, Q, k):
    return scores @ Q[:, :k].T  # transpose of Q @ scores.T which we had in class


# load dataset
X, y = load_dataset()
n, p = X.shape

# compute the covariance matrix and perform PCA
Sigma = (X.T @ X) / n
evals, evecs = linalg.eigh(Sigma)

# reverse the order of eigenvalues and eigenvectors (descending order)
evals = evals[::-1]
evecs = evecs[:,::-1]

# Select the top 2 eigenvectors for 2D PCA
k = 1
scores = encode(X, evecs, k)

plt.figure(figsize=(8, 6))

# Plot the scores along the 1D component
for party, color, label in [(0, 'red', 'R'), (1, 'blue', 'D')]:
    plt.scatter(
        scores[y == party, 0],  # First (and only) principal component
        np.zeros_like(scores[y == party, 0]),  # Plot on a single axis
        c=color,
        label=label,
        alpha=0.7,
        edgecolor='k'
    )

# Add labels, legend, and title
plt.xlabel('Principal Component 1')
plt.title('1D PCA of Senators Votes')
plt.legend()
plt.grid(True)
plt.savefig('my_figure_1d.png')
plt.show()
