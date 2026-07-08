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


def compute_v(X, y):
    X_r = X[y == 0]  # Republican
    X_d = X[y == 1]  # Democrat

    # Compute the mean vote vectors for Republican and Democrat
    mu_r = np.mean(X_r, axis=0)
    mu_d = np.mean(X_d, axis=0)

    v = mu_r - mu_d

    return v


def project_1d(X, v):
    # Normalize vector v
    v_normalized = v / np.linalg.norm(v)
    # Project data onto v
    projection = v_normalized.T @ X.T
    return projection

# project the data onto the top-k eigenvectors
def encode(X, Q, k):
    return X @ Q[:, :k]  # transpose of Q.T @ X.T which we had in class

# reconstruct the data from the top-k eigenvectors
def decode(scores, Q, k):
    return scores @ Q[:, :k].T  # transpose of Q @ scores.T which we had in class


# load dataset
X, y = load_dataset()

# Compute vector v
v = compute_v(X, y)

# Perform 1D projection onto v
projections = project_1d(X, v)

# Plot the 1D projections
plt.figure(figsize=(8, 6))
for party, color, label in [(0, 'red', 'R'), (1, 'blue', 'D')]:
    plt.scatter(
        projections[y == party],  # 1D projections
        np.zeros_like(projections[y == party]),  # Zero height for 1D plot
        c=color,
        label=label,
        alpha=0.7,
        edgecolor='k'
    )

# Add labels, legend, and title
plt.xlabel('Projection onto v')
plt.ylabel('')
plt.title('1D Projection of Senators Votes onto v')
plt.legend()
plt.grid(True)
plt.savefig('projection_1d.png')
plt.show()
