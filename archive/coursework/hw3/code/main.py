import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt


def load_dataset(f):
   print(f'Loading {f} dataset...')
   data = pd.read_csv(f, header=None)
   y = data.iloc[1:, 0].to_numpy()

   # convert y to 0 for republican, 1 for democrat
   y = np.where([name[-1] == 'R' for name in y], 0, 1)

   # force strings to become numbers
   X = data.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy()

   # subtract the mean of each column from X
   mu = np.mean(X, axis=0)
   X = X - mu

   return X, y


# project the data onto the top-k eigenvectors
# transpose of Q.T @ X which we had in class
def encode(X, Q, k):
   return X @ Q[:, :k]


# reconstruct the data from the top-k eigenvectors
# transpose of Q @ scores.T which we had in class
def decode(scores, Q, k):
   return scores @ Q[:, :k].T


def main():
   # load dataset
   X, y = load_dataset('test.csv')

   # compute the eigenvalues and eigenvectors of X.T @ X
   evals, evecs = linalg.eigh(X.T @ X)

   # sort eigenvalues and eigenvectors
   idx = np.argsort(evals)[::-1]

   # V is the matrix whose columns are the eigenvectors of X.T @ X
   V = evecs[:, idx]
   V = V[:, :100]

   # here, just calculate the singular values; hold off on building the matrix
   # s_i = ||Xv_i||
   S = np.array([np.linalg.norm(X @ V[:, i]) for i in range(V.shape[1])])

   # u_i = (Xv_i) / ||Xv_i||
   U = X @ V / S

   # after computing U, make S into a diagonal matrix
   S = np.diag(S)

   print(f'U is a {U.shape[0]}x{U.shape[1]} matrix.')
   print(f'S is a {S.shape[0]}x{S.shape[1]} matrix.')
   print(f'V is a {V.shape[0]}x{V.shape[1]} matrix.')
   print(f'X is a {X.shape[0]}x{X.shape[1]} matrix.')

   print(f'\nDetermining if X = US(V^T)... {np.allclose(U @ S @ V.T, X)}\n')

   scores = U[:, :2] @ S[:2, :2]
   print(f'The PCA matrix is a {scores.shape[0]}x{scores.shape[1]} matrix')
   # PCA should be a 100 x 2 matrix; plot it like last time; should give same plot

   plt.figure(figsize=(8, 6))
   for party, color, label in [(0, 'red', 'R'), (1, 'blue', 'D')]:
      plt.scatter(
         scores[y == party, 0],  # First principal component
         scores[y == party, 1],  # Second principal component
         c=color,
         label=label,
         alpha=0.7,
         edgecolor='k'
      )

   # Add labels, legend, and title
   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.title('2D PCA of Senators Votes')
   plt.legend()
   plt.grid(True)
   plt.savefig('USV_PCA.png')
   plt.show()


if __name__ == '__main__': main()
