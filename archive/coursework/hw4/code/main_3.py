import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def sample_signal(p):
  s = np.zeros(p)
  s[0] = np.random.choice([1, 2, 3, 4])  # s_1 chosen randomly from {1,2,3,4}
  # with probability 0.9, s_i = s_{i-1}; with probability 0.1, s_i is random
  for i in range(1, p):
      if np.random.rand() < 0.9: s[i] = s[i - 1]
      else: s[i] = np.random.choice([1, 2, 3, 4])
  return s


def compute_z(X):
  Y = X - X.mean(axis=1, keepdims=True)  # row-center X
  Sigma = 1/(Y.shape[1]) * Y @ Y.T

  evals, evecs = np.linalg.eigh(Sigma)
  evals, evecs = np.maximum(evals[::-1], 0), evecs[:, ::-1]
  Q = evecs
  D_sqrt = np.diag(np.sqrt(evals)) # remove neg. numbers from rounding error

  M = Q @ D_sqrt
  Z = np.linalg.inv(M) @ Y
  return Z


p = 100           # number of elements in each signal
num_signals = 5   # number of signals
S = np.vstack([sample_signal(p) for _ in range(num_signals)])  # row bind the 5 signals to form S, which is 5 x p

# plot the 5 signals (the rows of S)
plt.figure(figsize=(10, 6))
for i in range(num_signals):
  plt.plot(S[i], label=f'Signal {i + 1}')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Sampled Signals')
plt.savefig('3b_Sampled_Signals.png')

n = 100
A = np.random.randn(n, num_signals)  # A is a random Gaussian matrix
X = A @ S  # compute the data matrix
X_noise = X + 0.1*stats.norm.rvs(size=X.shape)  # add noise to X
Z = compute_z(X_noise)
