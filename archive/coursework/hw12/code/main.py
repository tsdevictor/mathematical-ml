import pandas as pd
import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt


def load_data(f):
    data = pd.read_csv('dataset_separable.csv')
    X = data.iloc[:, :2].to_numpy()
    y = data.iloc[:, 2].to_numpy()

    return X, y


def solve_dual(X, y):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = y[i] * y[j] * (X[i] @ X[j])

    P = 0.5 * D + 1e-6 * np.eye(n)
    q = -1.0 * np.ones(n)
    G = -1.0 * np.eye(n)
    h = np.zeros(n)
    A = y.reshape(1, -1)
    b = np.array([0.0])
    
    nu_star = solve_qp(P, q, G, h, A, b, solver='quadprog')

    return nu_star


def compute_a_b(X, y, nu_star):
    support_idx = np.where(nu_star > 1e-7)[0]

    a = 0.5 * np.sum((nu_star * y)[:, None] * X, axis=0)  # shape = (2,)
    b_candidates = []
    for i in support_idx:
        xi = X[i]
        yi = y[i]
        val_b = (1.0 / yi) - np.dot(a, xi)
        b_candidates.append(val_b)

    b = np.mean(b_candidates)

    return a, b, support_idx


X, y = load_data('dataset_separable.csv')
nu_star = solve_dual(X, y)
a, b, support_idx = compute_a_b(X, y, nu_star)

scores = X.dot(a) + b
y_pred = np.sign(scores)
accuracy = np.mean(y_pred == y)
print(f'Accuracy = {accuracy * 100:.1f}%')

X_pos = X[y == 1]
X_neg = X[y == -1]

plt.figure(figsize=(8, 6))

plt.scatter(X_pos[:, 0], X_pos[:, 1], label='y=+1')
plt.scatter(X_neg[:, 0], X_neg[:, 1], label='y=-1')

plt.scatter(X[support_idx, 0],
            X[support_idx, 1],
            marker='x',
            s=100,
            label='Support Vectors')

xmin, xmax = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
xx = np.linspace(xmin, xmax, 200)
if abs(a[1]) > 1e-14:
    yy = -(b + a[0]*xx) / a[1]
    plt.plot(xx, yy, 'k--', label='Decision boundary')

plt.legend()
plt.title('SVM Decision Boundary and Support Vectors')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('SVM.png')
plt.show()
