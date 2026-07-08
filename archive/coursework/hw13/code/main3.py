import pandas as pd
import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(f, test_size=0.5):
    data = pd.read_csv(f)
    X = data.iloc[:, :2].to_numpy()
    y = data.iloc[:, 2].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=41)
    return X_train, X_test, y_train, y_test


def kernel(X1, X2):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            diff = X1[i] - X2[j]
            K[i, j] = np.exp(-np.dot(diff, diff) / 0.01)
    return K


def solve_dual(K, y, C):
    n = len(y)
    D = np.outer(y, y) * K
    P = D + 1e-6 * np.eye(n)
    q = -np.ones(n)
    G = np.vstack([np.eye(n), -np.eye(n)])
    h = np.hstack([C * np.ones(n), np.zeros(n)])
    A = y.reshape(1, -1)
    b = np.array([0.0])

    nu = solve_qp(P, q, G, h, A, b, solver='quadprog')
    return nu


def predict(X_train, y_train, X_test, nu):
    K = kernel(X_train, X_test)
    scores = np.sum((nu * y_train)[:, None] * K, axis=0)
    
    support_vectors = nu > 1e-5
    if np.any(support_vectors):
        b = np.mean(y_train[support_vectors] - np.sum((nu * y_train)[:, None] * kernel(X_train, X_train[support_vectors]), axis=0))
    else:
        b = 0
    
    return np.sign(scores + b)


def plot_decision_boundary(X_train, y_train, nu, f):
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(X_train, y_train, grid, nu)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.2, colors=['blue', 'red'])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', s=20)
    plt.title('Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(f)
    plt.show()


X_train, X_test, y_train, y_test = load_data('dataset_non_separable.csv')
C_vals = [0.01, 0.1, 1, 10, 100, 1000]
train_accs = []
test_accs = []

K_train = kernel(X_train, X_train)
for C in C_vals:
    nu = solve_dual(K_train, y_train, C)

    y_train_pred = predict(X_train, y_train, X_train, nu)
    y_test_pred = predict(X_train, y_train, X_test, nu)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f'C={C:.2f}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}')

    plot_decision_boundary(X_train, y_train, nu, f'imgs3/C{C}.png')

plt.plot(C_vals, train_accs, label='Train Accuracy')
plt.plot(C_vals, test_accs, label='Test Accuracy')
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C')
plt.legend()
plt.grid(True)
plt.show()

best_C_idx = np.argmax(test_accs)
print(f'\nBest C value: {C_vals[best_C_idx]}')
print(f'Best test accuracy: {test_accs[best_C_idx]:.3f}')
