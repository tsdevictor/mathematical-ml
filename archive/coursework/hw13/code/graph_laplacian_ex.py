import numpy as np
import matplotlib.pyplot as plt


def make_data(n=200, n_c=0):
    r = [1, 2, 3]
    theta = np.linspace(0, 2 * np.pi, 3 * n)
    theta = theta[np.random.permutation(3 * n)]
    x1 = np.cos(theta)
    x2 = np.sin(theta)
    label = np.zeros(3 * n)
    X = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1)
    for i in range(3):
        idx = np.arange(i * n, (i + 1) * n)
        X[idx] = r[i] * X[idx]
        X[idx, 0] = X[idx, 0]
        label[idx] = i

    # plot the data
    X = X + np.random.normal(0, 0.1, X.shape)
    if n_c > 0:
        x2 = np.linspace(-3, 3, n_c)
        x1 = np.zeros(n_c)
        X_c = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1)
        X = np.concatenate([X, X_c], axis=0)
        label = np.concatenate([label, np.ones(n_c) * 3], axis=0)

    plt.scatter(X[:, 0], X[:, 1], c=label)
    plt.show()

    return X, label


def make_knn(X, k=10):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(X[i] - X[j])

    A = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(dist[i])[:k]
        A[i, idx] = 1

    A = A + A.T - A * A.T
    return A


def make_L(A):
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L


def show_f(X, f):
    plt.figure(1)
    plt.scatter(range(len(f)), f)
    plt.xlabel('Index')
    plt.ylabel('Eigenvector values')
    plt.title('Eigenvector Components')
    plt.show()

    plt.figure(2)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=f, cmap='viridis')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Data Points Colored by Eigenvector Values')
    plt.colorbar(scatter, label='Eigenvector value')
    plt.show()


X, labels = make_data(n_c=20)
A = make_knn(X)
L = make_L(A)
evals, evecs = np.linalg.eigh(L)

# for i in range(5):
#      print("Eigenvector", i)
#      show_f(X, evecs[:,i])


print(evals)

print("Eigenvector 1 and 2")
plt.scatter(evecs[:, 1], evecs[:, 2], c=labels)
plt.show()
