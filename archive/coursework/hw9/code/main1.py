import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_dataset(f):
    data = pd.read_csv(f)
    X = data.iloc[:, :2].to_numpy()
    y = data.iloc[:, 2].to_numpy()
    return X, y


def visualize_dataset(X, y, save_path, alpha=None):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    if alpha is not None:
        x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        x2 = -(alpha[0] + alpha[1] * x1) / alpha[2]
        plt.plot(x1, x2, 'r-', linewidth=2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Scatter Plot' if alpha is None else 'Decision Boundary')
    plt.savefig(save_path)
    plt.show()


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def ll(alpha, x, y):
    ll = 0
    for i in range(len(alpha)):
        a = np.clip(-alpha @ x[i], -500, 500)
        ll += (1 - y[i]) * (-alpha @ x[i]) - np.log(1 + np.exp(a))
    return ll


def ll_grad(alpha, x, y):
    grad = np.zeros_like(alpha)
    for i in range(len(alpha)):
        grad += (sigmoid(-alpha @ x[i]) - (1 - y[i])) * x[i]
    return grad


def ll_hessian(alpha, x, rho):
    h = np.zeros((x.shape[1], x.shape[1]))
    for i in range(len(alpha)):
        sig = sigmoid(-alpha @ x[i])
        h -= sig * (1 - sig) * np.outer(x[i], x[i])
    return h - rho * np.eye(h.shape[0])


def newton_method(alpha, x, y, rho, num_iter):
    loss_history = []
    for i in range(num_iter):
        lr = 0.1
        alpha = alpha - lr * np.linalg.inv(ll_hessian(alpha, x, rho)) @ ll_grad(alpha, x, y)
        loss_history.append(ll(alpha, x, y))
    return alpha, loss_history


def predict(alpha, x_i):
    return 1 if sigmoid(alpha @ x_i) >= 0.5 else 0


def compute_accuracy(alpha, X_test, y_test):
    y_pred = [predict(alpha, X_test[i]) for i in range(len(X_test))]
    return np.mean(y_pred == y_test)


random_state = 42
np.random.seed(random_state)

X, y = load_dataset('dataset.csv')
# visualize_dataset(X, y, 'scatter.png')
X_ones = np.hstack((np.ones((X.shape[0], 1)), X))
X_train, X_test, y_train, y_test = (
    train_test_split(X_ones, y, test_size=0.2, random_state=random_state))
alpha = np.random.rand(X_train.shape[1])

num_iter = 1000
best_alpha, loss_hist = newton_method(alpha, X_train, y_train, 0.1, num_iter)
accuracy = compute_accuracy(best_alpha, X_test, y_test)
print(f'\nAccuracy: {accuracy * 100:.2f}%')
visualize_dataset(X, y, 'decision_boundary.png', alpha)

plt.scatter(range(num_iter), loss_hist)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss History')
plt.savefig('loss_history.png')
plt.show()
