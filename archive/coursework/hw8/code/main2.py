import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(f, test_size=0.2):
    print(f"Loading {f} dataset...\n")
    data = pd.read_csv(f, header=None)

    X = data.iloc[:, 1:].to_numpy() / 255
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = data.iloc[:, 0].to_numpy()
    y = np.where(y == 3, 1, 0)

    # training-testing data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# compute log likelihood
def ll(alpha, x, y):
    ll = 0
    for i in range(len(alpha)):
        a = np.clip(-alpha @ x[i], -500, 500)
        ll += (1 - y[i]) * (-alpha @ x[i]) - np.log(1 + np.exp(a))
    return ll


# compute gradient of log likelihood
def ll_grad(alpha, x, y):
    grad = np.zeros_like(alpha)
    for i in range(len(alpha)):
        grad += (sigmoid(-alpha @ x[i]) - (1 - y[i])) * x[i]
    return grad


def gradient_ascent(alpha, x, y, lr, num_iter, v=False):
    for i in range(num_iter):
        alpha = alpha + lr * ll_grad(alpha, x, y)
        if v and i % 400 == 0:
            print(ll(alpha, x, y))
    return alpha


# compute hessian of log likelihood
def ll_hessian(alpha, x, rho):
    h = np.zeros((x.shape[1], x.shape[1]))
    for i in range(len(alpha)):
        sig = sigmoid(-alpha @ x[i])
        h -= sig * (1 - sig) * np.outer(x[i], x[i])
    return h - rho * np.eye(h.shape[0])


def newton_method(alpha, x, y, rho, num_iter, v=False):
    for i in range(num_iter):
        alpha = alpha - np.linalg.inv(ll_hessian(alpha, x, rho)) @ ll_grad(alpha, x, y)
        if v:
            print(ll(alpha, x, y))
    return alpha


# classifier that determines if an image is a 3
def predict(alpha, x_i):
    return 1 if sigmoid(alpha @ x_i) >= 0.5 else 0


def compute_accuracy(alpha, X_test, y_test):
    y_pred = [predict(alpha, X_test[i]) for i in range(len(X_test))]
    accuracy = np.mean(y_pred == y_test)
    return accuracy


X_train, X_test, y_train, y_test = load_dataset('mnist.csv')

np.random.seed(42)
alpha = np.random.rand(X_train.shape[1])

print('Gradient ascent method:')
optimal_alpha = gradient_ascent(alpha, X_train, y_train, 0.01, 2000, True)
accuracy = compute_accuracy(optimal_alpha, X_test, y_test)
print(f'\nAccuracy: {accuracy * 100:.2f}%\n\n')

print('Newton\'s method:')
optimal_alpha = newton_method(alpha, X_train, y_train, 1, 20, True)
accuracy = compute_accuracy(optimal_alpha, X_test, y_test)
print(f'\nAccuracy: {accuracy * 100:.2f}%')
