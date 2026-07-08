import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def generate_data(n, mu=1):
    X = np.zeros((n, 2))
    y = np.random.binomial(1, 0.5, n)

    for i in range(n):
        if y[i] == 0:
            X[i] = np.random.normal(0, 1, 2)
        else:
            X[i] = np.random.normal([mu, 0], 1, 2)

    return X, y


def visualize_dataset(X, y, save_path, clf=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.6)

    if clf is not None:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        plt.contour(xx, yy, Z, [0.5], colors='black', linewidths=2)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Logistic Regression Decision Boundary')
    plt.savefig(save_path)
    plt.show()


for n in [1000, 5000, 10000, 100000]:
    X, y = generate_data(n, 1)

    clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')
    clf.fit(X, y)

    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)

    w1, w2 = clf.coef_[0]
    b = clf.intercept_[0]
    
    print(f'\nn={n}, Accuracy: {accuracy * 100:.2f}%')
    print('Decision Boundary Equation:')
    print(f'{w1:.3f}x1 + {w2:.3f}x2 = {-b:.3f}')

    visualize_dataset(X, y, f'imgs2/logistic_regression_{n}.png', clf)
