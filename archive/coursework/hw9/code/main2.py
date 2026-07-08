import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_dataset(f):
   data = pd.read_csv(f)
   X = data.iloc[:, :2].to_numpy()
   y = data.iloc[:, 2].to_numpy()
   return X, y


def sigmoid(x):
   x = np.clip(x, -500, 500)
   return 1 / (1 + np.exp(-x))


def loss(beta, K, y):
   S = K @ beta
   P = sigmoid(S)
   P = np.clip(P, 1e-8, 1 - 1e-8)
   return np.sum(y * np.log(P) - np.log(1 + P))


def loss_grad(beta, K, y):
   S = K @ beta
   P = sigmoid(S)
   return K @ (y - P)


def gradient_ascent(beta, K, y, lr, num_iter, num_losses):
   loss_hist = []
   for i in range(num_iter):
       grad = loss_grad(beta, K, y)
       beta = beta + lr * grad
       if not i % (num_iter//num_losses): loss_hist.append(loss(beta, K, y))
   return beta, loss_hist


def find_beta(X, y, d, lr, num_losses, num_iter):
   beta = np.random.rand(X.shape[0])
   K = (X @ X.T + 1) ** d
   best_beta, loss_hist = gradient_ascent(beta, K, y, lr, num_iter, num_losses)

   plt.scatter(range(num_losses), loss_hist)
   plt.xlabel('Iterations (in {num_iter//num_losses}\'s)')
   plt.ylabel('Loss')
   plt.title('Loss History')
   plt.savefig(f'loss_history_d{d}.png')
   plt.show()

   return best_beta


def predict(X_train, X_test, beta, d):
   K_test = (X_test @ X_train.T + 1) ** d
   S_test = K_test @ beta
   P_test = sigmoid(S_test)
   return (P_test >= 0.5).astype(int)


def get_accuracy(y_pred, y_test):
   return np.mean(y_pred == y_test) * 100


def show_decision_boundary(X, y, beta, d):
   xx, yy = [np.linspace(X[:, i].min(), X[:, i].max(), 200) for i in range(2)]
   xx, yy = np.meshgrid(xx, yy)
   grid = np.column_stack((xx.ravel(), yy.ravel()))
   y_grid_pred = predict(X, grid, beta, d).reshape(xx.shape)

   plt.contourf(xx, yy, y_grid_pred, alpha=0.3, cmap=plt.cm.coolwarm)
   plt.scatter(X[:, 0], X[:, 1], c=y)
   plt.xlabel('x1')
   plt.ylabel('x2')
   plt.title(f'Decision Boundary (d={d})')
   plt.savefig(f'decision_boundary_d{d}.png')
   plt.show()


random_state = 39
np.random.seed(random_state)

X, y = load_dataset('dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
beta_d5 = find_beta(X_train, y_train, 5, 0.001, 100, 100000)
beta_d50 = find_beta(X_train, y_train, 50, 0.001, 100, 1000)
y_pred_d5 = predict(X_train, X_test, beta_d5, 5)
y_pred_d50 = predict(X_train, X_test, beta_d50, 50)

print(f'Accuracy for d=5: {get_accuracy(y_pred_d5, y_test):.2f}%')
print(f'Accuracy for d=50: {get_accuracy(y_pred_d50, y_test):.2f}%')
show_decision_boundary(X_train, y_train, beta_d5, 5)
show_decision_boundary(X_train, y_train, beta_d50, 50)
