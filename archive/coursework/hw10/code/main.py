import numpy as np
import matplotlib.pyplot as plt


def grad(x, y, A):
   return 2 * A.T @ (A @ x - y)


def projected_grad_descent(y, A, lr, num_iter):
   p = A.shape[1]
   x = np.random.rand(p)

   losses = []
   for i in range(num_iter):
       x = x - lr * lr * grad(x, y, A)
       x = np.maximum(0, x)
       loss = np.linalg.norm(y - A @ x) ** 2
       losses.append(loss)

   return x, losses


np.random.seed(42)
n, p = 100, 90
A = np.random.randn(n, p)
x = np.random.rand(p)
y = A @ x

num_iter = 300
x_hat, loss_hist = projected_grad_descent(y, A, lr=1e-2, num_iter=num_iter)

print("Final loss:", loss_hist[-1])
plt.scatter(range(num_iter), loss_hist)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss History')
plt.savefig('loss_history.png')
plt.show()
