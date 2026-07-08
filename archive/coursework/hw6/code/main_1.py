import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
x = np.random.uniform(0, 10, 1000)
epsilon = np.random.normal(0, 1, 1000)
y = 1 + 2*x + epsilon

X = np.column_stack((np.ones(1000), x))
alpha = np.linalg.inv(X.T @ X) @ X.T @ y
predicted_y = alpha[0] + alpha[1] * x

plt.scatter(x, y, color='blue')
plt.plot(x, predicted_y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Data points and line of best fit")
plt.savefig('HW6_P1.png')
plt.show()
