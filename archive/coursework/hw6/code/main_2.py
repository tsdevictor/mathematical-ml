import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


x = np.linspace(1, 2, 500)  # discretization of the interval [1,2]
y = 1 / x

X = np.column_stack((np.sin(x), np.exp(x), gamma(x)))
alpha = np.linalg.inv(X.T @ X) @ (X.T @ y)
predicted_y = alpha[0] * np.sin(x) + alpha[1] * np.exp(x) + alpha[2] * gamma(x)

plt.scatter(x, y, color='blue')
plt.plot(x, predicted_y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title("f(x) = 1/x and approximation")
plt.savefig('HW6_P2.png')
plt.show()