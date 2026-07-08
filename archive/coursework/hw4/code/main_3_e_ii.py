import numpy as np


def G(x):
    return -np.exp(-x * x / 2)


p = 10**6
samples = np.random.normal(0, 1, p)

estimate = np.mean(G(samples))
exact = -(np.sqrt(2)) / 2

print(f"Estimate: {estimate}")
print(f"Exact Integral: {exact}")
print(f"Error (exact - estimate): {exact - estimate}")
