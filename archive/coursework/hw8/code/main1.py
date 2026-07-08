import numpy as np


# compute gradient of banana function
def banana_grad(alpha):
    x, y = alpha
    f_x = -400 * x * (y - x ** 2) - 2 * (1-x)
    f_y = 200 * (y - x ** 2)
    return np.array([f_x, f_y])


def gradient_descent(alpha, lr, num_iter, v=False):
    for i in range(num_iter):
        alpha = alpha - lr * banana_grad(alpha)
        if v and i % 100000 == 0:
            print(alpha)

    return alpha


# compute hessian of banana function
def banana_hessian(alpha):
    x, y = alpha
    f_xx = 1200 * x ** 2 - 400 * y + 2
    f_xy = -400 * x  # f_xy = f_yx by Clairaut's theorem
    f_yy = 200
    return np.array([[f_xx, f_xy], [f_xy, f_yy]])


def newton_method(alpha, num_iter, v=False):
    for i in range(num_iter):
        alpha = alpha - np.linalg.inv(banana_hessian(alpha)) @ banana_grad(alpha)
        if v:
            print(alpha)
    return alpha


alpha = np.array([4, 4])
print('Gradient descent method:')
print(alpha)
print(gradient_descent(alpha, 0.0001, 600000, True))

print('\n\nNewton\'s method:')
print(alpha)
print(newton_method(alpha, 5, True))
