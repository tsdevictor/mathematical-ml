import sys; args = sys.argv[1:]
import math
import random
import re

args = ['x*x+y*y<1.1']

def dot(v1, v2): return sum([v1[i] * v2[i] for i in range(len(v1))])
def logistic(x): return 1 / (1 + math.e ** -x)
def lst_str(lst): return str(lst).replace(',', '').replace('[', '').replace(']', '')
def mse(training, output): return 1/2 * sum((training[k] - output[k]) ** 2 for k in range(len(training)))
def in_circle(x, y, r, sign):
    if sign == '<': return x ** 2 + y ** 2 < r ** 2
    if sign == '>': return x ** 2 + y ** 2 > r ** 2
    if sign == '>=': return x ** 2 + y ** 2 >= r ** 2
    if sign == '<=': return x ** 2 + y ** 2 <= r ** 2


def feed_forward(inputs, weights):
    x_vals = [inputs + [1]]
    y_vals = []
    for i in range(len(weights)):
        if i < len(weights) - 1:
            y_vals.append([dot(x_vals[i], w) for w in weights[i]])
            x_vals.append([logistic(val) for val in y_vals[-1]])
        else:
            y_vals.append([x_vals[i][j] * weights[i][j] for j in range(len(weights[i]))])
    return x_vals, y_vals


def weight_structure(layer_sizes):
    last = layer_sizes.pop()  # no weight after last element (the output)
    rand_weights = []
    for i, size in enumerate(layer_sizes):
        if i < len(layer_sizes) - 1:
            rand_weights.append([random.random() for _ in range(size * layer_sizes[i + 1])])
        else:
            rand_weights.append([random.random() for _ in range(size)])
    layer_sizes.append(last)

    weights, size = [], 1
    for i, w in enumerate(rand_weights[::-1]):
        size = len(w) // size
        if i == 0:
            weights.append(w)
        else:
            weights.append([[w[k] for k in range(i * size, (i + 1) * size)] for i in range(len(w) // size)])
    weights.reverse()

    return weights


def update_weights(training, x_vals, y_vals, weights):
    scale = 0.1
    error = []
    for layer in range(len(weights) - 1, -1, -1):  # loop thru layers in reverse order
        w_lst = weights[layer]
        x, y = x_vals[layer], y_vals[layer]
        if layer == len(weights) - 1:
            error.insert(0, [(training[i] - x[i] * w_lst[i]) * w_lst[i] * x[i] * (1 - x[i]) for i in range(len(w_lst))])
        else:
            error.insert(0, [sum(error[0][j] * w_lst[j][i] for j in range(len(w_lst)))
                             * x[i] * (1 - x[i]) for i in range(len(w_lst[0]))])
        for j, w_to_j in enumerate(w_lst):
            if layer == len(weights) - 1:
                neg_grad = (training[j] - x[j] * w_lst[j]) * x[j]
                weights[layer][j] += scale * neg_grad
            else:
                for i, w in enumerate(w_to_j):
                    neg_grad = x[i] * error[1][j]
                    weights[layer][j][i] += scale * neg_grad


def backprop(input_lst, training_lst):
    layer_sizes = [2 + 1, 3, 2, 1, 1]
    weights = weight_structure(layer_sizes)
    errors, test_num = [], 0

    while True:
        for k in range(len(input_lst)):
            inputs, training = input_lst[k], training_lst[k]
            x_vals, y_vals = feed_forward(inputs, weights)
            errors.append(mse(training, y_vals[-1]))
            update_weights(training, x_vals, y_vals, weights)

            if test_num % 1000 == 0:
                print(errors[-1])

            if test_num > 10000 and sum(errors[-1000:]) < 0.0001: break
            elif test_num > 10000 and abs(errors[0] - errors[-1]) < 0.1:
                weights = weight_structure(layer_sizes)
                errors = []
            test_num += 1
        else: continue
        break

    return layer_sizes, weights


def main(test=False):
    global radius, sign
    split = re.search('[<>]=?', args[0])
    radius, sign = float(args[0][split.end():]) ** 0.5, split.group()

    inputs, training = [], []
    for i in range(10000):
        inputs.append([random.random() * 3 - 1.5, random.random() * 3 - 1.5])
        training.append([in_circle(inputs[-1][0], inputs[-1][1], radius, sign)])

    layer_sizes, weights = backprop(inputs, training)
    print('Layer counts', lst_str(layer_sizes))
    for weight_layer in weights: print(lst_str(weight_layer))

    if test:
        num_goofs = 0
        for k in range(100000):
            inputs = [random.random() * 3 - 1.5, random.random() * 3 - 1.5]
            out = feed_forward(inputs, weights)[1][-1][0]
            num_goofs += (out >= 0.5) != in_circle(inputs[0], inputs[1], radius, sign)
        print(num_goofs)


if __name__ == '__main__': main(True)


# Tristan Devictor, pd. 6, 2024
