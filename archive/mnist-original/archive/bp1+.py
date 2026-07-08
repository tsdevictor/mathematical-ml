import sys; args = sys.argv[1:]
import math
import random

args = ['C:/Users/puffe/OneDrive/Documents/School/AI/NeuralNetworks/weights.txt.txt']

def dot(v1, v2): return sum([v1[i] * v2[i] for i in range(len(v1))])
def logistic(x): return 1 / (1 + math.e ** -x)
def lst_str(lst): return str(lst).replace(',', '').replace('[', '').replace(']', '')


def feed_forward(inputs, weights):
    inputs.append(1)
    x_vals = [inputs]
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


def backprop():
    weights, layer_sizes = [], []

    for test_num in range(15000):
        for k, line in enumerate(open(args[0])):
            inputs, training = line.split(' => ')
            inputs = [*map(int, inputs.split(' '))]
            training = [*map(int, training.split(' '))]

            if test_num == 0:
                layer_sizes = [len(inputs) + 1, 3, len(training), len(training)]
                weights = weight_structure(layer_sizes)

            x_vals, y_vals = feed_forward(inputs, weights)
            update_weights(training, x_vals, y_vals, weights)

    return layer_sizes, weights


def main():
    layer_sizes, weights = backprop()
    print('Layer counts', lst_str(layer_sizes))
    for weight_layer in weights: print(lst_str(weight_layer))
    print(feed_forward([0, 0, 1], weights)[1][-1])


if __name__ == '__main__': main()

# Tristan Devictor, pd. 6, 2024
