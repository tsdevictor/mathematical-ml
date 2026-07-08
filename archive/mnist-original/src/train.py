import sys;

args = sys.argv[1:]
import math, re


def lst_str(lst): return str(lst).replace(',', '').replace('[', '').replace(']', '')


def dot(v1, v2): return sum([v1[i] * v2[i] for i in range(len(v1))])


def logistic(x): return 1 / (1 + math.e ** -x)


def set_weights():
    file = [[float(num) for num in nums] for line in open(args[0]) if (nums := re.findall(r'-?\.?\d+\.?\d*', line))][
           ::-1]
    layer_sizes = [len(file[0])]
    for i in range(len(file)): layer_sizes.append(len(file[i]) // layer_sizes[-1])
    layer_sizes.reverse()

    weights = []
    for layer, w in enumerate(file[::-1]):
        size = layer_sizes[layer]
        if layer < len(file) - 1:  # butterfly portion of network
            weights.append([[w[i] for i in range(j * size, (j + 1) * size)] for j in range(layer_sizes[layer + 1])])
        else:
            weights.append([[w[i]] for i in range(layer_sizes[layer])])
    return weights, layer_sizes


def bind_networks(weights_squarer, layer_sizes):
    num_layers = len(weights_squarer)
    weights = []
    for layer in range(num_layers):
        if layer == 0:  # first layer (to include bias)
            lst1 = [[w_to_j[0] / radius, 0, w_to_j[1] / radius] for j, w_to_j in enumerate(weights_squarer[layer])]
            lst2 = [[0, w_to_j[0] / radius, w_to_j[1] / radius] for j, w_to_j in enumerate(weights_squarer[layer])]
            weights.append(lst1 + lst2)
        elif layer < num_layers - 1:  # butterfly layers
            lst1 = [[w for w in w_to_j] + [0 for _ in w_to_j] for j, w_to_j in enumerate(weights_squarer[layer])]
            lst2 = [[0 for _ in w_to_j] + [w for w in w_to_j] for j, w_to_j in enumerate(weights_squarer[layer])]
            weights.append(lst1 + lst2)
        else:  # non-butterfly layer
            lst = [(w_to_j if '>' in sign else [-w for w in w_to_j]) for w_to_j in weights_squarer[layer]] * (
                        2 * layer_sizes[-2])
            weights.append(lst)
    return weights, [3] + [2 * size for size in layer_sizes[1:-1]] + [1]


def final_layer(weights, layer_sizes):
    # radius = 1  # float(re.search('\d*\.?\d+', args[1]).group()) ** 0.5
    # weights.append([[0.5 * (1 + math.e ** (-radius)), 0.5 * (1 + math.e ** (-radius))]])
    weights.append([[0.68394 if '>' in sign else 1.85914]])

    return weights, layer_sizes + [1]


def main():
    global radius, sign
    split = re.search('[<>]=?', args[1])
    radius, sign = float(args[1][split.end():]) ** 0.5, split.group()

    weights, layer_sizes = set_weights()
    weights, layer_sizes = bind_networks(weights, layer_sizes)
    weights, layer_sizes = final_layer(weights, layer_sizes)
    print(f'Layer counts: {lst_str(layer_sizes)}')
    for weight_layer in weights: print(lst_str(weight_layer))


if __name__ == '__main__': main()

# Tristan Devictor, pd. 6, 2024
