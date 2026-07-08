import sys; args = sys.argv[1:]
import math, random, re, time


def dot(v1, v2): return sum([v1[i] * v2[i] for i in range(len(v1))])
def logistic(x): return 1 / (1 + math.e ** -x)
def lst_str(lst): return str(lst).replace(',', '').replace('[', '').replace(']', '')
def get_error(training, output): return sum(training[k] != (output[k] >= 0.5) for k in range(len(training)))
def in_circle(x, y, r, op):
    if '<' in op: return x ** 2 + y ** 2 <= r ** 2
    if '>' in op: return x ** 2 + y ** 2 >= r ** 2


def feed_forward(inputs, weights):
    x_vals, y_vals = [inputs + [1]], []
    for i in range(len(weights)):
        if i < len(weights) - 1:
            y_vals.append([dot(x_vals[i], w) for w in weights[i]])
            x_vals.append([logistic(val) for val in y_vals[-1]])
        else: y_vals.append([x_vals[i][j] * weights[i][j] for j in range(len(weights[i]))])
    return x_vals, y_vals


def weight_structure(layer_sizes):
    layer_sizes, weights = layer_sizes[:-1][::-1], []
    for k, s in enumerate(layer_sizes):
        if k > 0: weights.append([[random.random()*2-1 for _ in range(i * s, i * s + s)] for i in range(layer_sizes[k - 1])])
        else: weights.append([random.random()*2-1 for _ in range(s)])
    return weights[::-1]


def update_weights(training, x_vals, y_vals, weights):
    error, learn_rate = [], 0.1
    for layer in range(len(weights) - 1, -1, -1):  # loop thru layers in reverse order
        w_lst, x, y = weights[layer], x_vals[layer], y_vals[layer]
        if layer == len(weights) - 1:
            error.insert(0, [(training[i] - x[i] * w_lst[i]) * w_lst[i] * x[i] * (1 - x[i]) for i in range(len(w_lst))])
        else:
            error.insert(0, [sum(error[0][j] * w_lst[j][i] for j in range(len(w_lst))) * x[i] * (1 - x[i])
                             for i in range(len(w_lst[0]))])
        for j, w_to_j in enumerate(w_lst):
            if layer == len(weights) - 1: weights[layer][j] += learn_rate * ((training[j] - x[j] * w_lst[j]) * x[j])
            else:
                for i, w in enumerate(w_to_j): weights[layer][j][i] += learn_rate * (x[i] * error[1][j])


def backprop(radius, sign):
    num_samples = 4000
    input_lst, training_lst = generate_training(radius, sign, num_samples)
    start = time.time()
    layer_sizes = [2 + 1, 4, 2, 1, 1]
    weights = weight_structure(layer_sizes)
    errors, error_sum, test_num = [], 0, 0

    while True:
        inputs, training = input_lst[test_num % num_samples], training_lst[test_num % num_samples]
        x_vals, y_vals = feed_forward(inputs, weights)
        errors.append(get_error(training, y_vals[-1]))
        error_sum += get_error(training, y_vals[-1])
        update_weights(training, x_vals, y_vals, weights)

        # if test_num % 1000 == 0: print(test_num, error_sum)
        # if test_num > 1000 and error_sum < 19:
        #    input_lst, training_lst, num_samples = generate_training(radius, sign)
        #    errors, error_sum, test_num = [], 0, 0
        if test_num > 1000 and error_sum < 20:  # 19:  # modify this threshold
            print('new test')
            input_lst, training_lst = generate_training(radius, sign, num_samples)
            errors, error_sum, test_num = [], 0, 0
        if time.time() - start > 97: return layer_sizes, weights
        if test_num > 1000: error_sum -= errors[test_num-1000]
        if test_num > 100000 and error_sum > 300:
            weights = weight_structure(layer_sizes)
            errors, error_sum, test_num = [], 0, 0
        test_num += 1


def generate_training(radius, sign, num_samples):
    input_lst, training_lst = [], []
    for k in range(num_samples):
        input_lst.append([random.random() * 3 - 1.5, random.random() * 3 - 1.5])
        training_lst.append([in_circle(input_lst[-1][0], input_lst[-1][1], radius, sign)])
    return input_lst, training_lst


def main(test=False):
    global args

    if test: args = ['x*x+y*y<1.1']
    start = time.time()
    split = re.search('[<>]=?', args[0])
    radius, sign = float(args[0][split.end():]) ** 0.5, split.group()

    layer_sizes, weights = backprop(radius, sign)
    print('Layer counts', lst_str(layer_sizes))
    for weight_layer in weights: print(lst_str(weight_layer))

    if test:
        print(f'Time {time.time() - start}')
        num_goofs = 0
        for k in range(100000):
            inputs = [random.random() * 3 - 1.5, random.random() * 3 - 1.5]
            out = feed_forward(inputs, weights)[1][-1][0]
            num_goofs += (out >= 0.5) != in_circle(inputs[0], inputs[1], radius, sign)
        print(num_goofs)


if __name__ == '__main__': main(True)


# Tristan Devictor, pd. 6, 2024
