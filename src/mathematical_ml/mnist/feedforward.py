import sys; args = sys.argv[1:]
import math


def hadamard(v1, v2): return [v1[i] * v2[i] for i in range(len(v1))]
def dot(v1, v2): return sum(hadamard(v1, v2))
def multiply_weights(values, weights): return [dot(values, w) for w in weights]


def identity(x): return x
def ramp(x): return (x > 0) * x
def logistic(x): return 1 / (1 + math.e ** -x)
def logistic2(x): return 2 * logistic(x) - 1


def setup():
    global FUNC, WEIGHTS, OUTPUTS
    FUNC = {'T1': identity, 'T2': ramp, 'T3': logistic, 'T4': logistic2}[args[1]]

    WEIGHTS, size = [], 1
    for i, w in enumerate([[*map(float, line.split(' '))] for line in open(args[0])][::-1]):
        size = len(w) // size
        if i == 0: WEIGHTS.append(w)
        else: WEIGHTS.append([[w[k] for k in range(i * size, (i + 1) * size)] for i in range(len(w) // size)])
    WEIGHTS.reverse()


def feed_forward():
    global OUTPUTS

    OUTPUTS = [[float(inp) for inp in args[2:]]]

    for i in range(len(WEIGHTS)):
        if i < len(WEIGHTS) - 1:
            OUTPUTS.append([FUNC(val) for val in multiply_weights(OUTPUTS[i], WEIGHTS[i])])
        else:
            OUTPUTS.append([OUTPUTS[i][n] * WEIGHTS[i][n] for n in range(len(WEIGHTS[i]))])


def main():
    setup()
    feed_forward()
    print(OUTPUTS[-1])


if __name__ == '__main__': main()


# Tristan Devictor, pd. 6, 2024
