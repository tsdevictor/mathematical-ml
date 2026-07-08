def weight_structure(layer_sizes):
    last = layer_sizes.pop()  # no weight after last element (the output)
    rand_weights = []
    for i, size in enumerate(layer_sizes):
        if i < len(layer_sizes) - 1:
            rand_weights.append([random.random() * 2 - 1 for _ in range(size * layer_sizes[i + 1])])
        else:
            rand_weights.append([random.random() * 2 - 1 for _ in range(size)])
    layer_sizes.append(last)

    weights, size = [], 1
    for i, w in enumerate(rand_weights[::-1]):
        size = len(w) // size
        if i == 0: weights.append(w)
        else: weights.append([[w[k] for k in range(i * size, (i + 1) * size)] for i in range(len(w) // size)])
    weights.reverse()


def weight_structure(sizes):
    layer_sizes = sizes.pop()[::-1]
    weights = []
    for k, s in enumerate(layer_sizes):
        if k > 0:
            next_size = layer_sizes[k - 1]
            weights.append([[random.random()*2-1 for _ in range(i * s, i * s + s)] for i in range(next_size)])
        else: weights.append([random.random()*2-1 for _ in range(s)])
