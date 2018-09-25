import numpy as np


def relu(x):
    s = np.maximum(0, x)
    return s


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def init_params(layer_dims):
    parameters = {}
    n_layers = len(layer_dims)  # n_layers is number of layer plus one

    for l in range(1, n_layers):
        parameters["w" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
            2 / layer_dims[l - 1])  # He initialization
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def forward_prop(x, parameters):
    caches = {"a0": x}
    num_layers = len(parameters) // 2  # num_layers is the number of layer
    a = x.copy()
    for l in range(1, num_layers):
        a_prev = a
        w = parameters["w" + str(l)]
        b = parameters["b" + str(l)]
        a = relu(np.dot(w, a_prev) + b)
        caches["a" + str(l)], caches["w" + str(l)] = a, w

    wl = parameters["w" + str(l)]
    bl = parameters["b" + str(l)]
    al = sigmoid(np.dot(wl, a) + bl)
    caches["a" + str(l)], caches["w" + str(l)] = al, wl
    return al, caches


def compute_cost(al, y):
    m = y.shape[1]

    cost = -(np.dot(y, np.log(al.T)) + np.dot((1 - y), np.log((1 - al).T))) / m
    cost = np.squeeze(cost)
    return cost


def back_propagation(AL, Y, caches):
    m = Y.shape[1]
    grads = {}
    L = len(caches) // 2  # L is the number of layers
    dZL = AL - Y
    dWL = 1. / m * np.dot(dZL, caches["A" + str(L - 1)].T)
    dbL = 1. / m * np.sum(dZL, axis=1, keepdims=True)
    grads["dW" + str(L)], grads["db" + str(L)] = dWL, dbL
    for l in reversed(range(1, L)):
        W_next = caches["W" + str(l + 1)]
        dZ_next = dZL
        A = caches["A" + str(l)]
        A_prev = caches["A" + str(l - 1)]
        dA = np.dot(W_next.T, dZ_next)
        dZ = np.multiply(dA, np.int64(A > 0))
        dW = 1. / m * np.dot(dZ, A_prev.T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        grads["dW" + str(l)], grads["db" + str(l)] = dW, db
        dZL = dZ
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters


def model(X, Y, learning_rate=0.01, num_iteration=15000, layer_dims=[0, 10, 5, 1]):
    m = X.shape[1]
    layer_dims[0] = X.shape[0]

    # Parameters initialization

    parameters = init_params(layer_dims)

    # Gradient Decent
    for i in range(0, num_iteration):
        AL, caches = forward_prop(X, parameters)

        cost = compute_cost(AL, Y)

        grads = back_propagation(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

    return parameters