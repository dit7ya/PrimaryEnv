import numpy as np

eps = 1e-12

def relu(x):
    return max(x, 0)


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def layer_forward(x, w, b):
    z = np.dot(x, w) + b
    a = relu(z)
    return a


def mse(y, y_pred):
    mse = np.mean((y - y_pred) ** 2)
    return mse


def grad_relu(a):
    if a > 0:
        grad = 1
    else:
        grad = 0
    return grad


def grad_sigmoid(z):
    return z * (1 - z)


# forward pass

n_1 = 64
n_2 = 32
n_3 = 8
n_4 = 1


def initilize_params(X):
    n = X.shape[0]

    w1 = np.random.randn(n_1, n)
    b1 = np.zeros(n_1)

    w2 = np.random.randn(n_2, n_1)
    b2 = np.zeros(n_2)

    w3 = np.random.randn(n_3, n_2)
    b3 = np.zeros(n_3)

    w4 = np.random.randn(n_4, n_3)
    b4 = np.zeros(n_4)

    params = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4,
              'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4}

    return params


def forward_pass_single(X, params):

    w1 = params['w1']
    w2 = params['w2']
    w3 = params['w3']
    w4 = params['w4']

    b1 = params['b1']
    b2 = params['b2']
    b3 = params['b3']
    b4 = params['b4']

    z1 = np.dot(w1, X) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)

    z4 = np.dot(w4, a3) + b4
    a4 = sigmoid(z4)

    return a4, params


test_X = np.random.randint(low=0, high=100, size=32)


# backward pass

def backward_pass_single(X, params):




print(test_X.shape)

print(forward_pass_single(test_X))
