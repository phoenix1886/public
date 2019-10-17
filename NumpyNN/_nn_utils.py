import numpy as np

__all__ = ['COST_FUNCTIONS', 'FUNCTIONS_MAP']


def sigmoid(z):
    return (
            1 / (1 + np.exp(-z))
    )

def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    return np.tanh(z)


def d_tanh(z):
    return 1 - np.square(np.tanh(z))


def relu(z):
    return np.maximum(0, z)


def d_relu(z):
    return np.where(z <= 0, 0, 1)


def d_leaky_relu(z, scale=0.001):
    return np.where(z <= 0, scale, 1)


def leaky_relu(z, scale=0.001):
    return np.maximum(scale * z, z)


COST_FUNCTIONS = {
    'logloss': {
        'func': (
            lambda a, y: -(
                    y * np.log(a) + (1 - y) * np.log(1 - a)
            )
        ),
        'dfunc': (
            lambda a, y: -(
                    y / a - (1 - y) / (1 - a)
            )
        )
    }
}

FUNCTIONS_MAP = {
    'sigmoid': {
        'func': sigmoid,
        'dfunc': d_sigmoid
    },
    'tanh': {
        'func': tanh,
        'dfunc': d_tanh
    },
    'relu': {
        'func': relu,
        'dfunc': d_relu
    },
    'leaky_relu': {
        'func': leaky_relu,
        'dfunc': d_leaky_relu
    }
}

