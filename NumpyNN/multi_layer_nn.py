from ._nn_utils import FUNCTIONS_MAP, COST_FUNCTIONS
import numpy as np
from sklearn.base import ClassifierMixin

THRESHOLD_DEFAULT=0.5


class MultiLayerNN(ClassifierMixin):
    def __init__(
        self,
        dimensions=[1],
        cost_func_name='logloss',
        activation_functions=['sigmoid'],
        learning_rate=0.001,
        n_iter=20000,
        verbose=True
    ):
        self.verbose = verbose
        self._functions_map = FUNCTIONS_MAP
        self._cost_functions = COST_FUNCTIONS
        self.dimensions = dimensions
        self.cost_func_name = cost_func_name
        self.activation_functions = (
            activation_functions
            if hasattr(activation_functions, '__iter__')
            else [activation_functions] * len(dimensions)
        )
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.depth = len(self.dimensions)
        self._cost_func = self._cost_functions[self.cost_func_name]
        self.params = {}
        self.cache = {}
        self.grads = {}
        self.X = self.y = self.cost = None

    def _initiate_params(self):
        """Initiate parameters using He-initialization,
        see He et al., 2015
        """
        dimensions = self.dimensions
        params = {}
        previous_dimension = None
        for index, layer_dimension in enumerate(dimensions, 1):
            params['b' + str(index)] = np.zeros((layer_dimension, 1))
            dim = previous_dimension if previous_dimension else self.X.shape[0]
            params['W' + str(index)] = np.random.randn(
                layer_dimension,
                dim
            ) * np.sqrt(2.0/dim)
            previous_dimension = layer_dimension
        self.params = params

    def __calc_a(self, X):
        """Calculate activation function for output layer"""
        cache = {}
        params = self.params
        a = None

        for i in range(self.depth):
            layer_index = i + 1
            W = params['W' + str(layer_index)]
            b = params['b' + str(layer_index)]
            a_prev = X if layer_index == 1 else cache['a' + str(layer_index - 1)]
            z = np.dot(W, a_prev) + b
            cache['z' + str(layer_index)] = z
            activation_func_name = self.activation_functions[i]
            a = self._functions_map[activation_func_name]['func'](z)
            cache['a' + str(layer_index)] = a

        return a, cache

    def _forward_propagation(self):
        """Propagate forward and caclulate cost"""
        a, self.cache = self.__calc_a(self.X)
        self.cost = np.sum(self._cost_func['func'](a, self.y))

    def _backward_propagation(self):
        """Backward propagation and calculate gradients"""
        grads = {}
        m = self.X.shape[1]
        depth = self.depth
        for i in range(depth, 0, -1):
            a = self.cache['a' + str(i)]
            a_prev = self.cache['a' + str(i - 1)] if i > 1 else self.X
            y = self.y
            z = self.cache['z' + str(i)]
            g_name = self.activation_functions[i - 1]
            dg = self._functions_map[g_name]['dfunc']
            if i == depth:
                dJ_da = self._cost_func['dfunc']
                dz = dJ_da(a, y) * dg(z)
            else:
                dz = np.dot(self.params['W' + str(i + 1)].T, grads['dz' + str(i + 1)]) * dg(z)

            grads['dz' + str(i)] = dz
            grads['dW' + str(i)] = np.dot(dz, a_prev.T) / m
            grads['db' + str(i)] = np.mean(dz, axis=1, keepdims=True)
        self.grads = grads

    def _optimize(self):
        """Learning network"""
        for i in range(1, self.n_iter+1):
            self._forward_propagation()
            self._backward_propagation()
            for param_name, val in self.params.items():
                self.params[param_name] = (
                        self.params[param_name]
                        - self.learning_rate * self.grads['d' + param_name]
                )
            if self.verbose and not i % 1000:
                print('iter :{}, cost: {}'.format(i, self.cost))

    def fit(self, X, y):
        """Build a Multilayer Neural Network for binary classification

        Parameters:
        ----------
        X : array-like of shape = [n_samples, n_features]
        y: array of shape = [n_samples, ]

        Returns
        -------
        self : object
        """
        self.X = X.T
        self.y = y.reshape(1, -1)
        self._initiate_params()
        self._forward_propagation()
        self._backward_propagation()
        self._optimize()

    def predict(self, X, threshold=THRESHOLD_DEFAULT):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples, 1]
            The predicted classes (0 or 1).
        """
        a, _ = self.__calc_a(X.T)
        return np.where(a > threshold, 1, 0).reshape(-1, 1)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]

        Returns
        -------
        p : array of shape = [n_samples, 2]
            The class probabilities of the input samples.
        """
        a, _ = self.__calc_a(X.T)
        return np.vstack((a, 1-a)).T


