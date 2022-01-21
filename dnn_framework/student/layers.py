from dnn_framework import Layer
import numpy as np


class FullyConnectedLayer(Layer):

    def __init__(self, n_in, n_out):
        super().__init__()

        self.w = np.random.normal(loc=0.0, scale=np.sqrt(2 / (n_in + n_out)), size=(n_out, n_in))
        self.b = np.random.normal(loc=0.0, scale=np.sqrt(2 / n_out), size=(n_out,))

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        return {'w': self.w, 'b': self.b}

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        return {}

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        return np.matmul(x, np.transpose(self.w)) + self.b, {'x': x, 'w': self.w, 'b': self.b}

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        return np.matmul(output_grad, cache['w']), {'w': np.matmul(np.transpose(output_grad), cache['x']),
                                                    'b': output_grad.sum(axis=0)}


class BatchNormalization(Layer):

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.gamma = np.ones(n)
        self.beta = np.zeros(n)
        self.global_mean = np.zeros(n)
        self.global_variance = np.zeros(n)
        self.alpha = 0.1

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        return {'gamma': self.gamma, 'beta': self.beta}

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        return {'global_mean': self.global_mean, 'global_variance': self.global_variance}

    def forward(self, x, eps=1e-9):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :param eps: epsilon
        :return: A tuple containing the output value and the cache (y, cache)
        """

        if self.is_training():
            u_b = x.mean(axis=0)
            var_b = x.var(axis=0)

            self.global_mean = (1 - self.alpha) * self.global_mean + self.alpha * u_b
            self.global_variance = (1 - self.alpha) * self.global_variance + self.alpha * var_b

            xhat = (x - u_b) / np.sqrt(var_b + eps)
            cache = {'x': x, 'xhat': xhat, 'u_b': u_b, 'var_b': var_b, 'eps': eps}
        else:
            xhat = (x - self.global_mean) / np.sqrt(self.global_variance + eps)
            cache = {'xhat': xhat}

        y = self.gamma * xhat + self.beta
        return y, cache

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        grad_xhat = output_grad * self.gamma
        grad_var_b = np.sum(grad_xhat * (cache['x'] - cache['u_b']) * -1/2*((cache['var_b'] + cache['eps'])**(-3/2)), axis=0)
        grad_u_b = (-np.sum(grad_xhat / np.sqrt(cache['var_b'] + cache['eps']), axis=0)) + (-2/cache['x'].shape[0] * grad_var_b) * np.sum((cache['x'] - cache['u_b']), axis=0)
        grad_xi = grad_xhat / np.sqrt(cache['var_b'] + cache['eps']) + 2/cache['x'].shape[0] * grad_var_b * (cache['x'] - cache['u_b']) + 1/cache['x'].shape[0] * grad_u_b
        grad_gamma = np.sum(output_grad * cache['xhat'], axis=0)
        grad_beta = np.sum(output_grad, axis=0)

        return grad_xi, {'gamma': grad_gamma, 'beta': grad_beta}


class Sigmoid(Layer):

    def __init__(self):
        super().__init__()

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        return {}

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        return {}

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        return (1 / (1 + np.exp(-x))), {'x': x}

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        ds = self.forward(cache['x'])[0] * (1 - self.forward(cache['x'])[0])
        return output_grad * ds, {}


class ReLU(Layer):

    def __init__(self):
        super().__init__()

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        return {}

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        return {}

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        return np.where(x > 0, x, 0), {'x': x}

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        return np.where(cache['x'] > 0, output_grad, 0), {}



