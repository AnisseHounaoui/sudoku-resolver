from dnn_framework import Layer
import numpy as np

class FullyConnectedLayer(Layer):

    def __init__(self, n_in, n_out):
        super().__init__()

        self.w = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2 / (n_in + n_out)),
                                  size=(n_out, n_in))
        self.b = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2 / n_out),
                                  size=(n_out,))

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        return {'w': self.w, 'b': self.b}

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

    def __init__(self):
        super().__init__()

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        raise NotImplementedError()


class Sigmoid(Layer):

    def __init__(self):
        super().__init__()

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        raise NotImplementedError()


class ReLU(Layer):

    def __init__(self):
        super().__init__()

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        raise NotImplementedError()



